"""Resumable, atomic corpus preparation from published encoders and recorded RGB.

Stages run separately so no generator or optimizer is ever loaded. Components
are shared across target windows. A disk reserve and process lock fail closed.
"""
import argparse
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from gmlib import _paths
from all_scene_dataset import DATA, RecordedStream, pack_bf16, recipe, source_combinations
from prepare_all_scenes import write, lines, TRAIN_PILOT, DEV_PILOT
from prepare_manifest import file_sha, digest
from geometry_policy import SINGLE_SOURCE_POLICY, geometry_path, validate_geometry_policy

HERE=Path(__file__).resolve().parent
CAPTION_ID='Qwen/Qwen3-VL-2B-Instruct'
CAPTION_REVISION='89644892e4d85e24eaac8bacfd4f463576704203'
CAPTION_PROMPT=('Describe directly visible objects, their arrangement and lighting in one or two English sentences, '
    'at most 35 words. Use broad object categories. Do not classify the venue, infer a person\'s role or intentions, '
    'identify plant species or product types, quote writing, or describe future motion or camera commands. '
    'Omit uncertain details. Return only the description.')


def guard(root, reserve_gib):
    if shutil.disk_usage(root).free < (reserve_gib+0.15)*2**30:
        raise OSError('disk reserve reached; no existing files will be deleted')


def npz(path, **arrays):
    temporary=path.with_suffix('.npz.tmp')
    with temporary.open('wb') as f:np.savez_compressed(f,**arrays)
    temporary.replace(path)


def ordered_rows(root):
    rows=list(map(json.loads,(root/'corpus/streams.jsonl').open()))
    primary={}
    for row in rows:
        sid=row['scene_id']
        if sid not in primary or row['raw_frames']>primary[sid]['raw_frames']:primary[sid]=row
    pilot=TRAIN_PILOT|{DEV_PILOT}
    return sorted(rows,key=lambda r:(r['stream_id']!=primary[r['scene_id']]['stream_id'],
        r['scene_id'] not in pilot,digest(['materialization',r['scene_id']]),r['original_start']))


def build_plan(root,round_index=0):
    rows=ordered_rows(root); records=[]
    for row in rows:
        stream=RecordedStream(row);maximum=row['target_start_latent_max']
        rng=np.random.default_rng(int(digest([row['stream_id'],f'round{round_index}'])[:15],16))
        start=int(rng.integers(2,maximum+1)) if maximum>1 else 1
        if row['revisit_pose_candidates'] and (int(digest(row['stream_id'])[:2],16)+round_index)%4==0:
            start=row['revisit_pose_candidates'][round_index%len(row['revisit_pose_candidates'])]['target_start_latent']
        for prefix in sorted({1,start}):
            options=source_combinations(prefix,stream.c2w.numpy())
            # Two prepared variants now; third is available to future offline refresh.
            for variant in range(min(2,len(options))):
                records.append(recipe(row,prefix,variant,stream.c2w.numpy()))
    path=root/f'sampler/round_{round_index:03d}_recipes.jsonl'
    if path.exists():
        old=list(map(json.loads,path.open()))
        if old!=records:raise ValueError('round already fixed; create a new offline round')
    else:lines(path,records)
    write(root/'active_round.json',dict(round=round_index,recipes=str(path),sha256=file_sha(path)))
    print('PLAN',len(rows),'streams',len(records),'render recipes',flush=True)
    return records


def geometry_manifest(root, round_index=None):
    """Default to the active immutable revision, including round-0 expansions."""
    if round_index is not None:
        return root/f'sampler/round_{round_index:03d}_recipes.jsonl',round_index
    active=root/'active_round.json'
    if not active.exists():return root/'sampler/round_000_recipes.jsonl',0
    record=json.loads(active.read_text());path=Path(record['recipes'])
    if file_sha(path)!=record['sha256']:raise ValueError('active recipe manifest digest mismatch')
    return path,record['round']


class Producer:
    def __init__(self,stage,root):
        self.stage=stage;self.root=root
        self.provenance=dict(stage=stage,producer_code_sha256=file_sha(Path(__file__)),
            loader_code_sha256=file_sha(HERE/'all_scene_dataset.py'),published_models_only=True,
            adapted_generator_loaded=False,resolution=[256,448])
        (root/'code').mkdir(exist_ok=True)
        for source in (Path(__file__),HERE/'all_scene_dataset.py'):
            shutil.copy2(source,root/'code'/f'{file_sha(source)}.py')
        if stage in ('vae','geometry'):
            from gmlib.models.wan22_frame_renderer import load_wan22_vae
            self.vae=load_wan22_vae(device='cuda',dtype=torch.bfloat16).eval()
            self.provenance['vae_sha256']=file_sha(_paths.WAN22_DIR/'Wan2.2_VAE.pth')
            self.provenance['stream_code_sha256']=file_sha(_paths.GMLIB/'models/vae_stream.py')
        if stage in ('anchor','geometry'):
            from gmlib.training.gate_omega_recovery import PinnedDA3Predictor, DA3_MODEL
            self.da3=PinnedDA3Predictor(offload_after_call=False)
            self.provenance['da3_digest']=digest([(p.name,file_sha(p)) for p in sorted(DA3_MODEL.glob('*.safetensors'))])
        if stage=='anchor':
            from gmlib.training.gate_omega_recovery import PinnedMoGePredictor,MOGE_MODEL
            self.moge=PinnedMoGePredictor()
            self.provenance['moge_sha256']=file_sha(MOGE_MODEL)
        if stage=='geometry':
            from foundation_renderer import FoundationRenderer
            self.renderer=FoundationRenderer(device='cuda')
            self.provenance['renderer_code_sha256']=file_sha(HERE/'foundation_renderer.py')
            self.provenance['depth_code_sha256']=file_sha(HERE/'foundation_depth.py')
            self.provenance['single_source_policy']=SINGLE_SOURCE_POLICY
            self.provenance['geometry_policy_code_sha256']=file_sha(HERE/'geometry_policy.py')
            for source in (HERE/'foundation_renderer.py',HERE/'foundation_depth.py',HERE/'geometry_policy.py'):
                shutil.copy2(source,root/'code'/f'{file_sha(source)}.py')
        if stage=='caption':
            from transformers import Qwen3VLForConditionalGeneration,AutoProcessor
            from huggingface_hub import snapshot_download
            path=snapshot_download(CAPTION_ID,revision=CAPTION_REVISION,local_files_only=True,
                allow_patterns=['*.json','*.safetensors','*.txt','*.jinja'])
            self.captioner=Qwen3VLForConditionalGeneration.from_pretrained(path,torch_dtype=torch.bfloat16,
                attn_implementation='sdpa').cuda().eval()
            self.processor=AutoProcessor.from_pretrained(path)
            self.provenance.update(model=CAPTION_ID,revision=CAPTION_REVISION,prompt=CAPTION_PROMPT,
                caption_model_is_smaller_than_Memorizon_reference=True)
        if stage=='text':
            from gmlib.diffsynth.core.loader.model import load_model
            from gmlib.diffsynth.models.wan_video_text_encoder import HuggingfaceTokenizer,WanTextEncoder
            self.tokenizer=HuggingfaceTokenizer(str(_paths.WAN22_DIR/'google/umt5-xxl'),seq_len=None,
                clean='whitespace',local_files_only=True)
            path=_paths.WAN22_DIR/'models_t5_umt5-xxl-enc-bf16.pth'
            self.encoder=load_model(WanTextEncoder,str(path),torch_dtype=torch.bfloat16,device='cuda').eval()
            self.provenance['encoder_sha256']=file_sha(path)

    @torch.no_grad()
    def run(self,row,r=None):
        directory=self.root/'components'/row['stream_id'];directory.mkdir(parents=True,exist_ok=True)
        output=directory/f'{self.stage}.npz'
        metadata=dict(scene_id=row['scene_id'],stream_id=row['stream_id'],split=row['split'],
            provenance=self.provenance,source_transforms_sha256=row['transforms_sha256'])
        if self.stage=='vae':
            from gmlib.models.vae_stream import VaeStream
            stream=RecordedStream(row);encoder=VaeStream(self.vae,offload_device='cpu');parts=[]
            def pixels(ids):
                return torch.stack([stream.tensor(i) for i in ids],dim=1)[None].to('cuda',torch.bfloat16)
            parts.append(encoder.encode_chunk(pixels([0])).float().cpu())
            for start in range(1,row['encoded_frames'],32):
                parts.append(encoder.encode_chunk(pixels(range(start,min(start+32,row['encoded_frames'])))).float().cpu())
            z=torch.cat(parts,dim=2).numpy()
            assert z.shape==(1,48,row['latent_count'],16,28) and np.isfinite(z).all()
            # Decode-check the non-grid tail too; keep it in raw corpus, never pad it.
            for i in range(row['encoded_frames'],row['raw_frames']):stream.frame(i)
            pixels_hash=hashlib.sha256()
            for path in stream.paths:pixels_hash.update(bytes.fromhex(file_sha(path)))
            npz(output,latents_bf16=pack_bf16(z),K=stream.K.numpy(),c2w=stream.c2w.numpy())
            metadata.update(encoded_frames=row['encoded_frames'],full_stream_RGB_decode_checked=True,
                source_pixel_hash_digest=pixels_hash.hexdigest(),storage='lossless_bfloat16_bits_uint16',
                encoded_tail_pending=row['tail_frames_not_on_VAE_grid'],appearance_origin='recorded_stream_encode')
        elif self.stage=='anchor':
            from gmlib.training.gate_omega_recovery import build_lyra_seed_geometry
            from foundation_contract import normalize_cameras,fit_registered_scale,CONTRACT
            stream=RecordedStream(row)
            seed=build_lyra_seed_geometry(image_uint8=stream.frame(0),anchor_c2w=stream.c2w[0],target_size=(256,448),
                da3_predictor=self.da3,moge_predictor=self.moge,da3_weights_sha256=self.provenance['da3_digest'],
                moge_weights_sha256=self.provenance['moge_sha256'])
            depth=(seed.aligned_depth/float(seed.gauge.scale)).float();ids=[0,4,8,12,16]
            relative=normalize_cameras(stream.c2w,stream.c2w[0],1.)
            prediction=self.da3(np.stack([stream.frame(i) for i in ids]),
                np.repeat(stream.K.numpy()[None],5,0),torch.linalg.inv(relative[ids]).numpy())
            mask=seed.mask>.5
            if prediction.valid_masks is not None:mask &= prediction.valid_masks[0]>.5
            scale,fit=fit_registered_scale(prediction.depths[0],depth,mask)
            gauge=digest([row['stream_id'],seed.gauge.manifest(),scale,ids,CONTRACT])
            npz(output,depth=depth.numpy(),confidence=seed.mask.numpy(),K=stream.K.numpy())
            metadata.update(gauge_id=gauge,registered_units_per_anchor_depth=scale,scale_fit=fit,
                camera_units='anchor_median_depth',seed_frame_ids=[0],camera_label_frame_ids=ids,
                camera_label_may_use_future_RGB=True,camera_label_is_runtime_appearance=False,
                seed_gauge=seed.gauge.manifest(),seed_scale=seed.scale.manifest(),metric_ground_truth=False)
        elif self.stage=='caption':
            stream=RecordedStream(row);im=Image.fromarray(stream.frame(0))
            messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':CAPTION_PROMPT}]}]
            prompt=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            inputs=self.processor(text=[prompt],images=[im],return_tensors='pt').to('cuda')
            result=self.captioner.generate(**inputs,max_new_tokens=96,do_sample=False)
            caption=self.processor.batch_decode(result[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0].strip()
            if not caption or len(caption)>2000:raise ValueError('invalid caption')
            metadata.update(caption=caption,input_frame_ids=[0],anchor_pixel_sha256=file_sha(stream.paths[0]),
                future_RGB_used=False,human_verified=False)
            return metadata
        elif self.stage=='text':
            caption=json.loads((directory/'caption.json').read_text())['caption']
            ids,mask=self.tokenizer(caption,return_mask=True,add_special_tokens=True)
            if ids.shape[1]>256:raise ValueError('unexpectedly long caption')
            native=self.encoder(ids.cuda(),mask.cuda()).float().cpu().numpy()
            assert np.isfinite(native).all()
            npz(output,text_bf16=pack_bf16(native))
            metadata.update(caption_sha256=hashlib.sha256(caption.encode()).hexdigest(),
                caption_metadata_sha256=file_sha(directory/'caption.json'),token_count=ids.shape[1],
                padding='none_trimmed_native_sequence')
        elif self.stage=='geometry':
            from foundation_contract import normalize_cameras
            from foundation_depth import causal_depth
            from gmlib.training.gate_omega_recovery import _reliability
            from gmlib.models.vae_stream import VaeStream
            stream=RecordedStream(row);annotation=json.loads((directory/'anchor.json').read_text())
            with np.load(directory/'anchor.npz') as z:
                anchor_depth=torch.from_numpy(z['depth'].copy());anchor_mask=torch.from_numpy(z['confidence'].copy())
            poses=normalize_cameras(stream.c2w,stream.c2w[0],annotation['registered_units_per_anchor_depth'])
            ids=r['source_frame_ids'];boundary=r['prefix_frames']-1
            assert max(ids)<=boundary
            if len(ids)>1:
                depths,masks,depth_fit=causal_depth(self.da3,np.stack([stream.frame(i) for i in ids]),
                    stream.K,poses[ids],anchor_depth,anchor_mask)
            else:depth_fit={'anchor_only':True}
            sources=[]
            for j,i in enumerate(ids):
                depth=anchor_depth if i==0 else depths[j]
                confidence=anchor_mask if i==0 else _reliability(depth)*masks[j]
                sources.append(dict(frame_id=i,rgb=stream.tensor(i),depth=depth,confidence=confidence,
                    c2w=poses[i],gauge_id=annotation['gauge_id']))
            rendered=self.renderer.render(sources,poses[boundary+1:boundary+17],stream.K,(256,448),seed=0,
                camera_units='anchor_median_depth',gauge_id=annotation['gauge_id'],history_boundary=boundary)
            video=rendered['rgb'].permute(1,0,2,3)[None];video=torch.cat([video[:,:,:1],video],dim=2)
            render=VaeStream(self.vae).encode_chunk(video.cuda().to(torch.bfloat16))[:,:,1:].float().cpu().numpy()
            meta=torch.stack([F.interpolate(rendered[k].reshape(4,4,1,256,448).mean(1),size=(16,28),mode='area')[:,0]
                for k in ('valid','conf')])[None].numpy()
            assert np.isfinite(render).all() and np.isfinite(meta).all()
            output=geometry_path(self.root,r,'.npz');output.parent.mkdir(exist_ok=True)
            npz(output,render_bf16=pack_bf16(render),metadata=meta)
            metadata.update(recipe=r,coverage=float(rendered['valid'].mean()),depth_fit=depth_fit,
                logical_source_count=len(sources),renderer_input_view_count=max(2,len(sources)),
                anchor_metadata_sha256=file_sha(directory/'anchor.json'),
                source_RGB_hashes=[file_sha(stream.paths[i]) for i in ids],
                target_RGB_read=False,geometry_ground_truth=False)
        metadata.update(path=str(output),sha256=file_sha(output),bytes=output.stat().st_size)
        return metadata


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['plan','vae','anchor','caption','text','geometry'])
    p.add_argument('--root',type=Path,default=DATA);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--round',type=int,default=None,help='Explicit original round; default geometry follows active revision')
    p.add_argument('--reserve-gib',type=float,default=8);p.add_argument('--retry-errors',action='store_true')
    args=p.parse_args();root=args.root;root.mkdir(exist_ok=True)
    if args.stage=='plan':build_plan(root,0 if args.round is None else args.round);return
    lock=(root/'producer.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.manual_seed(20261009)
    rows=ordered_rows(root)
    if args.limit:rows=rows[:args.limit]
    by_id={r['stream_id']:r for r in rows}
    if args.stage=='geometry':
        manifest,round_index=geometry_manifest(root,args.round)
        jobs=[(by_id[r['stream_id']],r) for r in map(json.loads,manifest.open()) if r['stream_id'] in by_id]
    else:jobs=[(r,None) for r in rows]
    status=dict(stage=args.stage,status='running',jobs=len(jobs),completed=0,skipped=0,failed=0,
        dependency_missing=0,limit=args.limit,round=round_index if args.stage=='geometry' else args.round,
        reserve_gib=args.reserve_gib,started_unix=time.time())
    if args.stage=='geometry':status.update(recipes=str(manifest),recipes_sha256=file_sha(manifest))
    path=root/f'progress_{args.stage}.json';producer=None;consecutive_failures=0
    write(path,status)
    for row,r in jobs:
        destination=(geometry_path(root,r,'.json') if r else root/'components'/row['stream_id']/f'{args.stage}.json')
        destination.parent.mkdir(parents=True,exist_ok=True)
        error_path=destination.with_suffix('.error.json')
        if destination.exists():
            meta=json.loads(destination.read_text())
            if r:validate_geometry_policy(meta,r)
            if 'path' not in meta or (Path(meta['path']).exists() and file_sha(meta['path'])==meta['sha256']):
                status['skipped']+=1;continue
            raise ValueError('existing component digest mismatch; explicit repair required')
        if error_path.exists() and not args.retry_errors:status['failed']+=1;continue
        dependency={'text':'caption','geometry':'anchor'}.get(args.stage)
        if dependency and not (root/'components'/row['stream_id']/f'{dependency}.json').exists():
            status['dependency_missing']+=1;continue
        try:
            guard(root,args.reserve_gib)
            if producer is None:producer=Producer(args.stage,root)
        except Exception as exc:
            status.update(status='stopped_setup_or_disk_error',error=f'{type(exc).__name__}: {exc}')
            write(path,status);raise
        start=time.monotonic()
        try:
            meta=producer.run(row,r);meta['wall_seconds']=time.monotonic()-start
            write(destination,meta);status['completed']+=1;consecutive_failures=0
            if error_path.exists():error_path.unlink()
            print('DONE',args.stage,row['scene_id'][:8],row['stream_id'],round(meta['wall_seconds'],3),meta.get('bytes',0),flush=True)
        except Exception as exc:
            status['failed']+=1;consecutive_failures+=1
            write(error_path,dict(scene_id=row['scene_id'],stream_id=row['stream_id'],stage=args.stage,
                error=f'{type(exc).__name__}: {exc}',recipe=r))
            print('FAILED',args.stage,row['scene_id'][:8],str(exc),flush=True)
            gc.collect();torch.cuda.empty_cache()
            if consecutive_failures>=3:
                status['status']='stopped_repeated_errors';write(path,status);raise
        status['wall_seconds']=time.time()-status['started_unix'];write(path,status)
    status.update(status='complete' if not status['failed'] and not status['dependency_missing'] else 'complete_with_unusable_or_pending_components',
        wall_seconds=time.time()-status['started_unix'],free_bytes=shutil.disk_usage(root).free)
    write(path,status);print(json.dumps(status),flush=True)


if __name__=='__main__':main()
