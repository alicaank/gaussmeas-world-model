"""Shared full-corpus storage, causal recipe sampling and exact frame access."""
from collections import defaultdict
import io
import json
from pathlib import Path
import random
import numpy as np
import torch
from PIL import Image
from gaussverse.data.dl3dv import crop_and_resize, scale_intrinsics
from prepare_manifest import digest, file_sha
from geometry_policy import geometry_path, validate_geometry_policy

HERE=Path(__file__).resolve().parent
DATA=HERE/'dataset_v3'


def pack_bf16(x):
    """Lossless storage of already-bfloat16 values, never a new quantization."""
    x=np.asarray(x,dtype=np.float32); bits=x.view(np.uint32)
    if np.any(bits & 65535):
        raise ValueError('input is not exactly bfloat16; refusing lossy storage')
    return (bits >> 16).astype(np.uint16)


def unpack_bf16(x):
    if x.dtype!=np.uint16:raise ValueError('BF16 bit storage expected')
    return (x.astype(np.uint32) << 16).view(np.float32)


class RecordedStream:
    def __init__(self, row):
        self.row=row;path=Path(row['scene_path'])/'transforms.json'
        if file_sha(path)!=row['transforms_sha256']:raise ValueError('source transforms changed')
        td=json.loads(path.read_text());frames=sorted(td['frames'],key=lambda f:f['file_path'])
        frames=frames[row['original_start']:row['original_end_exclusive']]
        if len(frames)!=row['raw_frames']:raise ValueError('source frame count changed')
        self.paths=[Path(row['image_root'])/Path(f['file_path']).name for f in frames]
        if not all(p.is_file() for p in self.paths):raise ValueError('indexed frame now missing')
        gl=np.asarray([f['transform_matrix'] for f in frames],np.float32)
        self.c2w=torch.from_numpy(gl @ np.diag([1.,-1.,-1.,1.]).astype(np.float32))
        self.K=torch.from_numpy(scale_intrinsics(*(td[k] for k in ('fl_x','fl_y','cx','cy','w','h')),448,256))

    def frame(self, index):
        with Image.open(self.paths[index]) as im:
            return np.array(crop_and_resize(im.convert('RGB'),448,256),copy=True)

    def tensor(self, index):
        return torch.from_numpy(self.frame(index)).permute(2,0,1).float()/127.5-1


def source_combinations(start, poses):
    """At most three frozen choices. All source RGB precedes requested target.

    Variant 0 matches the validated live pilot policy. Variant 1 is a CAMERA-only
    proximity proxy, not depth/covisibility truth. Variant 2 is temporal coverage.
    The same function can operate on commanded poses at deployment.
    """
    candidates=list(range(1,max(1,start-8))); combinations=[]
    if not candidates:return [[]]
    combinations.append(list(dict.fromkeys([1,start-9])))
    if len(candidates)>1:
        p=np.asarray(poses);current=p[4*(start-1)]
        centers=p[[4*i for i in candidates],:3,3]
        distances=np.linalg.norm(centers-current[:3,3],axis=1)
        scale=max(float(np.median(distances)),1e-6)
        angles=np.arccos(np.clip(p[[4*i for i in candidates],:3,2]@current[:3,2],-1,1))
        ranked=np.argsort(angles+distances/scale,kind='stable')
        combinations.append(sorted(candidates[i] for i in ranked[:2]))
        combinations.append(sorted(set([candidates[len(candidates)//3],candidates[-1]])))
    return [list(x) for x in dict.fromkeys(tuple(x) for x in combinations)]


def recipe(row, start, variant=0, poses=None):
    if row['split'] not in ('train','development'):raise ValueError('evaluation reservation')
    if not 1<=start<=row['target_start_latent_max']:raise ValueError('target beyond stream')
    if poses is None:poses=RecordedStream(row).c2w.numpy()
    options=source_combinations(start,poses)
    if not 0<=variant<len(options):raise ValueError('invalid source variant')
    archive=options[variant];boundary=4*(start-1)
    source_ids=list(dict.fromkeys([0,*[4*i for i in archive],boundary,
        *range(max(0,boundary-12),boundary,4)]))
    assert max(source_ids)<=boundary
    assert all(0<i<max(1,start-8) for i in archive)
    group=digest([row['stream_id'],start,'forward'])
    return dict(group_id=group,recipe_id=digest([group,variant,'all_scene_sources.v1']),
        scene_id=row['scene_id'],stream_id=row['stream_id'],split=row['split'],
        prefix_latents=start,prefix_frames=boundary+1,
        target_latent_indices=list(range(start,start+4)),
        target_original_frame_range=[row['original_start']+boundary+1,row['original_start']+boundary+17],
        history_latent_indices=list(range(max(1,start-8),start)),archive_indices=archive,
        source_frame_ids=source_ids,source_variant=variant,
        flow_pair_seed=int(digest([group,20261009])[:15],16),
        task='first_call' if start==1 else 'recorded_forward',synthetic=False,
        future_RGB_in_condition=False,source_overlap_verified=False,
        appearance_origin='recorded_only',decoder_prefix='full_recorded_causal_prefix',
        camera_origin='registered_recording_with_separate_anchor_scale_annotation')


class SceneBalancedSampler:
    """Uniform scenes, then streams, then a task/prefix; deterministic by seed."""
    def __init__(self, rows, split='train', seed=20261009):
        self.scenes=defaultdict(list);self.rng=random.Random(seed)
        for row in rows:
            if row['split']==split:self.scenes[row['scene_id']].append(row)
        self.ids=sorted(self.scenes)
        if not self.ids:raise ValueError('no scenes')

    def draw(self):
        rows=self.scenes[self.rng.choice(self.ids)];row=self.rng.choice(rows)
        maximum=row['target_start_latent_max'];u=self.rng.random()
        if u<.10:start=1
        elif u<.25 and row['revisit_pose_candidates']:
            start=self.rng.choice(row['revisit_pose_candidates'])['target_start_latent']
        else:
            # Equal early/middle/late prefix ranges; never clip source timeline.
            bucket=self.rng.randrange(3)
            candidates=list(range(1+bucket*maximum//3,1+(bucket+1)*maximum//3))
            start=self.rng.choice(candidates) if candidates else 1
        return row,start


def cached_pair(row, r, root=DATA, *, native_archive_indices=None):
    """Build clean/missing pair; native selection does not alter frozen geometry.

    r['archive_indices'] belongs to the original render recipe. Override only
    the native reader, leaving recipe/cache equality and source_frame_ids intact.
    """
    from foundation_contract import build_condition, normalize_cameras
    directory=root/'components'/row['stream_id']
    vae_meta=json.loads((directory/'vae.json').read_text())
    annotation=json.loads((directory/'anchor.json').read_text())
    geometry_meta=json.loads(geometry_path(root,r,'.json').read_text())
    if geometry_meta['recipe']!=r:raise ValueError('recipe/cache mismatch')
    validate_geometry_policy(geometry_meta,r)
    for path,meta in [(directory/'vae.npz',vae_meta),(directory/'anchor.npz',annotation),
                      (geometry_path(root,r,'.npz'),geometry_meta)]:
        if file_sha(path)!=meta['sha256']:raise ValueError('component hash mismatch')
    with np.load(directory/'vae.npz') as z:
        prefix=torch.from_numpy(unpack_bf16(z['latents_bf16']))
        K=torch.from_numpy(z['K'].copy());poses=torch.from_numpy(z['c2w'].copy())
    with np.load(geometry_path(root,r,'.npz')) as z:
        render=torch.from_numpy(unpack_bf16(z['render_bf16']))
        meta=torch.from_numpy(z['metadata'].copy())
    canonical=normalize_cameras(poses,poses[0],annotation['registered_units_per_anchor_depth'])
    start=r['prefix_latents'];boundary=r['prefix_frames'];conditions={}
    native=r['archive_indices'] if native_archive_indices is None else list(native_archive_indices)
    for family in ('clean','memory_missing'):
        conditions[family]=build_condition(prefix=prefix[:,:,:start],observed_c2w=canonical[:boundary],
            requested_c2w=canonical[boundary:boundary+16],K=K,gauge_id=annotation['gauge_id'],
            camera_units='anchor_median_depth',archive_indices=native if family=='clean' else (),
            render=render if family=='clean' else None,render_metadata=meta if family=='clean' else None)
    return conditions,prefix[:,:,start:start+4]


class PreparedSceneSampler:
    """Training-facing reader; uniform scenes and matched clean/missing pairs.

    Refuses a partial corpus unless the caller explicitly selects that mode.
    This keeps a partly materialized corpus from silently biasing training.
    """
    def __init__(self,root=DATA,split='train',seed=20261009,allow_partial=False,
                 native_archive_max=None,native_archive_audit=None,planning_only=False):
        self.root=Path(root);self.rng=random.Random(seed);self.planning_only=planning_only
        readiness=json.loads((self.root/'readiness.json').read_text())
        active=dict(recipes=str(self.root/'sampler/round_000_recipes.jsonl'))
        if (self.root/'active_round.json').exists():
            active=json.loads((self.root/'active_round.json').read_text())
            if readiness.get('recipes_sha256')!=active['sha256']:
                raise RuntimeError('active offline round changed; rebuild and audit its caches first')
        if readiness['status']!='complete_components' and not allow_partial and not planning_only:
            raise RuntimeError('corpus materialization incomplete; partial training requires explicit selection')
        self.rows={r['stream_id']:r for r in map(json.loads,(self.root/'corpus/streams.jsonl').open())}
        self.groups=defaultdict(lambda:defaultdict(list))
        self.recipe_manifest_path=Path(active['recipes']) if planning_only else self.root/'sampler/ready_recipes.jsonl'
        for r in map(json.loads,self.recipe_manifest_path.open()):
            if r['split']==split:self.groups[r['scene_id']][r['group_id']].append(r)
        self.scenes=sorted(self.groups)
        if not self.scenes:raise ValueError('no ready scenes in requested split')
        self.native_index=None
        if split=='train':
            from native_archive import NativeArchiveIndex,DEFAULT_AUDIT
            policy_path=self.root/'sampler/native_archive_policy.json'
            policy=json.loads(policy_path.read_text()) if policy_path.exists() else {'training_budget_max':6}
            self.native_archive_max=policy['training_budget_max'] if native_archive_max is None else native_archive_max
            if not isinstance(self.native_archive_max,int) or not 0<=self.native_archive_max<=6:
                raise ValueError('native read budget must be 0..6')
            self.native_index=NativeArchiveIndex(native_archive_audit or DEFAULT_AUDIT,self.root)
        elif native_archive_max is not None:
            raise ValueError('v2 selection is train-only; development retrieval needs its own policy')

    def choose_recipe(self):
        """Freeze identities before comparing archive budgets on the same rows."""
        sid=self.rng.choice(self.scenes);groups=list(self.groups[sid].values())
        first=[g for g in groups if g[0]['prefix_latents']==1]
        continuation=[g for g in groups if g[0]['prefix_latents']>1]
        choices=first if self.rng.random()<.1 or not continuation else continuation
        if not choices:choices=groups
        return self.rng.choice(self.rng.choice(choices))

    def load(self,r,update):
        if self.planning_only:raise RuntimeError('planning-only sampler cannot load unprepared training data')
        row=self.rows[r['stream_id']]
        native=(self.native_index.select(row,r['prefix_latents'],self.native_archive_max)
                if self.native_index is not None else None)
        conditions,target=cached_pair(row,r,self.root,
            native_archive_indices=native['archive_indices'] if native is not None else None)
        directory=self.root/'components'/row['stream_id']
        meta=json.loads((directory/'text.json').read_text())
        if file_sha(directory/'text.npz')!=meta['sha256']:raise ValueError('text hash mismatch')
        with np.load(directory/'text.npz') as z:text=torch.from_numpy(unpack_bf16(z['text_bf16']))
        return dict(conditions=conditions,target=target,text=text,recipe=r,
            native_selection=native,native_reader_policy='v2_training' if native is not None else 'original_development_recipe',
            flow_seed=int(digest([update,r['group_id'],'flow'])[:15],16),
            family_weights={'clean':.8,'memory_missing':.2},
            same_noise_time_and_text_for_pair=True)

    def draw(self,update):
        return self.load(self.choose_recipe(),update)
