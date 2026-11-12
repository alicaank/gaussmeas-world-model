"""Check actual corpus, split and causal recipe contracts; report readiness."""
from collections import Counter,defaultdict
import argparse
import json
from pathlib import Path
import time
import numpy as np
from all_scene_dataset import DATA,cached_pair,unpack_bf16
from prepare_manifest import file_sha
from prepare_all_scenes import write,lines
from geometry_policy import geometry_path, validate_geometry_policy, SINGLE_SOURCE_POLICY


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DATA)
    p.add_argument('--verify-components',action='store_true');p.add_argument('--smoke-pairs',type=int,default=0)
    a=p.parse_args();root=a.root
    split=json.loads((root/'splits.json').read_text())
    assert file_sha(split['final_test_path'])==split['final_test_sha256']
    excluded=set(split['test_scene_ids'])|set(split['historical_evaluation_scene_ids'])
    dev=set(split['development_scene_ids'])
    rows=list(map(json.loads,(root/'corpus/streams.jsonl').open()));by_id={r['stream_id']:r for r in rows}
    active_path=root/'active_round.json'
    active=json.loads(active_path.read_text()) if active_path.exists() else dict(round=0,recipes=str(root/'sampler/round_000_recipes.jsonl'))
    recipes=list(map(json.loads,Path(active['recipes']).open()))
    assert not excluded & {r['scene_id'] for r in rows}
    assert not dev & {r['scene_id'] for r in rows if r['split']=='train'}
    assert len(by_id)==len(rows)
    groups=defaultdict(list)
    for r in recipes:
        row=by_id[r['stream_id']];start=r['prefix_latents'];boundary=4*(start-1)
        assert r['scene_id']==row['scene_id'] and r['split']==row['split']
        assert 1<=start<=row['target_start_latent_max']
        assert r['prefix_frames']==boundary+1 and max(r['source_frame_ids'])<=boundary
        assert r['target_latent_indices']==list(range(start,start+4))
        assert set(r['archive_indices']).isdisjoint(r['history_latent_indices'])
        assert all(0<i<max(1,start-8) for i in r['archive_indices'])
        assert r['target_original_frame_range']==[row['original_start']+boundary+1,row['original_start']+boundary+17]
        assert r['target_original_frame_range'][1]<=row['original_start']+row['encoded_frames']
        assert r['future_RGB_in_condition'] is False and r['appearance_origin']=='recorded_only'
        groups[r['group_id']].append(r)
    for g in groups.values():assert len({r['flow_pair_seed'] for r in g})==1
    ready_streams=set();component_counts=Counter();failed=Counter();bytes_by_stage=Counter();times=defaultdict(list)
    for row in rows:
        directory=root/'components'/row['stream_id'];complete=True
        for stage in ('vae','anchor','caption','text'):
            path=directory/f'{stage}.json'
            if not path.exists():
                complete=False
                if path.with_suffix('.error.json').exists():failed[stage]+=1
                continue
            meta=json.loads(path.read_text());component_counts[stage]+=1
            assert meta['scene_id']==row['scene_id'] and meta['split']==row['split']
            if a.verify_components and 'path' in meta:assert file_sha(meta['path'])==meta['sha256']
            bytes_by_stage[stage]+=meta.get('bytes',0);times[stage].append(meta['wall_seconds'])
        if complete:ready_streams.add(row['stream_id'])
    ready=[];smoked=0
    for r in recipes:
        path=geometry_path(root,r,'.json')
        if not path.exists():
            if path.with_suffix('.error.json').exists():failed['geometry']+=1
            continue
        meta=json.loads(path.read_text());assert meta['recipe']==r
        validate_geometry_policy(meta,r)
        component_counts['geometry']+=1;bytes_by_stage['geometry']+=meta['bytes'];times['geometry'].append(meta['wall_seconds'])
        if a.verify_components:assert file_sha(meta['path'])==meta['sha256']
        if r['stream_id'] in ready_streams:
            ready.append(r)
            if smoked<a.smoke_pairs:
                conditions,target=cached_pair(by_id[r['stream_id']],r,root)
                assert tuple(target.shape)==(1,48,4,16,28)
                assert np.isfinite(target.numpy()).all()
                clean=conditions['clean'];missing=conditions['memory_missing']
                assert not (missing.roles==2).any() and not (missing.roles==3).any()
                assert clean.appearance.shape[2]+4==len(clean.roles)
                assert clean.gauge_id==missing.gauge_id
                smoked+=1
    lines(root/'sampler/ready_recipes.jsonl',ready)
    expected={stage:len(rows) for stage in ('vae','anchor','caption','text')};expected['geometry']=len(recipes)
    complete=all(component_counts[s]==n for s,n in expected.items())
    out=dict(status='complete_components' if complete else 'materialization_in_progress_or_pending',
        active_round=active['round'],recipes_sha256=file_sha(active['recipes']),single_source_policy=SINGLE_SOURCE_POLICY,
        inspected_unix=time.time(),indexed_streams=len(rows),indexed_recipes=len(recipes),causal_recipes_verified=len(recipes),
        components=dict(component_counts),expected=expected,failed=dict(failed),ready_streams=len(ready_streams),
        ready_recipes=len(ready),ready_scenes=len({r['scene_id'] for r in ready}),
        bytes_by_stage=dict(bytes_by_stage),smoke_pairs_passed=smoked,
        seconds_per_job={k:dict(first=v[0],median=float(np.median(v)),mean=float(np.mean(v))) for k,v in times.items()},
        measured_cost_projection_bytes={k:int(bytes_by_stage[k]/component_counts[k]*expected[k]) for k in bytes_by_stage},
        measured_cost_projection_hours={k:float(np.median(v))*expected[k]/3600 for k,v in times.items()},
        final_test_unchanged=True,old_adapted_models_used=False,model_training_launched=False,
        complete_cache_is_not_model_quality_qualification=True,
        limitations=['full_RGB_decode_only_verified_for_finished_VAE_streams','captions_not_all_human_verified',
            'pose_revisit_candidates_not_surface_visibility_truth','cross_video_same_location_duplicates_not_certified'])
    write(root/'readiness.json',out);print(json.dumps(out,indent=2))


if __name__=='__main__':main()
