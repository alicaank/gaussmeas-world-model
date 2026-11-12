"""Full-scene CPU index, immutable test reservation and small reference pilot.

Layer 1: scene index (all recorded frames via pinned transforms + image root).
Layer 2: bounded offline source combinations, not materialized tensors yet.
Layer 3: task/family sampler references; synthetic tasks fail closed pending QA.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import time

import numpy as np
from prepare_manifest import ROOT, DEFAULT_SOURCE, DEFAULT_EXCLUSIONS, IMAGE_DIRS, digest, file_sha, FAMILIES

HERE = Path(__file__).resolve().parent
OUT = HERE / 'data_v2'


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')


def lock_split(entries, out, seed, heldout=()):
    path = out / 'locked_test.json'
    if path.exists():
        saved = json.loads(path.read_text())
        ids = {r['scene_id'] for r in saved['scenes']}
        if saved['seed'] != seed or len(ids) != 24 or not ids <= {e['scene_id'] for e in entries}:
            raise ValueError('existing split differs; never silently replace a locked test')
        if not set(heldout) <= set(saved['excluded_legacy_scene_ids']):
            raise ValueError('new upstream heldout exclusions require a versioned split audit')
        return saved
    qa_path = ROOT / 'analysis/plan_base_restart_20261008/scene_qa.jsonl'
    qa = {r['scene_id']: r for r in map(json.loads, qa_path.open())}
    old = json.loads(DEFAULT_EXCLUSIONS.read_text())
    excluded = set(old['excluded_evaluation_scenes']) | set(heldout)
    # Cache presence is a conservative exclusion for a NEW final test, even
    # though cache creation alone does not imply training or human inspection.
    used = {s for s, q in qa.items() if q.get('cached') or q.get('reserved')}
    used |= {r['scene_id'] for r in json.loads((HERE / 'data/splits.json').read_text())['training_scenes']}
    prior_paths = [DEFAULT_EXCLUSIONS, HERE / 'data/splits.json', qa_path]
    # Existing local outputs can be identified by full scene hashes or stems.
    # Do not count external source snapshots as our experimental exposure.
    stems = set()
    for root in (ROOT / 'analysis', ROOT / 'training_runs'):
        import os
        for parent, dirs, names in os.walk(root):
            dirs[:] = [d for d in dirs if d not in ('runtime', 'repos', 'sources', '__pycache__', '.git', 'data_v2')]
            for name in [Path(parent).name, *names]:
                import re
                stems.update(re.findall(r'(?<![0-9a-f])[0-9a-f]{8}(?![0-9a-f])', name))
    used |= {e['scene_id'] for e in entries if e['scene_id'][:8] in stems}
    eligible = [e for e in entries if e['scene_id'] not in used | excluded and not qa[e['scene_id']].get('qa_error')]
    bins = defaultdict(list)
    thresholds = np.quantile([qa[e['scene_id']]['total_rot_deg'] for e in eligible], [1/3, 2/3]).tolist()
    for e in eligible:
        q = qa[e['scene_id']]
        b = f"glass_proxy_{int(q['glass_vs_matte'] >= .6)}_rotation_{int(np.searchsorted(thresholds, q['total_rot_deg']))}"
        bins[b].append(e)
    selected = []
    for b, rows in sorted(bins.items()):
        rows.sort(key=lambda e: digest([seed, 'test', e['scene_id']]))
        if len(rows) < 4:
            raise RuntimeError(f'insufficient test candidates: {b}')
        selected.extend(dict(scene_id=e['scene_id'], scene_path=e['scene_path'], stratum=b,
                             source_video_id=e['scene_id'], identity_scope='DL3DV_hash_shared_across_resolutions') for e in rows[:4])
    assert len(selected) == 24
    result = dict(schema='base_restart.locked_test.v2', seed=seed, selection='four_per_proxy_stratum_deterministic_hash',
        scenes=selected, excluded_legacy_scene_ids=sorted(excluded),
        prior_exposure_excluded_count=len(used), eligible_count=len(eligible),
        provenance=[dict(path=str(p), sha256=file_sha(p)) for p in prior_paths],
        rotation_stratum_thresholds=thresholds,
        limitations=['CLIP strata are proxies, not verified material labels.',
                    'Inspection audit excludes known caches, old pilot and scene-named artifacts; not a claim about human memory.',
                    'DL3DV scene/video hash excludes resolution duplicates; cross-hash same-location duplicates are not certified.'],
        selection_uses_model_scores=False, test_outputs_generated=False)
    write(path, result)
    return result


def pose_stats(poses):
    p = np.asarray(poses, dtype=np.float64)
    if not np.isfinite(p).all():
        raise ValueError('nonfinite poses')
    pos = p[:, :3, 3]
    dist = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    rot = p[:-1, :3, :3].transpose(0, 2, 1) @ p[1:, :3, :3]
    ang = np.degrees(np.arccos(np.clip((np.trace(rot, axis1=1, axis2=2)-1)/2, -1, 1)))
    extent = float(np.linalg.norm(pos-pos.mean(0), axis=1).max())
    # Pose-only candidates. Translation threshold is relative to scene extent,
    # not metres (COLMAP gauge is not promised metric).
    still = (dist <= max(extent * 1e-4, 1e-8)) & (ang <= .1)
    stop_starts = [i for i in range(len(still)-15) if still[i:i+16].all()]
    # Long-gap revisit proxy: sampled views, >=128 recorded-frame gap.
    ids = np.arange(0, len(p), 4)
    near = []
    for i in ids:
        earlier = ids[ids <= i-128]
        if not len(earlier):
            continue
        d = np.linalg.norm(pos[earlier]-pos[i], axis=1)
        cos = (p[earlier, :3, 2] * p[i, :3, 2]).sum(1)
        candidates = earlier[(d <= .05*max(extent, 1e-8)) & (cos >= np.cos(np.radians(15)))]
        if len(candidates):
            near.append([int(candidates[0]), int(i)])
    return dict(frame_count=len(p), path_length=float(dist.sum()), total_rotation_deg=float(ang.sum()),
        scene_extent=extent, natural_stop_candidate_starts=stop_starts,
        long_revisit_candidate_pairs=near, pose_only_candidates_not_visual_QA=True)


def index_scenes(entries, split, out):
    reserved = {r['scene_id'] for r in split['scenes']} | set(split['excluded_legacy_scene_ids'])
    dest = out / 'corpus/scenes.jsonl'
    if dest.exists():
        rows = list(map(json.loads, dest.open()))
        if {r['scene_id'] for r in rows} != {e['scene_id'] for e in entries}:
            raise ValueError('existing corpus differs from source')
        for r in rows:
            if r['split'] != ('test_reserved' if r['scene_id'] in reserved else 'train'):
                raise ValueError('corpus split mismatch')
        return rows
    rows = []
    t0 = time.monotonic()
    dest.parent.mkdir(parents=True, exist_ok=True)
    for n, entry in enumerate(entries):
        path = Path(entry['scene_path'])
        row = dict(scene_id=entry['scene_id'], source_video_id=entry['scene_id'], scene_path=str(path),
                   split='test_reserved' if entry['scene_id'] in reserved else 'train')
        try:
            transform = path / 'transforms.json'
            td = json.loads(transform.read_text())
            frames = sorted(td['frames'], key=lambda f: f['file_path'])
            images = next(path/d for d in IMAGE_DIRS if (path/d).is_dir())
            present = {p.name for p in images.iterdir()}
            missing = [i for i, f in enumerate(frames) if Path(f['file_path']).name not in present]
            numbers = [int(Path(f['file_path']).stem.split('_')[-1]) for f in frames]
            gaps = [i for i in range(1, len(numbers)) if numbers[i] != numbers[i-1]+1]
            row.update(transforms_path=str(transform), transforms_sha256=file_sha(transform), image_root=str(images),
                order='transforms.frames_sorted_by_file_path_no_filtering_or_reindexing',
                missing_frame_indices=missing, filename_gap_indices=gaps,
                frame_count=len(frames), independent_VAE_tail_remainder=(len(frames)-1)%4,
                full_timeline_preserved=True, timestamps_available=False,
                stats=pose_stats([f['transform_matrix'] for f in frames]),
                ready_for_reference_sampling=not missing and not gaps, error=None)
        except (OSError, ValueError, KeyError, StopIteration) as exc:
            row.update(error=str(exc), ready_for_reference_sampling=False)
        rows.append(row)
        if (n+1) % 1000 == 0:
            print(f'INDEX {n+1}/{len(entries)} {time.monotonic()-t0:.1f}s', flush=True)
    write_lines(dest, rows)
    return rows


def group(scene, task, start, timeline, seed):
    # start is the number of observed frames: 1+4k. Target contributes 16.
    if start % 4 != 1 or len(timeline) != start + 16:
        raise ValueError('invalid temporal grid')
    history = list(range(max(1, start-32), start)) if start > 1 else []
    target = list(range(start, start+16))
    forbidden = {timeline[s] for s in target}
    candidates = [s for s in range(4, max(1, start-32), 4)
                  if not ({timeline[j] for j in range(s-3, s+1)} & forbidden)]
    sets = []
    if len(candidates) >= 4:
        for fractions in ((0., 1.), (.25, .75), (.5, 1.)):
            pair = sorted({candidates[round(f*(len(candidates)-1))] for f in fractions})
            if len(pair) == 2 and pair not in sets:
                sets.append(pair)
    elif len(candidates) >= 2:
        sets = [[candidates[0], candidates[-1]]]
    if not sets:
        sets = [[]]
    g = dict(scene_id=scene['scene_id'], source_video_id=scene['source_video_id'], task=task,
        prefix_frames=start, prefix_latents=1+(start-1)//4, recent_history_steps=history,
        target_steps=target, timeline_recorded_frame_ids=timeline, anchor_step=0,
        synthetic=task in ('reverse_return', 'fork_reverse', 'static_stop'),
        static_dynamic_QA='pending' if task in ('reverse_return', 'fork_reverse', 'static_stop') else 'not_synthetic',
        latent_origin='recorded_stream_encode_pending_not_native_generated',
        encoder_contract='full_causal_prefix_then_resumed_16_RGB_target',
        source_combinations=sets, source_selection='bounded_temporal_candidates_overlap_QA_pending',
        render_source_policy='pending_deployment_policy_lock_native_pairs_do_not_define_complete_render_bank',
        materialization_status='references_only', training_ready=False)
    g['group_id'] = digest(g)
    g['flow_pair_seed'] = int(digest([seed, g['group_id']])[:15],16)
    return g


def pilot(rows, out, seed, count):
    eligible = [r for r in rows if r['split']=='train' and r['ready_for_reference_sampling'] and r['frame_count']>=289]
    eligible.sort(key=lambda r: digest([seed, 'pilot', r['scene_id']]))
    selected = eligible[:count]
    if len(selected) != count:
        raise ValueError('not enough pilot scenes')
    groups = []
    for row in selected:
        n = row['frame_count']
        last = 1+4*((n-17)//4)
        # Include first-call and early/middle/late positions over the full scene.
        starts = sorted({1, 65, 1+4*((last//2)//4), last})
        for start in starts:
            groups.append(group(row, 'first_call' if start==1 else 'forward', start, list(range(start+16)), seed))
        start = 1+4*((min(n-17, max(161, n//2))-1)//4)
        forward = group(row, 'fork_forward', start, list(range(start+16)), seed)
        backward = group(row, 'fork_reverse', start, list(range(start))+list(range(start-2,start-18,-1)), seed)
        # Both forks use exactly the same archive candidate sets as well as H.
        common = [s for s in forward['source_combinations'] if s in backward['source_combinations']]
        if not common:
            raise ValueError('forks lack a shared archive set')
        fork_id = digest([row['scene_id'], start, 'fork'])
        for g in (forward, backward):
            g['source_combinations'] = common
            g['camera_fork_id'] = fork_id
            g['flow_pair_seed'] = int(fork_id[:15],16)
            g['group_id'] = digest({k:v for k,v in g.items() if k!='group_id'})
        groups += [forward, backward]
        groups.append(group(row, 'static_stop', start, list(range(start))+[start-1]*16, seed))
        # Revisit a view after an actual observation-time gap >128 frames.
        turn = 1+4*((n-1)//4)
        return_count = min(128, 4*((turn-17)//4))
        timeline = list(range(turn))+list(range(turn-2, turn-2-return_count-16,-1))
        groups.append(group(row, 'reverse_return', turn+return_count, timeline, seed))
    validate_groups(groups, {r['scene_id'] for r in selected})
    write_lines(out/'sampler/groups.jsonl', groups)
    recipes = []
    for g in groups:
        for variant, sources in enumerate(g['source_combinations']):
            recipes.append(dict(cache_key=digest([g['group_id'], sources, 'producer_pending']),
                group_id=g['group_id'], variant=variant, native_source_steps=sources,
                native_source_windows=[list(range(s-3,s+1)) for s in sources],
                status='pending_geometry_and_VAE', producer_hash=None,
                cache_key_final_requires=['renderer_version','source_RGB_hashes','source_pose_depth_versions',
                    'target_camera_hash','crop_K','VAE_version','encoder_state_contract','corruption_producer_version']))
    write_lines(out/'conditions/recipes.jsonl', recipes)
    samples=[]
    for g in groups:
        families = ('clean', 'memory_missing') if g['prefix_frames']==1 else FAMILIES
        for family in families:
            samples.append(dict(row_id=digest([g['group_id'], family]), group_id=g['group_id'], family=family,
                flow_pair_seed=g['flow_pair_seed'], synthetic=g['synthetic'], training_ready=False,
                corruption='own_deviation_archive_primary' if family in ('history_damaged','both_damaged') else family,
                own_state_kind='recorded_content_plus_deviation' if family in ('history_damaged','both_damaged') else None,
                model_inputs_exclude=['family','corruption_seed','teacher_quality_label']))
    write_lines(out/'sampler/rows.jsonl',samples)
    return selected, groups, recipes, samples


def validate_groups(groups, allowed):
    for g in groups:
        if g['scene_id'] not in allowed:
            raise ValueError('reserved scene leaked')
        t=g['target_steps']; h=g['recent_history_steps']; timeline=g['timeline_recorded_frame_ids']
        if g['prefix_frames']%4 != 1 or len(t)!=16 or (h and max(h)>=min(t)):
            raise ValueError('invalid temporal contract')
        for pair in g['source_combinations']:
            for s in pair:
                if s%4 or s >= min(h or t):
                    raise ValueError('archive leaks into history/future')
                if {timeline[j] for j in range(s-3,s+1)} & {timeline[j] for j in t}:
                    raise ValueError('exact target copied into archive')
    for fork in {g.get('camera_fork_id') for g in groups}-{None}:
        a,b=[g for g in groups if g.get('camera_fork_id')==fork]
        start=a['prefix_frames']
        if a['timeline_recorded_frame_ids'][:start]!=b['timeline_recorded_frame_ids'][:start] or a['source_combinations']!=b['source_combinations']:
            raise ValueError('fork conditions differ')


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--seed',type=int,default=20261009);p.add_argument('--pilot-scenes',type=int,default=3)
    a=p.parse_args()
    if not 1<=a.pilot_scenes<=24:
        p.error('pilot must contain 1..24 scenes; scale-up requires measured costs')
    source=json.load(gzip.open(DEFAULT_SOURCE));entries=source['entries']
    split=lock_split(entries,a.out,a.seed,source['heldout_scene_ids'])
    rows=index_scenes(entries,split,a.out)
    selected,groups,recipes,samples=pilot(rows,a.out,a.seed,a.pilot_scenes)
    training=[r for r in rows if r['split']=='train' and r.get('stats')]
    summary=dict(schema='base_restart.preparation_v2',source_manifest_sha256=file_sha(DEFAULT_SOURCE),
        locked_test_sha256=file_sha(a.out/'locked_test.json'),locked_test_scenes=24,
        corpus_scenes=len(rows),train_scenes=sum(r['split']=='train' for r in rows),
        eligible_reference_scenes=sum(r['split']=='train' and r['ready_for_reference_sampling'] for r in rows),
        full_scene_frames=sum(r.get('frame_count',0) for r in rows),
        natural_stop_candidate_scenes=sum(bool(r['stats']['natural_stop_candidate_starts']) for r in training),
        natural_long_revisit_candidate_scenes=sum(bool(r['stats']['long_revisit_candidate_pairs']) for r in training),
        thresholds=dict(stop_translation='scene_extent*1e-4',stop_rotation_deg=.1,stop_consecutive_steps=16,
                        revisit_min_gap_frames=128,revisit_translation='scene_extent*.05',revisit_view_angle_deg=15),
        candidate_counts_are_visual_QA=False,
        pilot_scenes=[r['scene_id'] for r in selected],groups=len(groups),offline_recipes=len(recipes),rows=len(samples),
        task_counts=dict(Counter(g['task'] for g in groups)),
        prefix_frame_counts=sorted({g['prefix_frames'] for g in groups}),
        artifacts={str(p.relative_to(a.out)):file_sha(p) for p in [a.out/'corpus/scenes.jsonl',a.out/'sampler/groups.jsonl',a.out/'sampler/rows.jsonl',a.out/'conditions/recipes.jsonl']},
        training_ready=False,blockers=['materialize_VAE_and_geometry','streaming_VAE_numerical_preflight',
            'source_overlap_and_synthetic_dynamic_QA','own_deviation_archive_materialization',
            'foundation_interface_and_trainable_preflight','pilot_GPU_hours_and_cache_bytes'])
    write(a.out/'PREPARATION_AUDIT.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
