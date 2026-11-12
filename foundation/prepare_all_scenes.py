"""Versioned full local DL3DV inventory and scene-disjoint dataset index.

No adapted checkpoint, generated pixels or model quality scores are consulted.
Frame IDs always refer to the sorted ORIGINAL transforms array. Missing/gapped
frames split streams; they are never silently removed to join unrelated frames.
"""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import numpy as np
from PIL import Image
from prepare_manifest import file_sha, digest
from prepare_v2 import pose_stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEST = HERE / 'dataset_v3'
SEED = 20261009
DEV_PILOT = 'ebe0d26c327cd71a58f446293454e3ef7aef76c915e0d8b0498b26c9ca599a9c'
TRAIN_PILOT = {'33991c6f629c4eddf573af8ce30a6780a0a7b7cd90bf248f6a0bd17dbb29c069',
               '61285fd8608d219abedbee5488698cf8a7d921ba8df27e112954ee2aeb969237'}


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n'); temp.replace(path)


def lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def contiguous_segments(valid, numbers):
    segments = []; start = None
    for i, good in enumerate(valid):
        if start is not None and (not good or numbers[i] != numbers[i - 1] + 1):
            segments.append([start, i]); start = None
        if good and start is None:
            start = i
    if start is not None:
        segments.append([start, len(valid)])
    return segments


def inspect_source(path, reserved):
    sid = path.parent.name
    base = dict(scene_id=sid, source_video_id=sid, scene_path=str(path.parent),
                transforms_path=str(path), transforms_sha256=file_sha(path), release=path.parts[-4])
    if sid in reserved:
        return dict(base, status='reserved_no_RGB_inspection', streams=[], usable_frames=0)
    try:
        td = json.loads(path.read_text()); frames = sorted(td['frames'], key=lambda f: f['file_path'])
        image_root = next((path.parent / d for d in ('images_4', 'images_8', 'images', 'images_2')
                           if (path.parent / d).is_dir()), None)
        if image_root is None:
            return dict(base, status='missing_image_directory', streams=[], usable_frames=0, frame_count=len(frames))
        K = np.asarray([td[k] for k in ('fl_x', 'fl_y', 'cx', 'cy', 'w', 'h')], float)
        if not np.isfinite(K).all() or (K[[0, 1, 4, 5]] <= 0).any():
            raise ValueError('invalid intrinsics')
        poses = np.asarray([f['transform_matrix'] for f in frames], dtype=np.float64)
        if poses.shape != (len(frames), 4, 4):
            raise ValueError('pose shape')
        present = {p.name for p in image_root.iterdir() if p.is_file()}
        names = [Path(f['file_path']).name for f in frames]
        numbers = [int(re.search(r'(\d+)$', Path(n).stem).group(1)) for n in names]
        finite = np.isfinite(poses).all(axis=(1, 2))
        rotations = poses[:, :3, :3]
        good_pose = finite & np.isclose(poses[:, 3], [0, 0, 0, 1], atol=1e-5).all(axis=1)
        good_pose &= np.isclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=2e-3).all(axis=(1, 2))
        good_pose &= np.isclose(np.linalg.det(rotations), 1, atol=2e-3)
        valid = good_pose & np.asarray([n in present for n in names])
        bounds = contiguous_segments(valid, numbers)
        streams = []; preview_hashes = []
        for start, end in bounds:
            if end - start < 17:
                continue
            check_ids = sorted({start, (start + end - 1) // 2, end - 1})
            for i in check_ids:
                image_path = image_root / names[i]
                with Image.open(image_path) as im:
                    im.load()
                    if min(im.size) < 64:
                        raise ValueError('thumbnail-sized source')
                preview_hashes.append(file_sha(image_path))
            count = end - start; encoded = 1 + 4 * ((count - 1) // 4)
            stats = pose_stats(poses[start:end])
            # A return must leave the vicinity, not merely remain nearby.
            extent = max(stats['scene_extent'], 1e-8); returns = []
            for earlier, later in stats.pop('long_revisit_candidate_pairs'):
                displacement = np.linalg.norm(poses[start+earlier:start+later+1, :3, 3] - poses[start+earlier, :3, 3], axis=1)
                directions = poses[start+earlier:start+later+1, :3, 2] @ poses[start+earlier, :3, 2]
                if displacement.max() > .15 * extent or directions.min() < np.cos(np.radians(45)):
                    target_start = 1 + 4 * ((later - 1) // 4)
                    if target_start + 16 <= encoded:
                        returns.append(dict(earlier_frame=earlier, returning_frame=later,
                            target_start_latent=1 + (target_start - 1)//4))
            stream_id = digest([sid, start, end, base['transforms_sha256']])[:24]
            streams.append(dict(stream_id=stream_id, original_start=start, original_end_exclusive=end,
                raw_frames=count, encoded_frames=encoded, latent_count=1+(encoded-1)//4,
                tail_frames_not_on_VAE_grid=count-encoded, target_start_latent_min=1,
                target_start_latent_max=(encoded-1)//4-3, stats=stats,
                revisit_pose_candidates=returns, visual_revisit_QA=False))
        return dict(base, image_root=str(image_root), frame_count=len(frames),
            status='eligible_reference' if streams else 'no_contiguous_17_frame_stream', streams=streams,
            usable_frames=sum(s['encoded_frames'] for s in streams),
            missing_frame_count=sum(n not in present for n in names), invalid_pose_count=int((~good_pose).sum()),
            filename_gap_count=sum(b != a+1 for a, b in zip(numbers, numbers[1:])),
            inspected_RGB_frames=len(preview_hashes), full_RGB_decode_checked=False,
            exact_sample_fingerprint=digest(preview_hashes) if preview_hashes else None,
            full_timeline_preserved=True, source_fps=None, time_units='recorded_frame_index_not_seconds',
            transform_frame_order='sorted_by_file_path_original_index_no_filtering')
    except Exception as exc:
        return dict(base, status='invalid_source', reason=f'{type(exc).__name__}: {exc}', streams=[], usable_frames=0)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, default=DEST)
    parser.add_argument('--dev-scenes', type=int, default=128); parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args(); out = args.out; start = time.monotonic()
    locked_path = HERE / 'data_v2/locked_test.json'; locked = json.loads(locked_path.read_text())
    test = {r['scene_id'] for r in locked['scenes']}; legacy_eval = set(locked['excluded_legacy_scene_ids'])
    reserved = test | legacy_eval
    roots = [ROOT / 'data/DL3DV-480P', ROOT / 'data/DL3DV-960P']
    paths = sorted(p for root in roots for p in root.rglob('transforms.json'))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        sources = list(pool.map(lambda p: inspect_source(p, reserved), paths))
    lines(out / 'corpus/sources.jsonl', sources)
    by_id = defaultdict(list)
    for row in sources:
        by_id[row['scene_id']].append(row)
    selected = []
    for sid, candidates in sorted(by_id.items()):
        # Prefer complete recordings, then native 480P for this 448x256 corpus.
        candidates.sort(key=lambda r: (-r['usable_frames'], 'DL3DV-480P' not in r['scene_path'], r['scene_path']))
        chosen = dict(candidates[0]); chosen['alternate_sources'] = [r['scene_path'] for r in candidates[1:]]
        selected.append(chosen)
    eligible = [r for r in selected if r['status'] == 'eligible_reference']
    fingerprints = defaultdict(list)
    for r in eligible:
        fingerprints[r['exact_sample_fingerprint']].append(r['scene_id'])
    qa_path = ROOT / 'analysis/plan_base_restart_20261008/scene_qa.jsonl'
    qa = {r['scene_id']: r for r in map(json.loads, qa_path.open())}
    bins = defaultdict(list)
    for row in eligible:
        q = qa.get(row['scene_id'], {})
        length = max(s['raw_frames'] for s in row['streams'])
        rotation = sum(s['stats']['total_rotation_deg'] for s in row['streams'])
        row['sampling_stratum'] = f"length_{int(length>=321)}_rotation_{int(rotation>=700)}_glass_{int(q.get('glass_vs_matte',0)>=.6)}"
        row['material_tags_are_proxies'] = True
        row['proxy_tags'] = {k:q[k] for k in ('glass_vs_matte','dynamic_vs_static') if k in q}
        bins[row['sampling_stratum']].append(row)
    dev = {DEV_PILOT}
    assert dev <= {r['scene_id'] for r in eligible}
    ordered = {k:sorted(v,key=lambda r:digest([SEED,'development',r['scene_id']])) for k,v in bins.items()}
    while len(dev) < args.dev_scenes:
        progress = False
        for key in sorted(ordered):
            while ordered[key]:
                row = ordered[key].pop(0); sid=row['scene_id']
                if sid in dev or sid in TRAIN_PILOT:
                    continue
                group=set(fingerprints[row['exact_sample_fingerprint']])
                if group & TRAIN_PILOT:
                    continue
                dev |= group; progress=True; break
            if len(dev) >= args.dev_scenes:
                break
        if not progress:
            raise ValueError('insufficient development candidates')
    # Ensure exact sampled-image duplicates stay in one split, including pilot dev.
    for ids in fingerprints.values():
        if set(ids) & dev:
            dev.update(ids)
    split = dict(schema='base_restart.all_scenes_split.v3', seed=SEED, final_test_path=str(locked_path),
        final_test_sha256=file_sha(locked_path), test_scene_ids=sorted(test),
        historical_evaluation_scene_ids=sorted(legacy_eval-test), development_scene_ids=sorted(dev),
        selection='deterministic_proxy_stratification_plus_existing_pilot_development',
        duplicate_policy='source_video_identity_across_resolutions_and_exact_three_image_fingerprints_grouped',
        same_location_near_duplicates_certified=False)
    split_path=out/'splits.json'
    if split_path.exists() and json.loads(split_path.read_text()) != split:
        raise ValueError('split already locked; create a new version rather than changing it')
    write(split_path,split)
    streams=[]
    for row in selected:
        sid=row['scene_id']
        row['split']='test_locked' if sid in test else 'historical_evaluation' if sid in legacy_eval else 'development' if sid in dev else 'train'
        for stream in row['streams']:
            streams.append(dict(stream,scene_id=sid,split=row['split'],scene_path=row['scene_path'],
                image_root=row['image_root'],transforms_sha256=row['transforms_sha256'],
                sampling_stratum=row['sampling_stratum'],source_video_id=sid))
    lines(out/'corpus/scenes.jsonl',selected);lines(out/'corpus/streams.jsonl',streams)
    for name in ('train','development'):
        lines(out/f'sampler/{name}_streams.jsonl',[s for s in streams if s['split']==name])
    active=[r for r in selected if r['streams']]
    old={r['scene_id'] for r in map(json.loads,(HERE/'data_v2/corpus/scenes.jsonl').open())}
    report=dict(schema='base_restart.all_scenes_inventory.v3',status='indexed_RGB_camera_references_not_fully_materialized',
        source_variants=len(sources),unique_scene_ids=len(selected),resolution_duplicates=len(sources)-len(selected),
        eligible_scenes=len(active),split_counts=dict(Counter(r['split'] for r in active)),
        inventory_status_counts=dict(Counter(r['status'] for r in selected)),
        total_streams=len(streams),encoded_frames=sum(s['encoded_frames'] for s in streams),
        full_recorded_frames=sum(r.get('frame_count',0) for r in active),
        target_windows=sum(s['target_start_latent_max'] for s in streams),
        new_eligible_ids_not_in_old_index=sorted(r['scene_id'] for r in active if r['scene_id'] not in old),
        recovered_fragmented_scenes=sum(len(r['streams'])>1 or r['missing_frame_count']>0 or r['filename_gap_count']>0 for r in active),
        revisit_candidate_streams=sum(bool(s['revisit_pose_candidates']) for s in streams),
        natural_stop_candidate_streams=sum(bool(s['stats']['natural_stop_candidate_starts']) for s in streams),
        duplicate_fingerprint_groups=[ids for ids in fingerprints.values() if len(ids)>1],
        full_sequence_RGB_decode_pending=True,synthetic_tasks_enabled=False,own_state_rows=0,
        old_adapted_models_or_outputs_used=False,final_test_RGB_read=False,
        free_bytes=shutil.disk_usage(out).free,wall_seconds=time.monotonic()-start,
        artifacts={str(p.relative_to(out)):file_sha(p) for p in [split_path,out/'corpus/scenes.jsonl',out/'corpus/streams.jsonl',out/'corpus/sources.jsonl']})
    write(out/'inventory.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('new_eligible_ids_not_in_old_index','duplicate_fingerprint_groups','artifacts')},indent=2),flush=True)


if __name__=='__main__':main()
