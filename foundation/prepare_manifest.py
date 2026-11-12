"""Prepare causal RGB/camera groups for a fresh Wan base experiment on CPU.

This does not run a model, estimate depth, apply corruption, or launch training.
Each counterfactual group shares the recorded future; corruption plans remain
explicitly unmaterialized. Revisit causality is checked on observation steps,
because original video frame IDs run backwards on the return trajectory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / 'gaussverse/outputs/recurrent_train_manifest_v1.json.gz'
DEFAULT_EXCLUSIONS = ROOT / 'training_runs/one_step_g2_control_20260920/data_inventory.json'
IMAGE_DIRS = ('images_4', 'images_8', 'images', 'images_2')
FAMILIES = ('clean', 'history_damaged', 'memory_damaged', 'both_damaged', 'memory_missing')
TASKS = ('forward', 'revisit', 'camera_fork_forward', 'camera_fork_return')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def read_scene(path):
    td = json.loads((path / 'transforms.json').read_text())
    images = next((path / d for d in IMAGE_DIRS if (path / d).is_dir()), None)
    if images is None:
        raise ValueError('missing scene images')
    frames = []
    for f in sorted(td['frames'], key=lambda f: f['file_path']):
        p = images / Path(f['file_path']).name
        if p.is_file():
            frames.append({'path': str(p.resolve()), 'c2w_opengl': f['transform_matrix']})
    return td, frames


def source_rank(frames, frame_id, target_ids):
    """Camera-only proxy; does not certify pixel or surface overlap."""
    a = frames[frame_id]['c2w_opengl']
    distances, angles = [], []
    for i in target_ids:
        b = frames[i]['c2w_opengl']
        distances.append(math.sqrt(sum((a[k][3] - b[k][3]) ** 2 for k in range(3))))
        dot = sum(a[k][2] * b[k][2] for k in range(3))
        norm = math.sqrt(sum(a[k][2] ** 2 for k in range(3)) *
                         sum(b[k][2] ** 2 for k in range(3)))
        angles.append(math.acos(max(-1., min(1., dot / max(norm, 1e-12)))))
    return sum(angles) / len(angles), sum(distances) / len(distances)


def make_group(entry, td, frames, task, seed):
    base = int(entry['trajectory']['base_frame_id'])
    if task == 'forward':
        timeline = list(range(base, base + 161))
        start = 97                         # seventh 16-frame target call
    elif task == 'revisit':
        timeline = list(range(base, base + 81)) + list(range(base + 79, base + 15, -1))
        start = 113                        # return at a genuinely earlier recorded view
    elif task == 'camera_fork_forward':
        timeline = list(range(base, base + 97))
        start = 81
    elif task == 'camera_fork_return':
        timeline = list(range(base, base + 81)) + list(range(base + 79, base + 63, -1))
        start = 81
    else:
        raise ValueError('unknown task')
    if max(timeline) >= len(frames):
        raise ValueError('insufficient recorded frames')
    target_steps = list(range(start, start + 16))
    history_steps = list(range(start - 16, start))
    target_ids = [timeline[s] for s in target_steps]
    target_set = set(target_ids)
    seen_ids = set()
    candidates = []
    # Keep archive sources separate from recent temporal history; avoid exact
    # target-frame copying even when that RGB was observed earlier on a revisit.
    for step in range(history_steps[0]):
        if step % 4:
            continue
        fid = timeline[step]
        window = [0] if step == 0 else list(range(step - 3, step + 1))
        if not ({timeline[s] for s in window} & target_set) and fid not in seen_ids:
            seen_ids.add(fid)
            candidates.append(step)
    candidates.sort(key=lambda s: (*source_rank(frames, timeline[s], target_ids), s))
    if len(candidates) < 2:
        raise ValueError('insufficient independent archive candidates')
    # Forks share exactly the same input history and archive. Only their target
    # cameras/recorded supervision change. These two old sources lie outside
    # both target sequences and outside the recent temporal window.
    memory_steps = [32, 48] if task.startswith('camera_fork_') else sorted(candidates[:2])
    memory_windows = [[0] if s == 0 else list(range(s - 3, s + 1)) for s in memory_steps]
    archive_steps = [s for window in memory_windows for s in window]
    relevant_steps = sorted(set([0] + history_steps + archive_steps + target_steps))
    observations = [{'step': step, 'recorded_frame_id': timeline[step],
                     **frames[timeline[step]]} for step in relevant_steps]
    cameras = {'convention': 'OpenGL_c2w',
               'intrinsics_original': {k: td[k] for k in ('fl_x', 'fl_y', 'cx', 'cy', 'w', 'h')},
               'requires_exact_resize_cover_center_crop_intrinsics': True}
    shared = {'scene_id': entry['scene_id'], 'scene_path': str(Path(entry['scene_path']).resolve()),
              'task': task, 'timeline_recorded_frame_ids': timeline[:start + 16],
              'target_steps': target_steps, 'recent_history_steps': history_steps,
              'memory_steps': memory_steps, 'memory_source_windows': memory_windows,
              'observations': observations, 'cameras': cameras}
    gid = digest(shared)
    return {'group_id': gid, **shared,
            'target_reference_sha256': digest([frames[i] for i in target_ids]),
            'target_reference_hash_scope': 'ordered_pixel_paths_and_camera_metadata_not_pixel_bytes',
            'flow_pair_seed': int(digest([seed, gid, 'flow_pair'])[:15], 16),
            'history_corruption_seed': int(digest([seed, gid, 'history'])[:15], 16),
            'memory_corruption_seed': int(digest([seed, gid, 'memory'])[:15], 16),
            'source_overlap_status': 'unverified_camera_proximity_only',
            'static_scene_status': 'unverified_reverse_tasks_require_dynamic_region_QA',
            'camera_fork_id': digest([entry['scene_id'], base, 'camera_fork'])
                if task.startswith('camera_fork_') else None,
            'materialization_status': 'RGB_camera_references_only'}


def validate_group(group, excluded):
    if group['scene_id'] in excluded:
        raise ValueError('excluded scene leaked into training')
    t, h, m = group['target_steps'], group['recent_history_steps'], group['memory_steps']
    archive_steps = [s for window in group['memory_source_windows'] for s in window]
    if len(t) != 16 or len(h) != 16 or len(m) != 2:
        raise ValueError('wrong group dimensions')
    if max(h + archive_steps) >= min(t) or max(archive_steps) >= min(h):
        raise ValueError('future or recent-history leakage into archive')
    if set(t) & (set(h) | set(m)):
        raise ValueError('target observation is a condition')
    timeline = group['timeline_recorded_frame_ids']
    if {timeline[s] for s in t} & {timeline[s] for s in archive_steps}:
        raise ValueError('exact target RGB reused as archive source')
    if group['task'] in ('revisit', 'camera_fork_return') and not all(timeline[a] > timeline[b] for a, b in zip(t, t[1:])):
        raise ValueError('revisit is not a backwards recorded trajectory')


def prepare(source, old_inventory, out, limit, seed):
    with gzip.open(source, 'rt') as f:
        inventory = json.load(f)
    legacy = json.loads(old_inventory.read_text())
    excluded = set(inventory['heldout_scene_ids']) | set(legacy['excluded_evaluation_scenes'])
    locked_test = Path(__file__).resolve().parent / 'data_v2/locked_test.json'
    if locked_test.exists():
        excluded.update(r['scene_id'] for r in json.loads(locked_test.read_text())['scenes'])
    entries = [e for e in inventory['entries'] if e['scene_id'] not in excluded]
    if len({e['scene_id'] for e in entries}) != len(entries):
        raise ValueError('duplicate scene hash in source inventory')
    entries.sort(key=lambda e: digest([seed, e['scene_id']]))
    groups, skipped, selected = [], [], []
    for entry in entries:
        if len(selected) >= limit:
            break
        try:
            path = Path(entry['scene_path'])
            td, frames = read_scene(path)
            local = [make_group(entry, td, frames, task, seed) for task in TASKS]
            for g in local:
                validate_group(g, excluded)
            groups.extend(local)
            selected.append({'scene_id': entry['scene_id'], 'scene_path': str(path.resolve())})
        except (OSError, ValueError, KeyError) as exc:
            skipped.append({'scene_id': entry['scene_id'], 'reason': str(exc)})
    if len(selected) != limit:
        raise RuntimeError(f'only {len(selected)} usable scenes; requested {limit}')
    out.mkdir(parents=True, exist_ok=True)
    pool = {'schema': 'gaussverse.base_restart_eligible_pool.v1',
            'source_manifest': str(source.resolve()), 'source_manifest_sha256': file_sha(source),
            'excluded_scene_ids': sorted(excluded),
            'scene_count': len(entries), 'status': 'inventory_candidates_not_geometry_certified',
            'scenes': [{'scene_id': e['scene_id'], 'scene_path': e['scene_path'],
                        'frame_count': e['frame_count'],
                        'base_frame_id': e['trajectory']['base_frame_id']}
                       for e in entries]}
    (out / 'eligible_pool.json').write_text(json.dumps(pool, indent=2) + '\n')
    group_path, row_path = out / 'groups.jsonl', out / 'rows.jsonl'
    with group_path.open('w') as f:
        for group in groups:
            f.write(json.dumps(group, sort_keys=True) + '\n')
    counts = Counter()
    with row_path.open('w') as f:
        for group in groups:
            for family in FAMILIES:
                row = {'row_id': digest([group['group_id'], family]),
                       'group_id': group['group_id'], 'family': family,
                       'target_reference_sha256': group['target_reference_sha256'],
                       'flow_pair_seed': group['flow_pair_seed'],
                       'history_damaged': family in ('history_damaged', 'both_damaged'),
                       'memory_damaged': family in ('memory_damaged', 'both_damaged'),
                       'memory_missing': family == 'memory_missing',
                       'history_corruption_seed': group['history_corruption_seed'],
                       'memory_corruption_seed': group['memory_corruption_seed'],
                       'corruption_materialization': 'pending',
                       'usable_memory_support': 'unverified',
                       'is_model_input': {'family': False, 'corruption_seed': False}}
                counts[family] += 1
                f.write(json.dumps(row, sort_keys=True) + '\n')
    split = {'schema': 'gaussverse.base_restart_split.v1', 'seed': seed,
             'training_scenes': selected, 'excluded_scene_ids': sorted(excluded),
             'excluded_scene_policy': 'reserve_all_legacy_and_expanded_heldout_scene_hashes',
             'fresh_final_eval_scene_selection': 'pending_prior_inspection_audit',
             'source_manifest': str(source.resolve()), 'source_manifest_sha256': file_sha(source)}
    (out / 'splits.json').write_text(json.dumps(split, indent=2) + '\n')
    audit = {'schema': 'gaussverse.base_restart_preparation_audit.v1',
             'scene_count': len(selected), 'group_count': len(groups),
             'row_count': sum(counts.values()), 'row_family_counts': dict(counts),
             'excluded_scene_count': len(excluded), 'eligible_source_scene_count': len(entries),
             'skipped': skipped, 'causality_key': 'observation_step',
             'memory_overlaps_recent_history': False, 'exact_target_rgb_in_memory': False,
             'counterfactual_target_reference_shared': True,
             'counterfactual_flow_seed_shared': True,
             'independent_modality_corruption_seeds': True,
             'same_history_camera_forks': len(selected),
             'synthetic_return_static_scene_QA_complete': False,
             'depth_or_overlap_qa_complete': False,
             'VAE_latents_prepared': False, 'corruptions_materialized': False,
             'training_ready': False, 'GPU_job_launched': False,
             'groups_sha256': file_sha(group_path), 'rows_sha256': file_sha(row_path),
             'source_pixel_hashes_verified': False}
    (out / 'DATA_AUDIT.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({k: audit[k] for k in ('scene_count', 'group_count', 'row_count',
          'excluded_scene_count', 'training_ready')}, indent=2))


def export_recorded_groups(out, count, height, width):
    import numpy as np
    from PIL import Image
    samples = out / 'recorded_samples'
    samples.mkdir(exist_ok=True)
    with (out / 'groups.jsonl').open() as f:
        groups = [json.loads(next(f)) for _ in range(count)]
    records = []
    for group in groups:
        by_step = {o['step']: o for o in group['observations']}
        def pixels(steps):
            result = []
            for step in steps:
                with Image.open(by_step[step]['path']) as image:
                    image = image.convert('RGB')
                    scale = max(width / image.width, height / image.height)
                    rw, rh = round(image.width * scale), round(image.height * scale)
                    image = image.resize((rw, rh), Image.Resampling.LANCZOS)
                    left, top = (rw - width) // 2, (rh - height) // 2
                    result.append(np.asarray(image.crop((left, top, left + width, top + height))))
            return np.stack(result)
        source_steps, target_steps = group['memory_steps'], group['target_steps']
        history = pixels(group['recent_history_steps'])
        memory, target = pixels(source_steps), pixels(target_steps)
        i = group['cameras']['intrinsics_original']
        scale = max(width / i['w'], height / i['h'])
        cx = i['cx'] * scale - (round(i['w'] * scale) - width) // 2
        cy = i['cy'] * scale - (round(i['h'] * scale) - height) // 2
        K = np.array([[i['fl_x'] * scale, 0., cx], [0., i['fl_y'] * scale, cy], [0., 0., 1.]],dtype=np.float32)
        poses = lambda steps: np.asarray([by_step[s]['c2w_opengl'] for s in steps], dtype=np.float32)
        path = samples / f"{group['group_id']}.npz"
        np.savez_compressed(path, history_rgb=history, source_rgb=memory, target_rgb=target,
                            K=K, history_c2w_opengl=poses(group['recent_history_steps']),
                            source_c2w_opengl=poses(source_steps), target_c2w_opengl=poses(target_steps),
                            source_window_0_rgb=pixels(group['memory_source_windows'][0]),
                            source_window_1_rgb=pixels(group['memory_source_windows'][1]))
        records.append({'group_id': group['group_id'], 'path': str(path.resolve()),
                        'file_sha256': file_sha(path), 'bytes': path.stat().st_size,
                        'target_pixel_sha256': hashlib.sha256(target.tobytes()).hexdigest(),
                        'task': group['task'], 'contains_corrupted_examples': False,
                        'depth_and_source_overlap': 'not_prepared'})
    (samples / 'manifest.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps({'exported_recorded_groups': count,
                      'total_bytes': sum(r['bytes'] for r in records)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--old-inventory', type=Path, default=DEFAULT_EXCLUSIONS)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'data')
    parser.add_argument('--scenes', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20261008)
    parser.add_argument('--export-groups', type=int, default=0)
    args = parser.parse_args()
    if args.scenes <= 0 or args.export_groups < 0 or args.export_groups > len(TASKS) * args.scenes:
        parser.error('invalid scene/export counts')
    prepare(args.source, args.old_inventory, args.out, args.scenes, args.seed)
    if args.export_groups:
        export_recorded_groups(args.out, args.export_groups, height=256, width=448)


if __name__ == '__main__':
    main()
