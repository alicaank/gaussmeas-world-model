"""Versioned, train-only 2x2 render and source-visibility cost pilot.

This prepares experimental inputs, not router labels or a training checkpoint.
Original corpus caches are read-only. Controlled corruptions are engineering
probes, not a fit to natural errors. Visibility is estimated from past geometry;
it is not a certified static-surface mask.
"""
import argparse
from collections import Counter
import fcntl
import gzip
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from all_scene_dataset import DATA, RecordedStream, pack_bf16, unpack_bf16
from geometry_policy import geometry_path, validate_geometry_policy
from prepare_manifest import digest, file_sha
from native_archive import NativeArchiveIndex

HERE = Path(__file__).resolve().parent
OUT = DATA / 'review_20261010/physical_interventions64_v2'
SCHEMA = 'foundation.physical_intervention_profile.v2'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def tensor_hash(value):
    import hashlib
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def make_plan(root, out, count=64):
    readiness = json.loads((root / 'readiness.json').read_text())
    active = json.loads((root / 'active_round.json').read_text())
    if readiness['status'] != 'complete_components':
        raise ValueError('complete audited original cache required')
    if readiness['recipes_sha256'] != active['sha256'] or file_sha(active['recipes']) != active['sha256']:
        raise ValueError('active recipe digest mismatch')
    rows = {r['stream_id']: r for r in map(json.loads, (root / 'corpus/streams.jsonl').open())}
    recipes = [r for r in map(json.loads, Path(active['recipes']).open())
               if r['split'] == 'train' and r['archive_indices'] and r['source_variant'] == 0]
    index = NativeArchiveIndex(data_root=root)
    # The proxy only stratifies this cost pilot; it is not a support label.
    candidate_file = index.root / 'native_only_candidates.jsonl.gz'
    with gzip.open(candidate_file, 'rt') as f:
        high = {(r['stream_id'], r['prefix_latents']) for r in map(json.loads, f)
                if r['gt10_in_all_depth_bands']}
    ordered = sorted(recipes, key=lambda r: digest([SCHEMA, r['group_id']]))
    chosen, scenes = [], set()
    for name, wanted, goal in [('high_proxy', True, count // 2), ('uniform_remaining', None, count)]:
        for r in ordered:
            if len(chosen) >= goal:
                break
            if r['scene_id'] in scenes or (wanted and (r['stream_id'], r['prefix_latents']) not in high):
                continue
            native = index.select(rows[r['stream_id']], r['prefix_latents'], max_refs=6)
            chosen.append(dict(recipe=r, selection_stratum=name, native_selection=native,
                               geometry_error='depth' if len(chosen) % 2 == 0 else 'source_pose'))
            scenes.add(r['scene_id'])
    if len(chosen) != count or Counter(x['selection_stratum'] for x in chosen)['high_proxy'] != count // 2:
        raise ValueError('insufficient unique scenes for prespecified strata')
    plan = dict(schema=SCHEMA, groups=chosen, group_count=count, split='train',
        purpose='cost_and_input_factorization_pilot_not_training_labels',
        source_files_sha256={str(root / 'active_round.json'): file_sha(root / 'active_round.json'),
                            str(root / 'corpus/streams.jsonl'): file_sha(root / 'corpus/streams.jsonl'),
                            str(Path(active['recipes'])): active['sha256'],
                            str(candidate_file): file_sha(candidate_file)},
        selected_source_policy='reuse_original_render_sources; profile_v2_native_support_separately',
        variants=['A0G0', 'A1G0', 'A0G1', 'A1G1'], render_seed=0,
        depth_rng='fixed_by_recipe_and_ordered_source_ids_in_isolated_torch_rng; model_loaded_before_seed_scope',
        original_clean_cache='diagnostic_only; paired_interventions_use_fresh_A0G0_from_identical_shared_depths',
        interventions=dict(affected='render_archive_sources_only; anchor_and_recent_unchanged',
            appearance='RGB01_channel_gain_[0.7,1.1,0.85]_offset_[0.12,-0.08,0.04]_clamped',
            depth='positive_multiplicative_smooth_field_1+0.2*sin(2pi*x)*cos(pi*y)',
            source_pose='local_yaw_3deg_and_local_x_translation_0.03_anchor_depth_units',
            magnitudes='fixed_engineering_probes_not_natural_error_calibration',
            factorization='input_RGB_vs_input_depth_pose; learned_render_output_not_perfectly_disentangled'),
        support=dict(classes=['anchor', 'recent_K8', 'render_archive', 'native_archive_v2'],
            target_offsets=[4, 8, 12, 16], source_confidence_min=0.5,
            zbuffer_relative_tolerance=0.05, zbuffer_absolute_tolerance=0.005,
            grid=[4, 16, 28], bit_threshold=0.5, static_surface_qualified=False,
            thresholds='diagnostic_only_pending_train_calibration',
            occlusion='nearest_projected_available_past_surface; no_target_depth_or_RGB'),
        camera_scope='offline_registered_camera_annotation_in_fixed_anchor_units_not_runtime_scale_evaluation',
        target_RGB_read=False, dev_or_test_inputs=False, training_enabled=False,
        native_appearance_corruption='not_materialized_by_this_render_cost_pilot',
        reserve_free_gib=8, profile_output_cap_gib=1)
    path = out / 'plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('immutable pilot plan differs; select a new output directory')
    save(path, plan)
    return plan


def perturb(sources, archive_ids, appearance, geometry, kind):
    """Branch-local input copies; neither factor can modify the other inputs."""
    result = []
    for source in sources:
        s = {k: v.clone() if torch.is_tensor(v) else v for k, v in source.items()}
        if s['frame_id'] in archive_ids:
            if appearance:
                gain = s['rgb'].new_tensor([.7, 1.1, .85])[:, None, None]
                offset = s['rgb'].new_tensor([.12, -.08, .04])[:, None, None]
                s['rgb'] = (((s['rgb'] + 1) / 2 * gain + offset).clamp(0, 1) * 2 - 1)
            if geometry and kind == 'depth':
                h, w = s['depth'].shape
                yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing='ij')
                s['depth'] *= 1 + .2 * torch.sin(2 * torch.pi * xx) * torch.cos(torch.pi * yy)
            elif geometry and kind == 'source_pose':
                angle = torch.tensor(3 * np.pi / 180, dtype=s['c2w'].dtype)
                rot = torch.eye(4, dtype=s['c2w'].dtype)
                rot[0, 0] = rot[2, 2] = angle.cos()
                rot[0, 2], rot[2, 0] = angle.sin(), -angle.sin()
                rot[0, 3] = .03
                s['c2w'] = s['c2w'] @ rot
        result.append(s)
    return result


def project_depth(source, target, K, size, device):
    """Per-source nearest-pixel z buffer in target camera coordinates."""
    h, w = size
    depth, conf = source['depth'].to(device), source['confidence'].to(device)
    k = K.to(device).float()
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    pix = torch.stack([xx, yy, torch.ones_like(xx)]).reshape(3, -1).float()
    xyz = (torch.linalg.inv(k) @ pix) * depth.reshape(1, -1)
    relative = torch.linalg.inv(target.to(device).float()) @ source['c2w'].to(device).float()
    cam = relative[:3, :3] @ xyz + relative[:3, 3:4]
    uv = k @ cam
    u = (uv[0] / cam[2].clamp_min(1e-8)).round().long()
    v = (uv[1] / cam[2].clamp_min(1e-8)).round().long()
    valid = torch.isfinite(depth.reshape(-1)) & (depth.reshape(-1) > 0) & (conf.reshape(-1) > .5)
    valid &= torch.isfinite(cam).all(0) & (cam[2] > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    out = torch.full((h * w,), float('inf'), device=device)
    out.scatter_reduce_(0, (v[valid] * w + u[valid]), cam[2, valid], reduce='amin', include_self=True)
    return out.reshape(h, w)


def support_masks(sources, groups, targets, K, size=(256, 448), device='cuda'):
    result = []
    for target in targets:
        projected = {i: project_depth(s, target, K, size, device) for i, s in sources.items()}
        infinity = torch.full(size, float('inf'), device=device)
        depths = torch.stack([torch.stack([projected[i] for i in ids]).amin(0) if ids else infinity
                              for ids in groups])
        nearest = depths.amin(0)
        visible = torch.isfinite(depths) & ((depths - nearest).abs() <= .005 + .05 * nearest)
        result.append(visible)
    masks = torch.stack(result, dim=1)  # class, target-time, H, W
    exclusive = masks[2:] & ~(masks[0] | masks[1])[None]
    def area(x):
        return F.adaptive_avg_pool2d(x.float(), (16, 28)).cpu().numpy()
    return area(masks), area(exclusive)


@torch.no_grad()
def run_group(producer, row, entry, root, out, index):
    from foundation_contract import normalize_cameras
    from foundation_depth import causal_depth
    from gmlib.training.gate_omega_recovery import _reliability
    from gmlib.models.vae_stream import VaeStream
    r = entry['recipe']; directory = root / 'components' / r['stream_id']
    annotation = json.loads((directory / 'anchor.json').read_text())
    old_meta = json.loads(geometry_path(root, r, '.json').read_text())
    validate_geometry_policy(old_meta, r)
    if old_meta['recipe'] != r or file_sha(old_meta['path']) != old_meta['sha256']:
        raise ValueError('clean cache mismatch')
    with np.load(directory / 'anchor.npz') as z:
        ad, am = torch.from_numpy(z['depth'].copy()), torch.from_numpy(z['confidence'].copy())
    stream = RecordedStream(row)
    poses = normalize_cameras(stream.c2w, stream.c2w[0], annotation['registered_units_per_anchor_depth'])
    boundary = r['prefix_frames'] - 1; source_ids = r['source_frame_ids']
    archive_ids = {4 * i for i in r['archive_indices']}
    history_ids = {4 * i for i in r['history_latent_indices']}
    native_ids = {4 * i for i in entry['native_selection']['archive_indices']}
    all_ids = set(source_ids) | history_ids | native_ids
    assert row['split'] == r['split'] == 'train' and max(all_ids) <= boundary
    images = {i: stream.frame(i) for i in all_ids}
    rgb = {i: torch.from_numpy(v).permute(2, 0, 1).float() / 127.5 - 1 for i, v in images.items()}
    depths = {0: ad}; confidence = {0: am}; fits = []
    timings = {}; torch.cuda.reset_peak_memory_stats(); start = time.monotonic()
    def estimate(ids):
        t = time.monotonic()
        seed = int(digest(['physical_depth_v2', r['recipe_id'], ids])[:8], 16)
        def inference():
            # Lazy model construction consumes RNG; exclude it from the inference
            # seed scope so the first call and all repeats see the same RNG state.
            producer.da3._load()
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(seed)
                return causal_depth(producer.da3, np.stack([images[i] for i in ids]), stream.K, poses[ids], ad, am)
        ds, masks, fit = inference()
        torch.cuda.synchronize()
        check = None
        if index < 2 and ids == source_ids:
            repeat_depth, repeat_masks, _ = inference()
            relative_l1 = float((repeat_depth - ds).abs().mean() / ds.abs().mean().clamp_min(1e-6))
            check = dict(relative_depth_L1=relative_l1, masks_equal=torch.equal(masks, repeat_masks))
            if relative_l1 > 1e-5 or not check['masks_equal']:
                raise ValueError('seeded depth repeatability failed: ' + str(check))
        fits.append(dict(source_ids=ids, seed=seed, fit=fit, same_input_repeatability=check))
        for j, i in enumerate(ids):
            if i not in depths:
                depths[i] = ds[j]; confidence[i] = _reliability(ds[j]) * masks[j]
        return time.monotonic() - t
    # Preserve original joint source-depth context for the 2x2 render controls.
    timings['original_source_depth_seconds'] = estimate(source_ids)
    extra = sorted(all_ids - set(source_ids)); timings['extra_support_depth_seconds'] = 0.
    for k in range(0, len(extra), 4):
        timings['extra_support_depth_seconds'] += estimate([0, *extra[k:k + 4]])
    sources = {i: dict(frame_id=i, rgb=rgb[i], depth=depths[i], confidence=confidence[i],
                      c2w=poses[i], gauge_id=annotation['gauge_id']) for i in all_ids}
    render_sources = [sources[i] for i in source_ids]
    targets = poses[boundary + 1:boundary + 17]
    t = time.monotonic()
    masks, exclusive = support_masks(sources, [{0}, history_ids, archive_ids, native_ids],
                                    targets[[3, 7, 11, 15]], stream.K)
    torch.cuda.synchronize(); timings['support_projection_seconds'] = time.monotonic() - t
    arrays = dict(support_area=masks.astype(np.float16), exclusive_area=exclusive.astype(np.float16),
                  support_bits=np.packbits(masks > .5), exclusive_bits=np.packbits(exclusive > .5))
    variants = []; clean_rgb = None
    for a, g in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        branch = perturb(render_sources, archive_ids, a, g, entry['geometry_error'])
        name = f'A{a}G{g}'; t = time.monotonic()
        rendered = producer.renderer.render(branch, targets, stream.K, (256, 448), seed=0,
            camera_units='anchor_median_depth', gauge_id=annotation['gauge_id'], history_boundary=boundary)
        torch.cuda.synchronize(); render_seconds = time.monotonic() - t
        t = time.monotonic(); video = rendered['rgb'].permute(1, 0, 2, 3)[None]
        video = torch.cat([video[:, :, :1], video], dim=2)
        z = VaeStream(producer.vae).encode_chunk(video.cuda().to(torch.bfloat16))[:, :, 1:].float().cpu().numpy()
        torch.cuda.synchronize(); encode_seconds = time.monotonic() - t
        meta = torch.stack([F.interpolate(rendered[k].reshape(4, 4, 1, 256, 448).mean(1), size=(16, 28), mode='area')[:, 0]
                            for k in ('valid', 'conf')])[None].numpy()
        assert z.shape == (1, 48, 4, 16, 28) and np.isfinite(z).all() and np.isfinite(meta).all()
        arrays[name + '_render_bf16'] = pack_bf16(z); arrays[name + '_metadata'] = meta
        if clean_rgb is None: clean_rgb = rendered['rgb'].clone()
        rgb_delta = float((rendered['rgb'] - clean_rgb).abs().mean())
        record = dict(name=name, render_seconds=render_seconds, render_VAE_seconds=encode_seconds,
            RGB_L1_to_clean=rgb_delta, coverage=float(rendered['valid'].mean()),
            source_hashes=[{k: tensor_hash(s[k]) for k in ('rgb', 'depth', 'confidence', 'c2w')} for s in branch])
        if a == g == 0:
            with np.load(old_meta['path']) as cache:
                delta = np.abs(z - unpack_bf16(cache['render_bf16']))
                record['clean_cache_replay_L1'] = float(delta.mean())
                record['clean_cache_replay_max_abs'] = float(delta.max())
        variants.append(record)
    # Factor identities must hold at the actual consumed renderer inputs.
    for j in range(len(source_ids)):
        hs = [v['source_hashes'][j] for v in variants]
        for k in ('depth', 'confidence', 'c2w'):
            assert hs[0][k] == hs[1][k] and hs[2][k] == hs[3][k]
        assert hs[0]['rgb'] == hs[2]['rgb'] and hs[1]['rgb'] == hs[3]['rgb']
    from materialize_all_scenes import npz
    artifact = out / 'groups' / f'{r["group_id"]}.npz'; artifact.parent.mkdir(exist_ok=True)
    npz(artifact, **arrays)
    record = dict(index=index, recipe=r, geometry_error=entry['geometry_error'],
        selection_stratum=entry['selection_stratum'], native_selection=entry['native_selection'],
        source_RGB_hashes={str(i): file_sha(stream.paths[i]) for i in sorted(all_ids)},
        anchor_metadata_sha256=file_sha(directory / 'anchor.json'), clean_geometry_sha256=old_meta['sha256'],
        command_pose_sha256=tensor_hash(targets), gauge_id=annotation['gauge_id'],
        depth_batches=fits, source_classes=[sorted(x) for x in [{0}, history_ids, archive_ids, native_ids]],
        estimated_archive_exclusive_fraction=exclusive.mean(axis=(1, 2, 3)).tolist(),
        static_surface_qualified=False, target_RGB_read=False, input_factorization_verified=True,
        variants=variants, timings=timings, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), wall_seconds=time.monotonic() - start,
        artifact=str(artifact), artifact_bytes=artifact.stat().st_size, artifact_sha256=file_sha(artifact))
    save(artifact.with_suffix('.json'), record)
    return record


def summarize(plan, records, out, setup_seconds, started):
    times = [r['wall_seconds'] for r in records]
    variant_times = [sum(v['render_seconds'] + v['render_VAE_seconds'] for v in r['variants'][1:]) for r in records]
    unit_bytes = [r['artifact_bytes'] for r in records]
    summary = dict(schema=SCHEMA, status='complete' if len(records) == plan['group_count'] else 'partial',
        completed=len(records), planned=plan['group_count'], setup_seconds=setup_seconds,
        invocation_wall_seconds=time.monotonic() - started, measured_group_wall_seconds=sum(times),
        plan_sha256=file_sha(out / 'plan.json'), producer_code_sha256=file_sha(Path(__file__)),
        median_group_seconds=float(np.median(times)) if times else None,
        mean_group_seconds=float(np.mean(times)) if times else None,
        p90_group_seconds=float(np.quantile(times, .9)) if times else None,
        median_three_extra_render_and_encode_seconds=float(np.median(variant_times)) if times else None,
        peak_allocated_bytes=max((r['peak_allocated_bytes'] for r in records), default=0),
        output_bytes=sum(unit_bytes), output_mean_bytes=float(np.mean(unit_bytes)) if times else None,
        projection_base_recipes=16898,
        all_train_continuation_projection_hours=float(np.mean(times) * 16898 / 3600) if times else None,
        all_train_continuation_projection_bytes=int(np.mean(unit_bytes) * 16898) if times else None,
        projection_limits='64_unique_scenes_half_high_proxy_not_corpus_weighted; includes_clean_replay_and_extra_support; no_native_corrupt_encode_or_training_or_observer_cost',
        static_masks_qualified=False, training_labels_produced=False, main_training_started=False,
        group_records=[str(out / 'groups' / (r['recipe']['group_id'] + '.json')) for r in records])
    save(out / 'summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['plan', 'run'])
    p.add_argument('--root', type=Path, default=DATA); p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--groups', type=int, default=64); p.add_argument('--limit', type=int, default=0)
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(20261010)
    np.random.seed(20261010)
    if args.stage == 'plan':
        plan = make_plan(args.root, args.out, args.groups)
        print(json.dumps(dict(groups=plan['group_count'], strata=dict(Counter(x['selection_stratum'] for x in plan['groups'])), path=str(args.out / 'plan.json')))); return
    plan = json.loads((args.out / 'plan.json').read_text())
    for name, sha in plan['source_files_sha256'].items():
        if file_sha(name) != sha: raise ValueError('pilot source changed: ' + name)
    lock = (args.root / 'producer.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pilot_lock = (args.out / 'pilot.lock').open('a'); fcntl.flock(pilot_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = {r['stream_id']: r for r in map(json.loads, (args.root / 'corpus/streams.jsonl').open())}
    started = time.monotonic(); producer = None; setup_seconds = 0.; records = []
    for index, entry in enumerate(plan['groups']):
        if args.limit and index >= args.limit: break
        meta = args.out / 'groups' / (entry['recipe']['group_id'] + '.json')
        if meta.exists():
            rec = json.loads(meta.read_text())
            if file_sha(rec['artifact']) != rec['artifact_sha256']: raise ValueError('pilot artifact changed')
            records.append(rec); continue
        if shutil.disk_usage(args.out).free < 8.2 * 2**30: raise OSError('8 GiB reserve; no deletion')
        if sum(r['artifact_bytes'] for r in records) > 2**30: raise OSError('1 GiB pilot cap reached')
        if producer is None:
            from materialize_all_scenes import Producer
            t = time.monotonic(); producer = Producer('geometry', args.out)
            setup_seconds = time.monotonic() - t; save(args.out / 'producer.json', producer.provenance)
        try:
            rec = run_group(producer, rows[entry['recipe']['stream_id']], entry, args.root, args.out, index)
        except Exception as exc:
            save(args.out / 'failure.json', dict(index=index, group_id=entry['recipe']['group_id'], error=repr(exc)))
            raise
        records.append(rec)
        summary = summarize(plan, records, args.out, setup_seconds, started)
        print('GROUP', len(records), '/', plan['group_count'], 'seconds', round(rec['wall_seconds'], 2),
              'peak_GiB', round(rec['peak_allocated_bytes'] / 2**30, 2),
              'exclusive', rec['estimated_archive_exclusive_fraction'], flush=True)
    print(json.dumps(summarize(plan, records, args.out, setup_seconds, started), indent=2), flush=True)


if __name__ == '__main__':
    main()
