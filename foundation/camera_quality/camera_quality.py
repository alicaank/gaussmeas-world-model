"""One camera-quality manifest for every data path (dataset_v3 streams and Memorizon shards300 windows).

Poses in the anchor-depth gauge (registered units / registered_units_per_anchor_depth). Per stream (all frames) and per
shard window (the 161 frames actually used):
  scale : max per-latent (4-frame) translation step > 2 anchor depths  (broken registration scale, e.g. units ~1e-5)
  jump  : a single frame-to-frame step > max(0.5 anchor depth, 20 x the stream's median step)  (raw pose discontinuity)
  nonfinite : NaN/inf in the poses
Flagged = any reason. A flagged STREAM is not necessarily bad in every window (0591d638: jump at frames 384->388 only);
the window flags say whether a given shard window is affected. Consumers that sample arbitrary windows use the stream
flags (conservative); shard consumers use the window flags.
Output: manifest.json {thresholds, streams: {id: {reasons, max_latent_step, max_frame_step, jump_frames}}, shards: {...}}
"""
import gm_env  # machine paths: gm_env.py at the repository root (scripts/env.sh puts it on PYTHONPATH)
import json
from pathlib import Path
import numpy as np
DATA = Path((gm_env.FOUNDATION + '/dataset_v3'))
SHARDS = Path((gm_env.FOUNDATION + '/finetune/shards300'))
OUT = Path(__file__).parent / 'manifest.json'
TH = dict(latent_step=2.0, jump_abs=0.5, jump_rel=20.0)


def check(c2w):
    t = c2w[:, :3, 3]
    if not np.isfinite(c2w).all():
        return ['nonfinite'], dict(max_latent_step=None, max_frame_step=None, jump_frames=[])
    fs = np.linalg.norm(np.diff(t, axis=0), axis=1); ls = np.linalg.norm(np.diff(t[::4], axis=0), axis=1)
    med = float(np.median(fs)) if len(fs) else 0.0
    jumps = np.nonzero(fs > max(TH['jump_abs'], TH['jump_rel'] * med))[0]
    reasons = (['scale'] if len(ls) and ls.max() > TH['latent_step'] else []) + (['jump'] if len(jumps) else [])
    return reasons, dict(max_latent_step=float(ls.max()) if len(ls) else 0.0, max_frame_step=float(fs.max()) if len(fs) else 0.0,
                         jump_frames=[int(j) for j in jumps[:20]])


def main():
    rows = [json.loads(l) for l in open(DATA / 'corpus/streams.jsonl')]
    streams, shards = {}, {}
    for r in rows:
        p = DATA / 'components' / r['stream_id'] / 'vae.npz'
        if not p.exists():
            continue
        u = json.loads((DATA / 'components' / r['stream_id'] / 'anchor.json').read_text())['registered_units_per_anchor_depth']
        c = np.load(p)['c2w'].astype(np.float64); c = np.linalg.inv(c[0]) @ c; c[:, :3, 3] /= u
        reasons, info = check(c)
        if reasons:
            streams[r['stream_id']] = dict(split=r['split'], scene_id=r['scene_id'], reasons=reasons, **info)
    for ep in (json.load(open(SHARDS / 'index.json'))['episode_ids'] if SHARDS.exists() else []):   # optional: an older shard set, not part of this pipeline
        c = np.load(SHARDS / 'aux' / f'{ep}.npz')['c2w_frame'].astype(np.float64)
        reasons, info = check(c)
        if reasons:
            shards[ep] = dict(reasons=reasons, stream_flagged=ep in streams, **info)
    out = dict(thresholds=TH, n_streams_checked=len(rows), streams=streams, shards=shards,
               counts=dict(streams=len(streams), streams_scale=sum('scale' in v['reasons'] for v in streams.values()),
                           streams_jump=sum('jump' in v['reasons'] for v in streams.values()), shard_windows=len(shards)))
    json.dump(out, open(OUT, 'w'), indent=1); print(json.dumps(out['counts']))


if __name__ == '__main__':
    main()
