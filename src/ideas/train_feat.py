"""Gaussian-feature memory pilot (status page / user 10-26): can the 3D memory hand the generator more than an RGB render?
Arm 'feat' = the control C's training (train_m864v recipe from run_v864/mem, same seed / rows / windows / renders) with the
render slots' metadata widened from (valid, conf) to (valid, conf, depth, conflict, support) -- renderer864.render_features:
fused z-buffer depth, the per-source depth disagreement (contradictions) and the source-coverage count (matches) of the
memory views at the target cameras, precomputed by precompute_feat_864.py for the same windows. The metadata projection is
widened at init with the trained 2-channel weights and ZERO weights on the new channels, so the arm starts as exactly the
control; md_ch=5 is written into the checkpoint lineage (closed_loop_864 builds the model from it and computes the features
online from the model's own frames). Matched control: run_op/C (250 updates, seed 20261025).
Usage: train_feat.py <train_m864v.py args incl. --init_receipt run_v864/mem/... --render_p .75>"""
import json, os, sys
from pathlib import Path
import numpy as np, torch
import train_m864v  # noqa: F401  (load_init + the cached render rows)
import train_b864, render_rows, train, model_io
from data import DATA864

MD_CH = 5
RENDERS_FEAT = DATA864 / 'renders_feat'


class CachedBuilderFeat(train_b864.CachedBuilder):
    def render_chunk0(self, stream, start, bank):
        rz, meta = super().render_chunk0(stream, start, bank)
        with np.load(RENDERS_FEAT / stream.row['stream_id'] / f'start_{start}.npz') as z:
            if [int(b) for b in z['bank']] != [int(b) for b in bank]:
                raise RuntimeError('feature cache bank differs from the render cache bank')
            feat = torch.from_numpy(z['feat_f16'].astype(np.float32))[None]                     # (1,3,4,h,w)
        return rz, torch.cat([meta, feat], 1)


def fresh_model_feat(**kw):
    return model_io.fresh_model(md_ch=MD_CH, **kw)


_load_init = train_m864v.load_init


def load_init_feat(model, receipt_path, strict_sha=True):
    """train_m864v.load_init on a 2-channel checkpoint into the 5-channel model: metadata.weight[:, :2] <- trained, [:, 2:] <- 0."""
    receipt = json.loads(Path(receipt_path).read_text())
    if strict_sha and model_io.sha256(receipt['path']) != receipt['sha256']:
        raise ValueError('checkpoint sha mismatch: ' + receipt['path'])
    state = torch.load(receipt['path'], map_location='cpu')['trainables']
    w = state['metadata.weight']
    if w.shape[1] == MD_CH:
        return _load_init(model, receipt_path, strict_sha)
    if w.shape[1] != 2:
        raise ValueError(f'metadata.weight has {w.shape[1]} input channels')
    wide = torch.zeros((w.shape[0], MD_CH, *w.shape[2:]), dtype=w.dtype); wide[:, :2] = w; state['metadata.weight'] = wide
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    missing, extra = set(params) - set(state), set(state) - set(params)
    if extra or any(not n.startswith(train_m864v.EXTRA) for n in missing):
        raise ValueError(f'trainable scope differs beyond our input branches: missing {sorted(missing)[:4]} extra {sorted(extra)[:4]}')
    with torch.no_grad():
        for n in state:
            params[n].copy_(state[n].to(params[n]))
    print('[feat] metadata projection widened 2 -> 5 channels (new channels zero)', flush=True); return receipt


def save_trainables_feat(model, path, meta):
    meta = dict(meta); meta['lineage'] = dict(meta.get('lineage') or {}, md_ch=MD_CH)
    return model_io.save_trainables(model, path, meta)


render_rows.RenderBuilder = CachedBuilderFeat
train.fresh_model = fresh_model_feat; train.load_trainables = load_init_feat; train.save_trainables = save_trainables_feat

if __name__ == '__main__':
    if not (RENDERS_FEAT / '_done').exists() or not (train_b864.RENDERS / '_done').exists():
        raise SystemExit('precomputed renders / features missing (precompute_renders_864.py, precompute_feat_864.py)')
    out = Path(sys.argv[sys.argv.index('--out') + 1]); out.mkdir(parents=True, exist_ok=True); (out / 'feat.flag').write_text(f'md_ch={MD_CH} features=depth,conflict,support\n')
    train.main(); os._exit(0)
