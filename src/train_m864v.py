"""Memorizon-recipe settings (lr 2e-5/1e-4, lora 0, prompt dropout 0) WITH our Plucker/metadata/role inputs trained, continued
from plain-350 on the matched train_b864 rows (seed 20261023): --render_p 0 is the no-memory twin, --render_p .75 the memory
twin (10-23: ours-no-memory 600 lost to the Memorizon recipe 600 on camera, .558 vs .209; this removes our three training
settings at once and keeps our inputs). plain-350 holds no weights for those inputs (frozen at zero there, as in our own
350 init), so the init load accepts exactly that gap and leaves them at their zero init."""
import json, os
from pathlib import Path
import torch
import model_io
import train_b864  # noqa: F401  (installs the matched-window sampler + precomputed render cache)
import train

EXTRA = ('camera.', 'metadata.', 'roles.')


def load_init(model, receipt_path, strict_sha=True):
    receipt = json.loads(Path(receipt_path).read_text())
    if strict_sha and model_io.sha256(receipt['path']) != receipt['sha256']:
        raise ValueError('checkpoint sha mismatch: ' + receipt['path'])
    state = torch.load(receipt['path'], map_location='cpu')['trainables']
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    missing, extra = set(params) - set(state), set(state) - set(params)
    if extra or any(not n.startswith(EXTRA) for n in missing):
        raise ValueError(f'trainable scope differs beyond our input branches: missing {sorted(missing)[:4]} extra {sorted(extra)[:4]}')
    if any(float(params[n].abs().max()) != 0. for n in missing):
        raise ValueError('our input branches are not at their zero init')
    with torch.no_grad():
        for n in state:
            params[n].copy_(state[n].to(params[n]))
    return receipt


train.load_trainables = load_init

if __name__ == '__main__':
    import sys
    if float(sys.argv[sys.argv.index('--render_p') + 1]) > 0 and not (train_b864.RENDERS / '_done').exists():
        raise SystemExit('precomputed renders missing: run precompute_renders_864.py first')
    train.main(); os._exit(0)
