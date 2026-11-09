"""Stock model loading, trainable checkpoints with sha receipts, resume slots with a disk floor."""
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
import torch
from packed_model import PackedTeacher

SRC = Path('/home/hj/Alican/training_runs/joint_samplerfix_20261012')
EMPTY_TEXT = SRC / 'empty_text.pt'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def stock_fingerprint():
    from gaussverse import _paths
    from gaussverse.training.gate_omega_checkpoint import wan_source_fingerprint
    expected = json.loads((SRC / 'BASE_MODEL_AUDIT.json').read_text())['recorded_stock_fingerprint']
    actual = wan_source_fingerprint(_paths.WAN22_DIR)['sha256']
    if actual != expected:
        raise RuntimeError('stock Wan weights changed')
    return actual


def fresh_model(prope=True, compress=8, device='cuda', lora_r=0, init='stock', md_ch=2):
    from gaussverse.models.wan22_frame_renderer import load_wan22_dit
    fp = stock_fingerprint(); dit = load_wan22_dit(device=device, dtype=torch.bfloat16)
    model = PackedTeacher(dit, prope=prope, compress=compress, lora_r=lora_r, md_ch=md_ch).to(device); params = model.configure_trainables()
    if init == 'memorizon50':
        info = load_memorizon(model); fp = 'memorizon50:' + sha256(MEMORIZON50 / 'model-00001-of-00003.safetensors')[:12]
    elif init != 'stock':
        raise ValueError('unknown init ' + init)
    return model, params, fp


MEMORIZON50 = Path('/home/hj/Alican/analysis/plan_base_restart_20261008/fresh_teacher_20261021/memorizon50')


def load_memorizon(model, directory=MEMORIZON50, zero_extra=True):
    """Luffuly/memorizon-1280x704-50step (Wan2.2-TI2V-5B, train_backbone=true, PRoPE compress 8): map the diffusers-style
    keys to our diffsynth DiT and the prope_attn branches to our CameraBranch modules. Every tensor must be consumed.
    zero_extra: our Plucker/metadata/roles projections are zeroed so the loaded model behaves as Memorizon at step 0."""
    import glob
    from safetensors.torch import load_file
    from gaussverse.diffsynth.utils.state_dict_converters.wan_video_dit import WanVideoDiTFromDiffusers
    sd = {}
    for f in sorted(glob.glob(str(Path(directory) / 'model-*.safetensors'))):
        sd.update(load_file(f))
    if not sd:
        raise FileNotFoundError('no Memorizon shards under ' + str(directory))
    prope = {k[len('model.'):]: v for k, v in sd.items() if '.prope_attn.' in k}
    base = {k[len('model.'):]: v for k, v in sd.items() if '.prope_attn.' not in k}
    mapped = WanVideoDiTFromDiffusers(base)
    if len(mapped) != len(base):
        missing = [k for k in base if k not in {m for m in base}]
        raise ValueError(f'{len(base) - len(mapped)} backbone tensors not mapped')
    dit_sd = model.dit.state_dict(); used = 0
    with torch.no_grad():
        for k, v in mapped.items():
            key = k
            if key not in dit_sd:                           # LoRA-wrapped linears keep the base weight under .base
                for pat in ('.ffn.0.', '.ffn.2.', '.cross_attn.q.', '.cross_attn.k.', '.cross_attn.v.', '.cross_attn.o.'):
                    if pat in key:
                        key = key.replace(pat, pat + 'base.'); break
            if key not in dit_sd:
                raise KeyError('unmapped backbone key ' + k)
            if dit_sd[key].shape != v.shape:
                raise ValueError(f'shape mismatch {k}: {tuple(dit_sd[key].shape)} vs {tuple(v.shape)}')
            dit_sd[key].copy_(v.to(dit_sd[key].dtype)); used += 1
        if model.prope is None:
            raise ValueError('PRoPE branches required to load Memorizon')
        for k, v in prope.items():
            _, i, _, name, param = k.split('.')                 # blocks.{i}.prope_attn.{q_proj|k_proj|v_proj|out_proj|norm}.{weight|bias}
            target = getattr(getattr(model.prope[int(i)], name), param)
            if target.shape != v.shape:
                raise ValueError(f'prope shape mismatch {k}')
            target.copy_(v.to(target.dtype)); used += 1
        if zero_extra:
            for m in (model.camera, model.metadata, model.roles):
                m.weight.zero_()
                if getattr(m, 'bias', None) is not None:
                    m.bias.zero_()
    if used != len(sd):
        raise ValueError(f'consumed {used} of {len(sd)} tensors')
    return dict(source=str(directory), tensors=used)


def load_vae(device='cuda'):
    from gaussverse.models.wan22_frame_renderer import load_wan22_vae
    return load_wan22_vae(device=device, dtype=torch.bfloat16)


def empty_text():
    return torch.load(EMPTY_TEXT, map_location='cpu')['embedding']


def save_trainables(model, path, meta):
    """Immutable: an existing checkpoint file is never overwritten."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError('checkpoint exists; checkpoints are immutable: ' + str(path))
    state = {n: p.detach().to(torch.bfloat16).cpu() for n, p in model.named_parameters() if p.requires_grad}
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        torch.save(dict(trainables=state, meta=meta), f); f.flush(); os.fsync(f.fileno())
    tmp.replace(path); receipt = dict(path=str(path.resolve()), sha256=sha256(path), bytes=path.stat().st_size, meta=meta, written=time.time())
    path.with_suffix('.json').write_text(json.dumps(receipt, indent=1)); return receipt


def load_trainables(model, receipt_path, strict_sha=True):
    receipt = json.loads(Path(receipt_path).read_text())
    mode=receipt.get('meta',{}).get('lineage',{}).get('context_mode','shared')
    if mode not in ('shared','reference'):
        raise ValueError('unsupported checkpoint context_mode: '+str(mode))
    if strict_sha and sha256(receipt['path']) != receipt['sha256']:
        raise ValueError('checkpoint sha mismatch: ' + receipt['path'])
    state = torch.load(receipt['path'], map_location='cpu')['trainables']
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if params.keys() != state.keys():
        raise ValueError('trainable scope differs from checkpoint')
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(state[n].to(p))
    # New checkpoints declare their context topology; old checkpoints keep the
    # original shared-context semantics. No weights or trainable scopes change.
    model.context_mode=mode
    return receipt


def run_id(out):
    """Stable identity of one training run: its output directory name + a short hash of its absolute path."""
    out = Path(out).resolve(); return out.name + '-' + hashlib.sha256(str(out).encode()).hexdigest()[:8]


def resume_dir(out, needed_bytes, floor_gb):
    """Resume slots (fp32 trainables + optimizer) go to the run dir if that keeps the disk above the floor, else to a
    /dev/shm directory that is private to this run (2026-10-21: a shared /dev/shm slot let an overfit run's state
    resume the main run)."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out).free - needed_bytes > floor_gb * 2 ** 30:
        return out
    alt = Path('/dev/shm/fresh_teacher_resume') / run_id(out); alt.mkdir(parents=True, exist_ok=True); return alt


def save_resume(model, opt, sampler, update, out, floor_gb, extra=None):
    rid = run_id(out)
    state = dict(run_id=rid, trainables={n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad},
                 optimizer=opt.state_dict(), sampler=sampler.state_dict(), update=update,
                 rng=dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()), extra=extra or {})
    size = sum(v.numel() * v.element_size() for v in state['trainables'].values()) * 3 + 2 ** 30
    root = resume_dir(out, size, floor_gb); path = root / 'resume.pt'; tmp = root / 'resume.tmp.pt'
    for stale in (path, tmp):                      # one copy at a time: /dev/shm (32 GB) filled up with old + tmp + new slots on 10-21
        if stale.exists():
            stale.unlink()
    with open(tmp, 'wb') as f:
        torch.save(state, f); f.flush(); os.fsync(f.fileno())
    tmp.replace(path); receipt = dict(path=str(path), sha256=sha256(path), update=update, bytes=path.stat().st_size, run_id=rid)
    (Path(out) / 'resume.json').write_text(json.dumps(receipt, indent=1)); return receipt


def load_resume(model, opt, sampler, out):
    """Refuses a slot whose sha, run identity or update disagree with the receipt."""
    receipt = json.loads((Path(out) / 'resume.json').read_text())
    if sha256(receipt['path']) != receipt['sha256']:
        raise ValueError('resume slot sha mismatch: ' + receipt['path'])
    state = torch.load(receipt['path'], map_location='cpu', weights_only=False)
    if state.get('run_id') != run_id(out) or receipt.get('run_id') != run_id(out) or state['update'] != receipt['update']:
        raise ValueError(f"resume slot belongs to another run or update: {state.get('run_id')} vs {run_id(out)}")
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if params.keys() != state['trainables'].keys():
        raise ValueError('resume trainable scope differs')
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(state['trainables'][n].to(p))
    opt.load_state_dict(state['optimizer']); sampler.load_state_dict(state['sampler'])
    torch.set_rng_state(state['rng']['torch']); torch.cuda.set_rng_state_all(state['rng']['cuda'])
    return state['update'], state['extra']


def resolve_init(lineage):
    """Base identity of a checkpoint: lineage['init'] is a string for first-generation runs ('memorizon50:<sha>' / stock
    fingerprint) and a dict {receipt, sha256, meta} for continuations (--init_receipt); walk the parents to the string.
    (10-22 audit: str(dict).startswith('memorizon50') was False for every child checkpoint -> wrong frozen base.)"""
    init = (lineage or {}).get('init', ''); hops = 0
    while isinstance(init, dict) and hops < 16:
        init = ((init.get('meta') or {}).get('lineage') or {}).get('init', ''); hops += 1
    return 'memorizon50' if str(init).startswith('memorizon50') else 'stock'
