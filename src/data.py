"""Packed training rows from the recorded corpus (dataset_v3): no future RGB, no render, no sampler.

Row = one stream window: anchor latent 0, a pose-retrieved bank (<=6 past latents outside the recent block),
the <=4 latest latents before the first chunk, and N chunks of 4 consecutive recorded latents. Cameras are the
offline-registered poses in anchor-depth units (anchor.json), RGB-rate; a latent p>0 covers RGB frames 4p-3..4p.
"""
import json
import random
from pathlib import Path
import numpy as np
import torch
from all_scene_dataset import DATA, unpack_bf16
from foundation_contract import normalize_cameras
from foundation_model import packed_rays
from packed_model import ROLE_ANCHOR, ROLE_RECENT, ROLE_BANK, ROLE_RENDER, ROLE_TARGET, rope_positions

A = Path('/home/hj/Alican/analysis/plan_base_restart_20261008')
CAMERA_QUALITY = A / 'camera_quality_20261015/manifest.json'
DEV10 = Path('/home/hj/Alican/training_runs/joint_samplerfix_20261012/dataset_v3/sampler/dev_monitor_10.json')
N_CHOICES, N_PROBS = (2, 4, 6, 8), (.15, .25, .30, .30)


def corpus(split='train', exclude_flagged=True):
    flagged = set(json.load(open(CAMERA_QUALITY))['streams']) if exclude_flagged else set()
    rows = [json.loads(l) for l in (DATA / 'corpus/streams.jsonl').open()]
    rows = [r for r in rows if r['split'] == split and r['stream_id'] not in flagged and r['latent_count'] >= 9]
    return rows, flagged


def dev_scenes():
    return json.load(open(DEV10))['scenes']


DATA864 = Path(__file__).parent / 'data864'
TEXT_PAD = 512


def pad_text(t, n=TEXT_PAD):
    """(1,L,4096) -> (1,n,4096) zero-padded, as Wan/Memorizon's umT5 inputs (text_max_sequence_length 512)."""
    if t.shape[1] >= n:
        return t[:, :n]
    return torch.cat([t, t.new_zeros((t.shape[0], n - t.shape[1], t.shape[2]))], 1)


def corpus864(split='train'):
    rows = [json.loads(l) for l in (DATA864 / 'streams.jsonl').open()]
    return [r for r in rows if r['split'] == split and r['latent_count'] >= 9 and r.get('mpu_valid', True)], set()


class Stream864:
    """data864 component: latents (1,48,L,30,54), K864, canonical c2w (anchor gauge); text from the 448 corpus; camera scale
    = metres_per_anchor_unit * scale_mult / 4 (Memorizon units are metres/4; the zero-shot test preferred x2)."""

    def __init__(self, row, scale_mult=2., flip_y=False):
        d = DATA864 / 'components' / row['stream_id']
        with np.load(d / 'vae.npz') as z:
            self.latents = torch.from_numpy(unpack_bf16(z['latents_bf16'])); self.K = torch.from_numpy(z['K'].copy()); self.c2w = torch.from_numpy(z['c2w'].copy())
        with np.load(row['text']) as z:
            self.text = pad_text(torch.from_numpy(unpack_bf16(z['text_bf16'])))                    # Memorizon was trained with 512-token padded T5
        self.row = row; self.L = self.latents.shape[2]; self.gauge_id = row['gauge_id']; self.metres_per_unit = row['metres_per_anchor_unit']
        self.cam_scale = self.metres_per_unit * scale_mult / 4.; self.cam_flip_y = flip_y

    def frame_index(self, p):
        return 0 if p == 0 else 4 * p


class Stream:
    """Everything a row needs, loaded once per stream."""

    def __init__(self, row):
        d = DATA / 'components' / row['stream_id']; ann = json.loads((d / 'anchor.json').read_text())
        with np.load(d / 'vae.npz') as z:
            self.latents = torch.from_numpy(unpack_bf16(z['latents_bf16']))          # (1,48,L,16,28)
            self.K = torch.from_numpy(z['K'].copy()); c2w = torch.from_numpy(z['c2w'].copy())
        self.c2w = normalize_cameras(c2w, c2w[0], ann['registered_units_per_anchor_depth'])   # RGB-rate, anchor = identity
        with np.load(d / 'text.npz') as z:
            self.text = torch.from_numpy(unpack_bf16(z['text_bf16']))                # (1,40,4096)
        self.row = row; self.L = self.latents.shape[2]; self.gauge_id = ann['gauge_id']
        if self.c2w.shape[0] < 1 + 4 * (self.L - 1):
            raise ValueError('fewer poses than latents')

    def frame_index(self, p):
        return 0 if p == 0 else 4 * p


def select_bank(observed_c2w, requested_c2w, K, prefix_latents, recent=4, max_refs=6, size=(256, 448), exclude=()):
    """joint_v9.retrieval.select_archive with a 4-latent recent block: greedy frustum-coverage gain of the
    requested 16-frame command over anchor + recent, candidates = latents 1..prefix-recent-1."""
    candidates = [c for c in range(1, max(1, prefix_latents - recent)) if c not in exclude]   # exclude: idea-2 memory-contribution test
    if not candidates or max_refs == 0:
        return []
    h, w = size
    yy, xx = torch.meshgrid(torch.linspace(.1 * h, .9 * h, 6), torch.linspace(.1 * w, .9 * w, 8), indexing='ij')
    ray = torch.linalg.inv(K.float()) @ torch.stack([xx, yy, torch.ones_like(xx)]).reshape(3, -1)
    world = []
    for target in requested_c2w[[3, 7, 11, 15]]:
        for d in (.5, 1., 2., 4.):
            world.append(target[:3, :3] @ (ray * d) + target[:3, 3, None])
    world = torch.cat(world, 1)

    def covers(position):
        camera = torch.linalg.inv(observed_c2w[4 * position])
        xyz = camera[:3, :3] @ world + camera[:3, 3, None]; uv = K @ xyz; xy = uv[:2] / xyz[2].clamp_min(1e-8)
        return (xyz[2] > .1) & (xy[0] >= 0) & (xy[0] < w) & (xy[1] >= 0) & (xy[1] < h)
    covered = covers(0)
    for pos in range(max(1, prefix_latents - recent), prefix_latents):
        covered |= covers(pos)
    selected = []; visibility = {i: covers(i) for i in candidates}
    for _ in range(min(max_refs, len(candidates))):
        scored = [(int((visibility[i] & ~covered).sum()), -i, i) for i in candidates if i not in selected]
        gain, _, best = max(scored)
        if gain == 0:
            break
        selected.append(best); covered |= visibility[best]
    return sorted(selected)


PER_CHUNK_RENDER = False      # idea 4 (rev 10-26, train_r2.py): every target chunk gets its own render group; chunk_index -(2+j) per group


def assemble(stream, bank, recent, chunk_latents, n_chunks, render=False, per_chunk=False):
    """Shared by training rows and the inference roller. chunk_latents: list of true latent indices (4N).
    With render=True four render slots (role 3) sit right before the chunks, at chunk-0's cameras/positions, as context
    (chunk_index -1: shared by every chunk). per_chunk=True: 4N render slots, group j at chunk j's positions with
    chunk_index -(2+j) (visible to chunks >= j only, packed_model.chunk_causal_bias)."""
    r = (list(chunk_latents) if per_chunk else chunk_latents[:4]) if render else []
    true_positions = [0, *bank, *recent, *r, *chunk_latents]
    roles = torch.tensor([ROLE_ANCHOR] + [ROLE_BANK] * len(bank) + [ROLE_RECENT] * len(recent) + [ROLE_RENDER] * len(r) + [ROLE_TARGET] * 4 * n_chunks)
    pos = rope_positions(len(bank), len(recent), n_chunks)
    positions = torch.cat([pos[:1 + len(bank) + len(recent)], pos[-4 * n_chunks:][:len(r)], pos[-4 * n_chunks:]])
    rci = [-(2 + g) for g in range(len(r) // 4) for _ in range(4)] if per_chunk else [-1] * len(r)
    chunk_index = torch.tensor([-1] * (1 + len(bank) + len(recent)) + rci + [j for j in range(n_chunks) for _ in range(4)])
    return true_positions, roles, positions, chunk_index


def build_row(stream, rng, n_chunks=None, start=None, first_call_p=.2, render_builder=None, render_p=0., max_bank=6):
    L = stream.L
    if n_chunks is None:
        n_chunks = rng.choices(N_CHOICES, N_PROBS)[0]
    n_chunks = min(n_chunks, (L - 1) // 4)
    if start is None:
        hi = L - 4 * n_chunks                                                            # short streams: only the first-call window exists
        start = 1 if hi < 2 or rng.random() < first_call_p else rng.randint(2, hi)
    if start < 1 or start + 4 * n_chunks > L:
        raise ValueError('window outside the stream')
    recent = list(range(max(1, start - 4), start))
    import packed_model
    if start == 1 and packed_model.LAYOUT == 'memorizon':
        recent = [0]                                                                     # Memorizon's first-call layout: A duplicated as recent
    H, W = stream.latents.shape[-2:]; size = (16 * H, 16 * W)
    command = stream.c2w[4 * start - 3: 4 * start + 13]                                   # chunk-0 command, RGB frames of latents start..start+3
    bank = select_bank(stream.c2w[:4 * (start - 1) + 1], command, stream.K, start, size=size, max_refs=max_bank) if start > 1 else []
    chunks = list(range(start, start + 4 * n_chunks))
    render = render_builder is not None and random.Random(rng.random()).random() < render_p   # consumes one draw whether or not a builder exists
    per_chunk = bool(render and PER_CHUNK_RENDER)
    true_positions, roles, positions, chunk_index = assemble(stream, bank, recent, chunks, n_chunks, render=render, per_chunk=per_chunk)
    poses_needed = 4 * chunks[-1] + 1
    poses = stream.c2w[:poses_needed].clone(); poses[:, :3, 3] *= getattr(stream, 'cam_scale', 1.)    # model camera units (864/Memorizon: metres/2)
    if getattr(stream, 'cam_flip_y', False):                                                 # Memorizon's rotation convention: conjugate by diag(1,-1,1,1)
        Fy = torch.diag(torch.tensor([1., -1., 1., 1.])); poses = Fy @ poses @ Fy
    rays = packed_rays(poses, stream.K, true_positions, height=size[0], width=size[1])    # (1,24,F,h,w); only past+chunk poses
    c2w = torch.stack([poses[stream.frame_index(p)] for p in true_positions])
    context = stream.latents[:, :, [0, *bank, *recent]]; target = stream.latents[:, :, chunks]; metadata = None
    if render:
        rz, meta = render_builder.render_chunk0(stream, start, bank); nctx = 1 + len(bank) + len(recent)
        metadata = torch.zeros((1, meta.shape[1], len(true_positions), *rz.shape[-2:])); metadata[:, :, nctx:nctx + 4] = meta   # channels: (valid, conf) or + the feature pilot's 3
        for j in range(1, n_chunks if per_chunk else 1):                                          # chunk j's own render: recorded prefix up to latent start+4j-1
            sj = start + 4 * j; cmd_j = stream.c2w[4 * sj - 3: 4 * sj + 13]
            bank_j = select_bank(stream.c2w[:4 * (sj - 1) + 1], cmd_j, stream.K, sj, size=size, max_refs=max_bank)
            rz_j, meta_j = render_builder.render_chunk0(stream, sj, bank_j); rz = torch.cat([rz, rz_j], 2); metadata[:, :, nctx + 4 * j:nctx + 4 * j + 4] = meta_j
        context = torch.cat([context, rz], 2)
    return dict(context=context, target=target, rays=rays, roles=roles, positions=positions, chunk_index=chunk_index, c2w=c2w, metadata=metadata,
                K=stream.K, text=stream.text, n_chunks=n_chunks, start=start, bank=bank, recent=recent, render=render, stream_id=stream.row['stream_id'])


class RowSampler:
    """Scene-balanced stream draw; streams are cached (one row per stream per draw)."""

    def __init__(self, rows, seed, cache=256, render_builder=None, render_p=0., stream_cls=None, n_choices=None, n_probs=None, max_bank=6):
        self.rng = random.Random(seed); self.by_scene = {}; self.render_builder = render_builder; self.render_p = render_p
        self.stream_cls = stream_cls or Stream; self.n_choices = n_choices or N_CHOICES; self.n_probs = n_probs or N_PROBS; self.max_bank = max_bank
        for r in rows:
            self.by_scene.setdefault(r['scene_id'], []).append(r)
        self.scenes = sorted(self.by_scene); self.pending = []; self.cache = {}; self.cache_max = cache

    def draw(self):
        if not self.pending:
            self.pending = list(self.scenes); self.rng.shuffle(self.pending)
        row = self.rng.choice(self.by_scene[self.pending.pop()]); sid = row['stream_id']
        if sid not in self.cache:
            if len(self.cache) >= self.cache_max:
                self.cache.pop(next(iter(self.cache)))
            self.cache[sid] = self.stream_cls(row)
        return build_row(self.cache[sid], self.rng, n_chunks=self.rng.choices(self.n_choices, self.n_probs)[0], render_builder=self.render_builder, render_p=self.render_p, max_bank=self.max_bank)

    def state_dict(self):
        return dict(rng=self.rng.getstate(), pending=list(self.pending))

    def load_state_dict(self, s):
        self.rng.setstate(s['rng']); self.pending = list(s['pending'])
