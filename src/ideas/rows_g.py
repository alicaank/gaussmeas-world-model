"""Rows for the 2Xplat-memory arms G / G+B (PILOT_PLAN_CVPR_20261104.md): visibility-addressed bank selection, the
map-addressed correspondence matrix of a packed row, pool rows of the round-2 on-policy pool (gen_onpolicy2.py), and the cached
builder for recorded windows rendered by precompute_renders_2x.py. CPU only; tested by tests_rows_g.py.

Token geometry: 864x480 frames -> latent 30x54 -> 2x2 patches -> 15x27 tokens per slot, 32 px per token (PIX). Depth maps at
token resolution ('dtok') are gauge-unit z-depths of the scene rendered at a slot's camera; nan = no coverage.
Packed order (data.assemble): [anchor | bank | recent | render x4 | chunk0 x4 | chunk1 x4 ...]; the correspondence matrix maps
target tokens to anchor/bank/recent tokens only (render slots sit at the target cameras already)."""
from pathlib import Path
import numpy as np, torch
import map_geometry as mg
from data import assemble, select_bank, DATA864
from foundation_model import packed_rays
from all_scene_dataset import unpack_bf16
from closed_loop import interp_rgb_poses

PIX = 32; HT, WT = 15, 27; TPS = HT * WT
POOL2 = DATA864 / 'onpolicy2'; RENDERS2X = DATA864 / 'renders_2x'
N_CHUNKS = 2


def bank_by_visibility(K, command_c2w, dtok_command, candidates, cand_c2w, cand_dtok, covered_by, max_refs):
    """Greedy visibility bank: command_c2w (16,4,4) with dtok_command (4,HT,WT) at the latent ends (frames 3,7,11,15);
    candidates = latent indices with their gauge c2w and token depth; covered_by = list of (c2w, dtok) for anchor + recent."""
    tgt = np.asarray(command_c2w, np.float64)[[3, 7, 11, 15]]; dt = np.asarray(dtok_command, np.float64)
    if not candidates:
        return []
    cov = mg.visibility_coverage(K, tgt, dt, [np.asarray(c, np.float64) for c in cand_c2w], HT, WT, PIX, candidate_depth_tok=list(cand_dtok))
    base = mg.visibility_coverage(K, tgt, dt, [np.asarray(c, np.float64) for c, _ in covered_by], HT, WT, PIX, candidate_depth_tok=[d for _, d in covered_by])
    covered0 = base.any(0) if len(base) else np.zeros(cov.shape[1], bool)
    return mg.select_bank_visibility(cov, list(candidates), covered0, max_refs)


def row_correspondences(K, slot_c2w, slot_dtok, roles, chunk_index, n_sources):
    """Sparse (S,S) correspondence matrix for a packed row. slot_c2w (F,4,4) gauge c2w per slot (model camera units are fine as
    long as all slots share them), slot_dtok list of (HT,WT) or None per slot (same units as the translations), roles/chunk_index
    (F,) as in the row; sources = the first n_sources slots (anchor + bank + recent)."""
    F_ = len(slot_c2w); targets = [q for q in range(F_) if int(chunk_index[q]) >= 0]; sources = list(range(n_sources))
    corr = {}
    src_c2w = [np.asarray(slot_c2w[j], np.float64) for j in sources]; src_d = [slot_dtok[j] for j in sources]
    for qi, q in enumerate(targets):
        if slot_dtok[q] is None:
            continue
        c = mg.correspondences(K, np.asarray(slot_c2w[q], np.float64), slot_dtok[q], src_c2w, src_d, HT, WT, PIX)
        for j in range(len(sources)):
            if (c[j] >= 0).any():
                corr[(qi, j)] = c[j]
    return mg.correspondence_matrix(F_, targets, sources, corr, TPS)


def dtok_from_depth(depth_480x864):
    """(n,480,864) gauge depth with nan holes -> (n,HT,WT) token depth (nan-aware mean over 32x32 blocks, nan if < 25% covered)."""
    d = np.asarray(depth_480x864, np.float64).reshape(-1, HT, PIX, WT, PIX); ok = np.isfinite(d) & (d > 0)
    cnt = ok.sum((2, 4)); s = np.where(ok, d, 0).sum((2, 4)); out = np.where(cnt >= PIX * PIX // 4, s / np.maximum(cnt, 1), np.nan)
    return out


def build_row_g(item, stream, with_corr):
    """Round-2 pool item -> row (same format as onpolicy_rows.build_onpolicy_row, variant 'meas' labels, 2Xplat render and bank).
    Item keys: s, gen_latents_bf16 (1,48,s,h,w), meas_end (s,4,4), bank_g, rz_g_bf16 (1,48,4,h,w), meta_g (1,2,4,h,w),
    dtok_tgt (8,HT,WT) at the recorded latent ends s..s+7, dtok_src (s,HT,WT) at the measured poses of latents 0..s-1."""
    s = int(item['s']); bank = [int(b) for b in item['bank_g']]; recent = list(range(max(1, s - 4), s)); chunks = list(range(s, s + 4 * N_CHUNKS))
    if chunks[-1] >= stream.L:
        raise ValueError('target outside the stream')
    true_positions, roles, positions, chunk_index = assemble(stream, bank, recent, chunks, N_CHUNKS, render=True, per_chunk=False)
    poses = stream.c2w[:4 * chunks[-1] + 1].clone().double(); labels = np.asarray(item['meas_end'], np.float64)
    for j in range(1, s):                                                      # own history: measured ends, interior RGB poses interpolated
        poses[4 * j - 3:4 * j + 1] = torch.from_numpy(np.stack(interp_rgb_poses(labels[j - 1], [labels[j]])))   # between them (= LoopRoller's write)
    gauge = poses.clone(); poses = poses.float(); poses[:, :3, 3] *= stream.cam_scale
    H, W = stream.latents.shape[-2:]
    rays = packed_rays(poses, stream.K, true_positions, height=16 * H, width=16 * W)
    c2w = torch.stack([poses[stream.frame_index(p)] for p in true_positions])
    gen = torch.from_numpy(unpack_bf16(item['gen_latents_bf16']))
    rz = torch.from_numpy(unpack_bf16(item['rz_g_bf16'])); meta = torch.from_numpy(np.asarray(item['meta_g'], np.float32))
    context = torch.cat([gen[:, :, [0, *bank, *recent]], rz], 2)
    metadata = torch.zeros((1, meta.shape[1], len(true_positions), H, W)); nctx = 1 + len(bank) + len(recent); metadata[:, :, nctx:nctx + 4] = meta
    row = dict(context=context, target=stream.latents[:, :, chunks], rays=rays, roles=roles, positions=positions, chunk_index=chunk_index, c2w=c2w,
               metadata=metadata, K=stream.K, text=stream.text, n_chunks=N_CHUNKS, start=s, bank=bank, recent=recent, render=True,
               stream_id='OPg' + stream.row['stream_id'])
    if with_corr:
        dts, dtt = np.asarray(item['dtok_src'], np.float64), np.asarray(item['dtok_tgt'], np.float64)
        slot_c2w = np.stack([gauge[stream.frame_index(p)].numpy() for p in true_positions])
        slot_d = [dts[p] for p in [0, *bank, *recent]] + [None] * 4 + [dtt[q] for q in range(4 * N_CHUNKS)]
        row['corr'] = row_correspondences(stream.K.numpy(), slot_c2w, slot_d, roles, chunk_index, nctx)
    return row


class CachedBuilder2X:
    """Recorded windows rendered by precompute_renders_2x.py: data864/renders_2x/<sid>/start_<s>.npz with rz_bf16 (48,4,h,w),
    meta_f16 (2,4,h,w), bank (visibility), dtok_tgt (8,HT,WT), dtok_src (start,HT,WT). The row's bank must equal the cached one
    (train_g.py routes data.select_bank through cached_bank for recorded rows)."""

    def __init__(self, vae=None):
        self.index = {k: [int(s) for s in v] for k, v in __import__('json').load(open(RENDERS2X / 'index.json')).items()}

    @staticmethod
    def path(stream, start):
        return RENDERS2X / stream.row['stream_id'] / f'start_{start}.npz'

    def cached_bank(self, stream, start):
        with np.load(self.path(stream, start)) as z:
            return [int(b) for b in z['bank']]

    def render_chunk0(self, stream, start, bank):
        with np.load(self.path(stream, start)) as z:
            if [int(b) for b in z['bank']] != [int(b) for b in bank]:
                raise RuntimeError(f'cached visibility bank {list(z["bank"])} != row bank {list(bank)} for {stream.row["stream_id"][:8]} start {start}')
            rz = torch.from_numpy(z['rz_bf16'].copy()).view(torch.bfloat16).float()[None]; meta = torch.from_numpy(z['meta_f16'].astype(np.float32))[None]
            self.last = dict(dtok_tgt=z['dtok_tgt'].astype(np.float64), dtok_src=z['dtok_src'].astype(np.float64), start=start, stream=stream.row['stream_id'])
        return rz, meta

    def attach_corr(self, row, stream):
        """Add the correspondence matrix to a recorded row just built with this builder (render=True)."""
        if not row.get('render') or self.last is None or self.last['stream'] != stream.row['stream_id'] or self.last['start'] != row['start']:
            return row
        bank, recent = row['bank'], row['recent']; nctx = 1 + len(bank) + len(recent); n = row['n_chunks']
        true_positions = [0, *bank, *recent, *range(row['start'], row['start'] + 4)[:4], *range(row['start'], row['start'] + 4 * n)]
        slot_c2w = np.stack([stream.c2w[stream.frame_index(p)].numpy().astype(np.float64) for p in true_positions])
        dts, dtt = self.last['dtok_src'], self.last['dtok_tgt']
        slot_d = [dts[p] for p in [0, *bank, *recent]] + [None] * 4 + [dtt[q] if q < len(dtt) else None for q in range(4 * n)]
        row['corr'] = row_correspondences(stream.K.numpy(), slot_c2w, slot_d, row['roles'], row['chunk_index'], nctx)
        return row
