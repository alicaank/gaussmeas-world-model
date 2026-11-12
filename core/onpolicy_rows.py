"""Measured-write training test (user 10-23): on-policy memory rows.

A pool item (gen_onpolicy.py) = one rollout prefix of the current memory model (Memorizon recipe + 3D memory) on a TRAIN
stream: its own generated latents 1..s-1 (open loop, recorded commands), the DA3-measured pose of every own latent end, and
two ReSplat renders of the next command from the same own frames -- one with the frames placed at the COMMANDED poses
(variant 'cmd'), one at the MEASURED poses (variant 'meas'). The target is always the RECORDED next 2 chunks.

build_onpolicy_row() returns a row in build_row()'s format with render=True. Variant 'cmd' labels the own history with the
commanded (= recorded) poses, 'meas' with the measured ones (PRoPE c2w + rays of the context, bank selection, render).
Everything else -- anchor, recent window, chunk positions, target, text, K -- is identical between the variants, so the two
training arms differ ONLY in where the model's own frames are placed."""
from pathlib import Path
import numpy as np, torch
from data import assemble, DATA864
from foundation_model import packed_rays
from all_scene_dataset import unpack_bf16

POOL = DATA864 / 'onpolicy'
N_CHUNKS = 2


def split_variant(variant):
    """'cmd' | 'meas' -> the same source for all three uses; a 3-letter code of c/m = (bank selection, pose labels, render), e.g.
    'mcc' = measured bank only, 'cmc' = measured pose labels only, 'ccm' = measured render only (10-26 follow-up: which part
    of the measured-write gain is the reference selection, the camera conditioning or the render placement)."""
    if variant in ('cmd', 'meas'):
        return variant, variant, variant
    if len(variant) == 3 and set(variant) <= {'c', 'm'}:
        return tuple({'c': 'cmd', 'm': 'meas'}[ch] for ch in variant)
    raise ValueError('variant: cmd | meas | 3-letter c/m code')


def render_sources(item, bank_v):
    """OwnRender864.render_own's source latents for the bank of labelling bank_v: anchor, the last two own latents, the best bank view."""
    s = int(item['s']); bank = [int(b) for b in item[f'bank_{bank_v}']]
    return list(dict.fromkeys([0, s - 1, max(0, s - 2), *bank[:1]]))[:4]


def render_key(item, bank_v, render_v):
    """Which cached render a mixed variant reads (review 10-26, E1): the stock 'cmd'/'meas' render used the sources of ITS OWN
    bank, so a placement-only arm needs the render with the bank-variant's sources placed at the render-variant's poses
    (rz_<render>_src<bank>, pool_render_fixed.py) whenever the two source lists differ (45 of 600 items); identical source
    lists make the two renders identical, so the stock one is used."""
    if bank_v == render_v or render_sources(item, bank_v) == render_sources(item, render_v):
        return f'rz_{render_v}_bf16', f'meta_{render_v}'
    return f'rz_{render_v}_src{bank_v}_bf16', f'meta_{render_v}_src{bank_v}'


def build_onpolicy_row(item, stream, variant, *, n_chunks=N_CHUNKS):
    """Build one or two recorded target chunks; historical callers retain two.

    Single-chunk recovery can use the final available continuation without
    requiring an unused second chunk. Context and its cached render are unchanged.
    """
    if isinstance(n_chunks, bool) or n_chunks not in (1, 2):
        raise ValueError('on-policy target requires one or two chunks')
    bank_v, pose_v, render_v = split_variant(variant)
    s = int(item['s']); bank = [int(b) for b in item[f'bank_{bank_v}']]; recent = list(range(max(1, s - 4), s))
    chunks = list(range(s, s + 4 * n_chunks))
    if chunks[-1] >= stream.L:
        raise ValueError('target outside the stream')
    import data as _data
    per_chunk = bool(getattr(_data, 'PER_CHUNK_RENDER', False))                        # train_r2op.py: chunk 1's own render group
    true_positions, roles, positions, chunk_index = assemble(stream, bank, recent, chunks, n_chunks, render=True, per_chunk=per_chunk)
    poses = stream.c2w[:4 * chunks[-1] + 1].clone().double()
    labels = torch.from_numpy(np.asarray(item[f'{pose_v}_end'], np.float64))           # (s,4,4) latent-end labels, [0] = anchor
    for j in range(1, s):
        poses[4 * j] = labels[j]
    poses = poses.float(); poses[:, :3, 3] *= stream.cam_scale
    H, W = stream.latents.shape[-2:]
    rays = packed_rays(poses, stream.K, true_positions, height=16 * H, width=16 * W)
    c2w = torch.stack([poses[stream.frame_index(p)] for p in true_positions])
    gen = torch.from_numpy(unpack_bf16(item['gen_latents_bf16']))                       # (1,48,s,h,w), latent 0 = recorded anchor
    rz_key, meta_key = render_key(item, bank_v, render_v)
    if rz_key not in item:
        raise KeyError(f'{rz_key} missing: run pool_render_fixed.py (sources fixed, placement varied)')
    rz = torch.from_numpy(unpack_bf16(item[rz_key])); meta = torch.from_numpy(np.asarray(item[meta_key], np.float32))
    if per_chunk and n_chunks > 1:
        if render_v != 'meas':
            raise ValueError('per-chunk on-policy rows exist for the measured render only (precompute_pool_next.py)')
        if 'rz_next_meas_bf16' not in item:
            raise KeyError('rz_next_meas_bf16 missing: run precompute_pool_next.py')
        rz = torch.cat([rz, torch.from_numpy(unpack_bf16(item['rz_next_meas_bf16']))], 2); meta = torch.cat([meta, torch.from_numpy(np.asarray(item['meta_next_meas'], np.float32))], 2)
    context = torch.cat([gen[:, :, [0, *bank, *recent]], rz], 2)
    metadata = torch.zeros((1, meta.shape[1], len(true_positions), H, W)); nctx = 1 + len(bank) + len(recent); metadata[:, :, nctx:nctx + meta.shape[2]] = meta
    return dict(context=context, target=stream.latents[:, :, chunks], rays=rays, roles=roles, positions=positions, chunk_index=chunk_index, c2w=c2w,
                metadata=metadata, K=stream.K, text=stream.text, n_chunks=n_chunks, start=s, bank=bank, recent=recent, render=True,
                stream_id='OP' + (variant[0] if variant in ('cmd', 'meas') else variant) + stream.row['stream_id'])
