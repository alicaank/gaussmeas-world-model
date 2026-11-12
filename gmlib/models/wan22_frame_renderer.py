"""Wan2.2-TI2V-5B single-target frame renderer.

Adapted from FrameCrafter fb5b.model_fn_fb5b with the GS-evidence branch
removed. Sequence layout: [clean context latents ..., noisy target latent],
all 48-ch through the STOCK patch embedding (in_dim stays 48 — no surgery).
Conditioning (context latents + target-relative Plücker rays + mask/role)
enters via the zero-init residual RayContextProjection after patchify, so at
init the model is exactly the pretrained Wan2.2 backbone.

Zero-temporal RoPE: frames are an unordered set; only (h, w) carry position.
Ordered recent context is doc Phase 5 and will split the sequence into an
ordered-temporal + unordered-spatial stream — M1 deliberately stays unordered
(the Phase-3 gate demands context-permutation invariance).

CameraFiLM (fb5b.py) is the fallback lever if the pose-adherence gate fails;
model_fn keeps its hook but M1 training starts without it.
"""
from __future__ import annotations

import glob
import math

import torch
import torch.nn as nn

from gmlib import _paths  # noqa: F401  (sys.path for diffsynth)
from gmlib.diffsynth.core.loader.model import load_model
from gmlib.diffsynth.models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from gmlib.diffsynth.models.wan_video_vae import WanVideoVAE38
from gmlib.diffsynth.utils.state_dict_converters.wan_video_vae import WanVideoVAEStateDictConverter

from gmlib.models.ray_encoder import LATENT_CHANNELS, LAYOUT, RayContextProjection
from gmlib.models.correspondence_encoder import CorrespondenceAdapter
from gmlib.models.framepack import FramePackConditioner, FramePackPlan


def temporal_positions(frame_indices, ordered=True):
    """Map real scene frame indices -> integer temporal RoPE positions for model_fn.

    The order is [context..., target] (target LAST), positions made relative to the
    minimum index so the earliest frame is 0 and the SPACING between frames (how far
    apart in the trajectory) is preserved -- e.g. ctx [0,3,6] + target 18 -> [0,3,6,18].
    That spacing is the temporal signal the zero-temporal model threw away. Returns a
    long tensor of length len(frame_indices). If `ordered` is False, returns all -1
    (identity) -- the zero-temporal / unordered-set special case.
    """
    idx = torch.as_tensor(list(frame_indices), dtype=torch.long)
    if idx.numel() == 0:
        raise ValueError("temporal_positions requires at least one frame")
    if not ordered:
        return torch.full_like(idx, -1)
    return idx - int(idx.min())


def unordered_spatial_freqs(dit, frames: int, height: int, width: int,
                            *, device=None) -> torch.Tensor:
    """Wan 3D RoPE with identity time and native 2D source-patch addresses."""
    if frames < 1 or height < 1 or width < 1:
        raise ValueError("source RoPE grid dimensions must be positive")
    temporal = torch.ones_like(dit.freqs[0][0]).view(1, 1, 1, -1).expand(
        frames, height, width, -1
    )
    freqs = torch.cat([
        temporal,
        dit.freqs[1][:height].view(1, height, 1, -1).expand(
            frames, height, width, -1
        ),
        dit.freqs[2][:width].view(1, 1, width, -1).expand(
            frames, height, width, -1
        ),
    ], dim=-1).reshape(frames * height * width, 1, -1)
    return freqs.to(device if device is not None else dit.freqs[0].device)


DIT_KWARGS = {
    "has_image_input": False, "patch_size": [1, 2, 2], "in_dim": 48, "dim": 3072,
    "ffn_dim": 14336, "freq_dim": 256, "text_dim": 4096, "out_dim": 48,
    "num_heads": 24, "num_layers": 30, "eps": 1e-06,
}

# Tiny config for CPU structural tests: head_dim 24 splits 8/8/8 for 3D RoPE.
DIT_KWARGS_TINY = {
    "has_image_input": False, "patch_size": [1, 2, 2], "in_dim": 48, "dim": 96,
    "ffn_dim": 192, "freq_dim": 64, "text_dim": 32, "out_dim": 48,
    "num_heads": 4, "num_layers": 2, "eps": 1e-06,
}


def load_wan22_dit(device="cuda", dtype=torch.bfloat16) -> WanModel:
    shards = sorted(glob.glob(str(_paths.WAN22_DIR / "diffusion_pytorch_model-*.safetensors")))
    assert shards, f"no DiT shards under {_paths.WAN22_DIR}"
    return load_model(WanModel, shards, config=DIT_KWARGS, torch_dtype=dtype, device=device)


def load_wan22_vae(device="cuda", dtype=torch.bfloat16) -> WanVideoVAE38:
    return load_model(
        WanVideoVAE38, str(_paths.WAN22_DIR / "Wan2.2_VAE.pth"),
        torch_dtype=dtype, device=device,
        state_dict_converter=WanVideoVAEStateDictConverter,
    )


@torch.no_grad()
def decode_target(vae: WanVideoVAE38, latents: torch.Tensor, device="cuda") -> torch.Tensor:
    """Decode ONLY the last (target) latent frame -> (1, 3, H, W) in [-1,1]."""
    z_t = latents[:, :, -1:].to(device, next(vae.parameters()).dtype)
    frame = vae.decode(z_t, device=device)
    frame = frame[0] if isinstance(frame, list) else frame
    if frame.dim() == 5:
        frame = frame[:, :, 0]
    elif frame.dim() == 4 and frame.shape[0] == 1 and frame.shape[1] != 3:
        frame = frame[:, 0]
    return frame


def model_fn(
    dit: WanModel,
    latents: torch.Tensor,                 # (B, 48, F, H_lat, W_lat): ctx clean + noisy target last
    timestep: torch.Tensor,
    context: torch.Tensor,                 # text embedding input (B, L_text, text_dim)
    ray_cond: torch.Tensor | None = None,  # (B, LAYOUT.total, F, H_lat, W_lat)
    ray_adapter: RayContextProjection | None = None,
    film_adapter: nn.Module | None = None,     # fb5b.CameraFiLM (fallback lever)
    film_rays: torch.Tensor | None = None,     # (B, C_ray, F, H_lat, W_lat)
    mem_evidence: torch.Tensor | None = None,  # (B, EVIDENCE_LAYOUT.total, T, H_lat, W_lat)
    mem_adapter: nn.Module | None = None,      # MemoryEvidenceAdapter
    corr_evidence: torch.Tensor | None = None,  # (B, CORR_LAYOUT.total, T, H_lat, W_lat)
    corr_adapter: CorrespondenceAdapter | None = None,
    warp_evidence_scale: float = 1.0,
    cam_qk_adapter: "CameraQKAdapter | None" = None,   # UCPE-style attention-level camera
    trajectory_cam_adapter: "TrajectoryCameraQKAdapter | None" = None,
    trajectory_cam_features: torch.Tensor | None = None,  # (B,F,27) analytic motion/camera
    ray_local_ucpe: "RayLocalUCPEStack | None" = None,
    ray_local_transforms: torch.Tensor | None = None,  # (B,S,4,4), world-to-ray per token
    static_source_attention: "StaticSourceAttentionStack | None" = None,
    static_source_latents: torch.Tensor | None = None,  # (B,48,K,H,W), immutable observed
    static_source_caches: dict | None = None,
    static_source_geometry_bias: torch.Tensor | dict[int, torch.Tensor] | None = None,
    static_source_authority: torch.Tensor | None = None,
    static_source_provenance: torch.Tensor | None = None,
    static_source_scale: float = 1.0,
    tied_source_reader=None,  # TiedSourceReader: V3.1 host-weight joint read
    lattice_patch: nn.Module | None = None,
    lattice_latents: torch.Tensor | None = None,  # (B,48,F,H_lat,W_lat)
    lattice_scale: float = 1.0,
    ray_qk_only: bool = False,
    target_frames: int = 1,
    cam_attn: "CameraAttentionStack | None" = None,    # camera attention with its OWN Q/K/V
    cam_attn_rays: torch.Tensor | None = None,         # (B, C_ray, F, H_lat, W_lat)
    frame_temporal_pos: torch.Tensor | None = None,  # (F,) int temporal positions; <0 = identity
    framepack_plan: FramePackPlan | None = None,
    framepack_conditioner: FramePackConditioner | None = None,
    attn_logit_bias: torch.Tensor | dict[int, torch.Tensor] | None = None,
    sparse_qk_bias: dict[int, torch.Tensor] | None = None,
    sparse_residual: dict[int, torch.Tensor] | None = None,
    memory_control: nn.Module | None = None,
    memory_control_scaffold: torch.Tensor | None = None,
    # A tensor applies the prior to every self-attention layer. A mapping applies it
    # only at selected blocks, which is required for CAMEO-like sparse supervision and
    # zero-init geometry gates without perturbing the whole pretrained backbone.
    use_gradient_checkpointing: bool = False,
) -> torch.Tensor:
    assert latents.shape[1] == LATENT_CHANNELS, latents.shape
    assert dit.in_dim == LATENT_CHANNELS and dit.out_dim == LATENT_CHANNELS
    if not math.isfinite(warp_evidence_scale) or warp_evidence_scale < 0:
        raise ValueError("warp_evidence_scale must be finite and nonnegative")
    if not math.isfinite(lattice_scale) or lattice_scale < 0:
        raise ValueError("lattice_scale must be finite and nonnegative")
    if (lattice_patch is None) != (lattice_latents is None):
        raise ValueError("lattice patch projection and latents must be supplied together")
    if (framepack_plan is None) != (framepack_conditioner is None):
        raise ValueError("FramePack plan and conditioner must be supplied together")
    if (memory_control is None) != (memory_control_scaffold is None):
        raise ValueError("memory control and its scaffold must be supplied together")
    if framepack_plan is not None and cam_attn is not None:
        raise ValueError("FramePack currently requires CameraQK/FiLM, not dense cam_attn")
    for name, route in (("sparse_qk_bias", sparse_qk_bias),
                        ("sparse_residual", sparse_residual)):
        if route is not None:
            invalid = sorted(set(map(int, route)) - set(range(len(dit.blocks))))
            if invalid:
                raise ValueError(f"{name} blocks outside backbone: {invalid}")

    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).to(latents.dtype))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    x = dit.patch_embedding(latents)
    if framepack_conditioner is not None:
        context_frames = latents.shape[2] - target_frames
        if context_frames < 1:
            raise ValueError("FramePack teacher requires at least one history frame")
        history_hidden = framepack_conditioner.history_patch_embedding(
            latents[:, :, :context_frames]
        )
        if history_hidden.shape[2:] != x[:, :, :context_frames].shape[2:]:
            raise ValueError("history patch embedding changed the Wan token grid")
        x = torch.cat([history_hidden, x[:, :, context_frames:]], dim=2)
    # Wonder-aligned synthetic coordinate-field input. Scale zero deliberately skips
    # the convolution and addition, making L0 bitwise identical to branch absence.
    # At scale one, the zero-initialized projection is output-identical at step zero
    # while retaining a live gradient into its weights.
    if lattice_patch is not None and lattice_scale != 0.0:
        lattice_hidden = lattice_patch(
            lattice_latents.to(device=x.device, dtype=latents.dtype)
        )
        if lattice_hidden.shape != x.shape:
            raise ValueError(
                f"lattice patch grid {tuple(lattice_hidden.shape)} does not match "
                f"video grid {tuple(x.shape)}"
            )
        x = x + lattice_scale * lattice_hidden
    f, h, w = x.shape[2:]
    x = x.permute(0, 2, 3, 4, 1).reshape(x.shape[0], f * h * w, dit.dim).contiguous()
    if not 1 <= target_frames <= f:
        raise ValueError(f"target_frames must lie in [1,{f}], got {target_frames}")
    source_tokens = None
    source_freqs = None
    tied_src_state = None
    if tied_source_reader is not None:
        # Corrected V3.1 reader: host-weight joint read. Mutually exclusive with
        # the legacy separate-basis reader; sources advance through the host
        # blocks under the CURRENT timestep (not cacheable across steps).
        if static_source_attention is not None:
            raise ValueError("choose one source reader: tied or legacy")
        if static_source_caches is not None:
            raise ValueError("the tied reader's source stream is timestep-"
                             "matched and cannot consume cross-step caches")
        if static_source_latents is None:
            raise ValueError("tied source reading requires source latents")
        if (static_source_latents.ndim != 5
                or static_source_latents.shape[0] != x.shape[0]
                or static_source_latents.shape[1] != LATENT_CHANNELS
                or static_source_latents.shape[-2:] != latents.shape[-2:]):
            raise ValueError("static source latents must be Bx48xKxHxW on the "
                             "model grid")
        tied_hidden = dit.patch_embedding(
            static_source_latents.to(device=x.device, dtype=latents.dtype))
        tied_src_state = tied_hidden.permute(0, 2, 3, 4, 1).reshape(
            tied_hidden.shape[0], -1, dit.dim).contiguous()
        tied_src_freqs = unordered_spatial_freqs(
            dit, tied_hidden.shape[2], tied_hidden.shape[3],
            tied_hidden.shape[4], device=x.device)
    if static_source_attention is None:
        if tied_source_reader is None and (
                static_source_latents is not None
                or static_source_caches is not None):
            raise ValueError("static source inputs require a source-attention stack")
    else:
        if (static_source_latents is None) == (static_source_caches is None):
            raise ValueError("source attention requires exactly one of latents or caches")
        if static_source_latents is not None:
            if (static_source_latents.ndim != 5 or static_source_latents.shape[0] != x.shape[0]
                    or static_source_latents.shape[1] != LATENT_CHANNELS
                    or static_source_latents.shape[-2:] != latents.shape[-2:]):
                raise ValueError("static source latents must be Bx48xKxHxW on the model grid")
            source_hidden = dit.patch_embedding(
                static_source_latents.to(device=x.device, dtype=latents.dtype)
            )
            source_tokens = source_hidden.permute(0, 2, 3, 4, 1).reshape(
                source_hidden.shape[0], -1, dit.dim
            ).contiguous()
            source_freqs = unordered_spatial_freqs(
                dit, source_hidden.shape[2], source_hidden.shape[3],
                source_hidden.shape[4], device=x.device,
            )
        else:
            missing = set(static_source_attention.blocks) - set(static_source_caches)
            if missing:
                raise ValueError(f"source K/V cache missing blocks {sorted(missing)}")

    # Camera conditioning enters the residual stream by default, which means it reaches
    # V and can therefore influence *what* is painted, not only *where* attention looks.
    # Lyra 2.0 instead adds its camera and buffer embeddings as a Q/K bias only
    # (`kq_bias = cam_emb + buf_emb`, V untouched), the same discipline this file already
    # applies to the correspondence branch. `ray_qk_only` selects that routing; the
    # tokens are held here and folded into qk_bias once that channel exists below.
    ray_tokens = None
    if ray_cond is not None and ray_adapter is not None:
        cond_tokens = ray_adapter(ray_cond.to(dtype=x.dtype, device=x.device))
        assert cond_tokens.shape[1] == f * h * w, (cond_tokens.shape, (f, h, w))
        if ray_qk_only:
            ray_tokens = cond_tokens
        else:
            x = x + cond_tokens

    # Spatial-memory evidence (growworld representation). Rendered INTO the
    # target camera, so it lands on the trailing target tokens only; token order
    # is frame-major (f, h, w) and targets are last.
    if mem_evidence is not None and mem_adapter is not None:
        mem_tokens = mem_adapter(mem_evidence.to(dtype=x.dtype, device=x.device))
        n_mem = mem_tokens.shape[1]
        assert n_mem <= f * h * w and n_mem % (h * w) == 0, (
            f"evidence token count {n_mem} incompatible with grid f={f}, h={h}, w={w}"
        )
        x = torch.cat([
            x[:, : f * h * w - n_mem],
            x[:, f * h * w - n_mem:] + warp_evidence_scale * mem_tokens,
        ], dim=1)

    # Correspondence evidence (Proposal 1): canonical coords -> a bias added
    # ONLY before the Q/K projections inside each block's self-attention, never
    # to `x` itself and never to V. Same trailing-target-token placement as
    # mem_evidence above, but content generation stays untouched by this branch.
    qk_bias = None
    if corr_evidence is not None and corr_adapter is not None:
        corr_tokens = corr_adapter(corr_evidence.to(dtype=x.dtype, device=x.device))
        n_corr = corr_tokens.shape[1]
        assert n_corr <= f * h * w and n_corr % (h * w) == 0, (
            f"correspondence token count {n_corr} incompatible with grid f={f}, h={h}, w={w}"
        )
        qk_bias = torch.zeros(x.shape[0], f * h * w, dit.dim, dtype=x.dtype, device=x.device)
        qk_bias[:, f * h * w - n_corr:] = warp_evidence_scale * corr_tokens

    if ray_tokens is not None:
        qk_bias = ray_tokens if qk_bias is None else qk_bias + ray_tokens

    # UCPE-style ATTENTION-LEVEL camera conditioning: the same ray tensor the
    # input-level path consumes, emitted as a per-token q/k bias over ALL frames so
    # attention can compare camera geometry between any two tokens. Composes
    # additively with the correspondence bias above (they occupy the same channel;
    # correspondence covers only the trailing target tokens).
    if ray_cond is not None and cam_qk_adapter is not None:
        cam_src = ray_cond
        if getattr(cam_qk_adapter, "rays_only", False):
            # bare Plücker planes: [latent 48 | mask 1 | RAYS 1536 | role 1]
            o = LAYOUT.context_latent + LAYOUT.condition_mask
            cam_src = ray_cond[:, o:o + LAYOUT.ray_features]
        cam_tokens = cam_qk_adapter(cam_src.to(dtype=x.dtype, device=x.device))
        assert cam_tokens.shape[1] == f * h * w, (cam_tokens.shape, (f, h, w))
        qk_bias = cam_tokens if qk_bias is None else qk_bias + cam_tokens

    # Strong global-camera baseline: target-gauge relative pose, intrinsics, temporal
    # spacing, velocity and acceleration. It complements dense Pluecker rays and enters
    # Q/K only. The final MLP is zero-init, so adding this branch preserves an old
    # checkpoint exactly before training.
    if trajectory_cam_adapter is not None:
        if trajectory_cam_features is None:
            raise ValueError("trajectory camera adapter requires analytic camera features")
        cam_global = trajectory_cam_adapter(
            trajectory_cam_features.to(dtype=x.dtype, device=x.device), h * w
        )
        if cam_global.shape != x.shape:
            raise ValueError(
                f"global camera tokens {tuple(cam_global.shape)} != hidden {tuple(x.shape)}"
            )
        qk_bias = cam_global if qk_bias is None else qk_bias + cam_global

    # Temporal RoPE. Default (frame_temporal_pos=None) is ZERO-temporal: frames are
    # an unordered set, all get identity (ones), only (h, w) carry position -- the
    # FrameCrafter view-set behaviour, and what every existing checkpoint trained on.
    # When frame_temporal_pos is given (one int per frame), ordered frames (pos>=0)
    # get real temporal rotation from dit.freqs[0][pos], so the model can read the
    # trajectory order/spacing; set-context frames (pos<0 sentinel) keep identity.
    # An all-identity / all-negative vector reproduces the zero-temporal case exactly,
    # so this is checkpoint-compatible.
    if framepack_plan is None:
        if frame_temporal_pos is None:
            f_freqs = torch.ones_like(dit.freqs[0][:f])
        else:
            assert frame_temporal_pos.shape[0] == f, (frame_temporal_pos.shape, f)
            ordered_pos = frame_temporal_pos[frame_temporal_pos >= 0]
            if ordered_pos.numel() and int(ordered_pos.max()) >= len(dit.freqs[0]):
                raise ValueError(
                    f"temporal position {int(ordered_pos.max())} exceeds the model RoPE "
                    f"table bound {len(dit.freqs[0]) - 1}; resample/scale the trajectory "
                    "or extend the frequency table explicitly (positions are never clamped)"
                )
            ident = torch.ones_like(dit.freqs[0][0])
            f_freqs = torch.stack([
                dit.freqs[0][int(p)] if int(p) >= 0 else ident
                for p in frame_temporal_pos
            ], dim=0).to(dit.freqs[0].device)
        freqs = torch.cat([
            f_freqs.view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    else:
        if framepack_plan.input_frames != f or framepack_plan.target_frames != target_frames:
            raise ValueError("FramePack plan does not match model input/target frames")
        freqs = framepack_conditioner.packed_freqs(
            dit, framepack_plan, frame_temporal_pos, h, w, device=x.device
        )

        # Every per-token branch is packed by the same plan.  Applying FramePack only to
        # hidden states while leaving camera/correspondence tokens uncompressed would
        # silently associate geometry with the wrong visual token.
        x = framepack_conditioner.pack_tokens(
            x, framepack_plan, h, w, add_roles=True
        )
        if qk_bias is not None:
            qk_bias = framepack_conditioner.pack_tokens(
                qk_bias, framepack_plan, h, w
            )
        def pack_sparse_tokens(tokens: torch.Tensor) -> torch.Tensor:
            full_tokens = framepack_plan.input_frames * h * w
            if tokens.ndim != 3 or tokens.shape[1] > full_tokens \
                    or tokens.shape[1] % (h * w):
                raise ValueError("sparse route does not align to complete trailing frames")
            if tokens.shape[1] < full_tokens:
                padded = tokens.new_zeros(tokens.shape[0], full_tokens, tokens.shape[2])
                padded[:, -tokens.shape[1]:] = tokens
                tokens = padded
            return framepack_conditioner.pack_tokens(tokens, framepack_plan, h, w)
        if sparse_qk_bias is not None:
            sparse_qk_bias = {
                int(block): pack_sparse_tokens(tokens)
                for block, tokens in sparse_qk_bias.items()
            }
        if sparse_residual is not None:
            sparse_residual = {
                int(block): pack_sparse_tokens(tokens)
                for block, tokens in sparse_residual.items()
            }
        if ray_local_transforms is not None:
            ray_local_transforms = framepack_conditioner.pack_matrices(
                ray_local_transforms, framepack_plan, h, w
            )
    # Lazy initialization is retained for historical/non-final readers, but it
    # must use the sequence the joint softmax actually sees.  With FramePack,
    # ``f*h*w`` above is the pre-pack count and can overstate main-key mass by
    # more than 2x.  The all-on final teacher initializes explicitly from its
    # frozen saturated layout before any forward, so this branch is never
    # microstep-order dependent there.
    if tied_source_reader is not None and not tied_source_reader.beta_initialized:
        tied_source_reader.initialize_beta(x.shape[1], tied_src_state.shape[1])
    def prepare_attention_bias(value: torch.Tensor) -> torch.Tensor:
        sequence = x.shape[1]
        if value.shape[-2:] != (sequence, sequence):
            raise ValueError(
                f"pairwise attention bias {tuple(value.shape)} incompatible "
                f"with {sequence} tokens"
            )
        return value.to(device=x.device, dtype=x.dtype)

    if isinstance(attn_logit_bias, dict):
        invalid = sorted(set(attn_logit_bias) - set(range(len(dit.blocks))))
        if invalid:
            raise ValueError(f"attention-bias blocks outside backbone: {invalid}")
        attn_logit_bias = {
            int(block): prepare_attention_bias(value)
            for block, value in attn_logit_bias.items()
        }
    elif attn_logit_bias is not None:
        attn_logit_bias = prepare_attention_bias(attn_logit_bias)

    film = None
    if film_adapter is not None and film_rays is not None:
        film = film_adapter.compute(film_rays.to(dtype=x.dtype, device=x.device))
        if framepack_plan is not None:
            film = {
                block: (
                    framepack_conditioner.pack_tokens(gamma, framepack_plan, h, w),
                    framepack_conditioner.pack_tokens(beta, framepack_plan, h, w),
                )
                for block, (gamma, beta) in film.items()
            }

    memory_hints = None
    if memory_control is not None:
        target_tokens = target_frames * h * w
        memory_hints = memory_control(
            x[:, -target_tokens:],
            memory_control_scaffold.to(device=x.device, dtype=x.dtype),
            context, t_mod, freqs[-target_tokens:], timestep)
        invalid = sorted(set(map(int, memory_hints)) - set(range(len(dit.blocks))))
        if invalid:
            raise ValueError(f"memory control blocks outside backbone: {invalid}")

    for i, block in enumerate(dit.blocks):
        block_attn_bias = (attn_logit_bias.get(i) if isinstance(attn_logit_bias, dict)
                           else attn_logit_bias)
        if film is not None and i in film:
            gamma, beta = film[i]
            x = x * (1 + gamma) + beta
        if sparse_residual is not None and i in sparse_residual:
            residual = sparse_residual[i]
            if residual.shape[0] != x.shape[0] or residual.shape[-1] != x.shape[-1] \
                    or residual.shape[1] > x.shape[1]:
                raise ValueError(f"sparse residual shape mismatch at block {i}")
            x = torch.cat([
                x[:, :-residual.shape[1]], x[:, -residual.shape[1]:] + residual,
            ], dim=1) if residual.shape[1] < x.shape[1] else x + residual
        if memory_hints is not None and i in memory_hints:
            hint = memory_hints[i]
            if hint.shape != x[:, -hint.shape[1]:].shape:
                raise ValueError(f"memory control hint shape mismatch at block {i}")
            x = torch.cat([
                x[:, :-hint.shape[1]], x[:, -hint.shape[1]:] + hint,
            ], dim=1) if hint.shape[1] < x.shape[1] else x + hint
        block_qk_bias = qk_bias
        if sparse_qk_bias is not None and i in sparse_qk_bias:
            sparse = sparse_qk_bias[i]
            if sparse.shape[0] != x.shape[0] or sparse.shape[-1] != x.shape[-1] \
                    or sparse.shape[1] > x.shape[1]:
                raise ValueError(f"sparse Q/K shape mismatch at block {i}")
            if sparse.shape[1] < x.shape[1]:
                padded = torch.zeros_like(x)
                padded[:, -sparse.shape[1]:] = sparse
                sparse = padded
            block_qk_bias = sparse if block_qk_bias is None else block_qk_bias + sparse
        # Camera attention with its own Q/K/V basis, at a few depths only. Applied
        # BEFORE the block (like FiLM) so the block sees the camera-routed state; the
        # residual is zero at init, so this is exact backbone equivalence to start.
        if cam_attn is not None and cam_attn_rays is not None:
            site = cam_attn.site(i)
            if site is not None:
                x = x + site(x, cam_attn_rays)
        if ray_local_ucpe is not None:
            if ray_local_transforms is None:
                raise ValueError("ray-local UCPE requires per-token homogeneous transforms")
            site = ray_local_ucpe.site(i)
            if site is not None:
                x = x + site(x, ray_local_transforms)
        if static_source_attention is not None:
            site = static_source_attention.site(i)
            if site is not None:
                target_tokens = target_frames * h * w
                target_hidden = x[:, -target_tokens:]
                cache = (None if static_source_caches is None
                         else static_source_caches[i])
                geometry_bias = (
                    static_source_geometry_bias.get(i)
                    if isinstance(static_source_geometry_bias, dict)
                    else static_source_geometry_bias
                )
                source_residual = site(
                    target_hidden,
                    source_tokens=source_tokens if cache is None else None,
                    cache=cache,
                    geometry_bias=geometry_bias,
                    key_authority=static_source_authority,
                    source_provenance=(static_source_provenance if cache is None else None),
                    query_freqs=(freqs[:, -target_tokens:]
                                 if freqs.ndim == 4 else freqs[-target_tokens:]),
                    source_freqs=source_freqs if cache is None else None,
                    read_scale=static_source_scale,
                )
                x = torch.cat([x[:, :-target_tokens], target_hidden + source_residual], dim=1)
        if tied_source_reader is not None and i <= max(tied_source_reader.blocks):
            if tied_source_reader.site(i):
                if getattr(tied_source_reader, "omega_delta", False):
                    from gmlib.models.gate_omega_reader import (
                        omega_tied_block_forward as tied_block_forward,
                    )
                else:
                    from gmlib.models.tied_source_reader import (
                        tied_block_forward,
                    )
                tied_geometry = (
                    static_source_geometry_bias.get(i)
                    if isinstance(static_source_geometry_bias, dict)
                    else static_source_geometry_bias
                )
                tied_args = dict(
                    n_target=target_frames * h * w,
                    geometry_bias=tied_geometry,
                    key_authority=static_source_authority,
                    qk_bias=block_qk_bias, attn_logit_bias=block_attn_bias,
                )
                if use_gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        # Bind loop-local values now.  Checkpoint recomputes this
                        # closure during backward, after the block loop has
                        # advanced; an unbound ``i``/``block`` would make every
                        # sparse reader site recompute as the final backbone block.
                        lambda *tensors, _block=block, _index=i,
                               _tied_args=tied_args: tied_block_forward(
                            _block, tied_source_reader, _index, tensors[0],
                            tensors[1], context, t_mod, freqs,
                            tied_src_freqs, **_tied_args),
                        x, tied_src_state, use_reentrant=False,
                    )
                else:
                    x = tied_block_forward(
                        block, tied_source_reader, i, x, tied_src_state,
                        context, t_mod, freqs, tied_src_freqs, **tied_args)
            if i < max(tied_source_reader.blocks):
                # source stream advances through the host block natively so the
                # next site reads depth-matched representations
                tied_src_state = block(tied_src_state, context, t_mod,
                                       tied_src_freqs)
            if tied_source_reader.site(i):
                continue
        if use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                block, x, context, t_mod, freqs, block_qk_bias, block_attn_bias,
                use_reentrant=False
            )
        else:
            x = block(x, context, t_mod, freqs, qk_bias=block_qk_bias,
                      attn_logit_bias=block_attn_bias)

    x = dit.head(x, t)
    output_frames = target_frames if framepack_plan is not None else f
    if framepack_plan is not None:
        target_tokens = target_frames * h * w
        if x.shape[1] < target_tokens:
            raise RuntimeError("FramePack lost supervised target tokens")
        x = x[:, -target_tokens:]
    x = dit.unpatchify(x, (output_frames, h, w))
    assert x.shape[1] == LATENT_CHANNELS
    return x
