"""Three memory-use modes and spatial, current-command-only routing."""
from dataclasses import replace
from enum import IntEnum
import torch
from torch import nn
from torch.nn import functional as F
from foundation_model import Foundation


class Mode(IntEnum):
    NO_ARCHIVE = 0
    APPEARANCE = 1
    GEOMETRY = 2


def move_condition(c, device):
    return replace(c, **{k: getattr(c, k).to(device) for k in
        ('appearance', 'rays', 'metadata', 'roles', 'positions')},
        camera_mask=None if c.camera_mask is None else c.camera_mask.to(device),
        context_u=None if c.context_u is None else c.context_u.to(device),
        correspondence=None if c.correspondence is None else c.correspondence.to(device))


def select_slots(c, keep):
    if keep.shape != c.roles.shape or not keep[-4:].all():
        raise ValueError('all target slots must be retained')
    edges=c.correspondence
    if edges is not None:
        patches=(c.appearance.shape[-2]//2)*(c.appearance.shape[-1]//2)
        old_slots=edges[:,1].long()//patches
        survive=keep[old_slots]
        edges=edges[survive].clone()
        mapping=keep.long().cumsum(0)-1
        edges[:,1]=mapping[old_slots[survive]]*patches+edges[:,1].long()%patches
    return replace(c, appearance=c.appearance[:, :, keep[:-4]], rays=c.rays[:, :, keep],
        metadata=c.metadata[:, :, keep], roles=c.roles[keep], positions=c.positions[keep],
        camera_mask=None if c.camera_mask is None else c.camera_mask[keep],
        context_u=None if c.context_u is None else c.context_u[keep[:-4]],correspondence=edges)


def without_render(c):
    return select_slots(c, c.roles != 3)


def without_archive(c):
    """Evaluation control: remove native archive while preserving render exactly."""
    return select_slots(c, c.roles != 2)


def apply_mode(c, mode):
    mode = Mode(mode)
    if mode == Mode.NO_ARCHIVE:
        c = select_slots(c, (c.roles != 2) & (c.roles != 3))
    elif mode == Mode.APPEARANCE:
        c = without_render(c)
    mask = torch.ones_like(c.roles, dtype=torch.bool)
    if c.camera_mask is not None:
        mask &= c.camera_mask
    if mode == Mode.APPEARANCE:
        mask &= c.roles != 2
    return replace(c, camera_mask=mask,correspondence=c.correspondence if mode==Mode.GEOMETRY else None)


class SpatialRouter(nn.Module):
    def __init__(self, dim=128, grid=4):
        super().__init__()
        if dim % 4 or grid < 1:
            raise ValueError('router needs four attention heads and a positive grid')
        self.grid = grid
        self.source = nn.Linear(48+24+2, dim)
        self.query = nn.Linear(24, dim)
        self.spatial = nn.Linear(2, dim)
        self.role = nn.Embedding(4, dim)
        self.attention = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 3)
        for name, p in self.named_parameters():
            if p.ndim > 1:
                nn.init.trunc_normal_(p, std=dim**-.5, a=-3*dim**-.5, b=3*dim**-.5)
            elif 'norm.weight' not in name:
                nn.init.zeros_(p)

    def pool(self, x):
        b, channels, frames, h, w = x.shape
        y = F.adaptive_avg_pool2d(x.permute(0,2,1,3,4).reshape(b*frames,channels,h,w),self.grid)
        return y.reshape(b,frames,channels,self.grid**2).permute(0,1,3,2)

    def forward(self, c):
        # Targets are intentionally absent from this API. Only the current four
        # camera queries, past appearance and already-built render can enter.
        source = torch.cat([self.pool(c.appearance), self.pool(c.rays[:,:,:-4]),
                            self.pool(c.metadata[:,:,:-4])], -1).float()
        xy = torch.linspace(-1,1,self.grid,device=source.device)
        y,x = torch.meshgrid(xy,xy,indexing='ij')
        pos = self.spatial(torch.stack([x,y],-1).reshape(self.grid**2,2))
        keys = self.source(source) + pos[None,None] + self.role(c.roles[:-4])[None,:,None]
        query = self.query(self.pool(c.rays[:,:,-4:]).float()) + pos[None,None]
        query = query.flatten(1,2); keys = keys.flatten(1,2)
        attended,_ = self.attention(query,keys,keys,need_weights=False)
        return self.head(self.norm(query+attended).mean(1))


class JointMemoryCamera(Foundation):
    architecture = 'joint_memory_camera.v9'
    def __init__(self, dit, config=None):
        super().__init__(dit)
        cfg = dict(config or {})
        self.block_camera_qk=bool(cfg.get('block_camera_qk',False))
        self.native_patch_address=bool(cfg.get('native_patch_address',False))
        # Small, nonzero new paths; pretrained block weights are retained.
        if self.block_camera_qk:
            self.camera_qk_gains=nn.Parameter(torch.full((len(dit.blocks),),1e-3))
        if self.native_patch_address:
            self.archive_address_gains=nn.Parameter(torch.full((len(dit.blocks),),1e-3))
        self.router = SpatialRouter(cfg.get('router_dim',128),cfg.get('router_grid',4))
        self.stage = cfg.get('stage','teacher')
        self.set_stage(self.stage)

    def set_stage(self, stage):
        if stage not in ('teacher','causal_warmup','dmd'):
            raise ValueError('unknown training stage')
        self.stage = stage

    def configure_trainables(self):
        super().configure_trainables()
        self.router.float()
        return [(n,p) for n,p in self.named_parameters() if p.requires_grad]

    def forward(self, *args, **kwargs):
        default = self.stage != 'teacher'
        requested = kwargs.get('causal_context', default)
        if self.training and requested != default:
            raise ValueError('attention-mask override is evaluation-only; use the declared training stage')
        kwargs['causal_context'] = requested
        return super().forward(*args, **kwargs)

    def route(self, condition, forced_mode=None):
        logits = self.router(condition)
        if logits.shape[0] != 1:
            raise ValueError('one scene per microbatch is required')
        mode = int(logits.detach()[0].argmax()) if forced_mode is None else int(Mode(forced_mode))
        return apply_mode(condition,mode), logits, mode
