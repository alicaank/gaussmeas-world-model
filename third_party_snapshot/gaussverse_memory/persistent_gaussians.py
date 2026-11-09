"""Opt-in reuse of source-indexed ReSplat Gaussians in a fixed scene gauge.

Unchanged source blocks retain their first accepted parameters. New/revised
sources use the current reconstruction. This enforces primitive persistence;
it does not reconcile overlapping surfaces from different source images.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json

import torch


def _tensor_digest(value):
    value = value.detach().cpu().contiguous()
    header = f'{value.dtype}:{tuple(value.shape)}:'.encode()
    return hashlib.sha256(header + value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _fingerprint(source, intrinsics, size):
    material = {k: _tensor_digest(source[k]) for k in ('rgb', 'c2w', 'depth', 'confidence')}
    material.update(intrinsics=_tensor_digest(intrinsics), size=list(size))
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


class GaussianBlockCache:
    """Transactional CPU cache. Evicted render sources remain available on revisit."""

    def __init__(self, *, max_sources=128):
        self.max_sources = int(max_sources)
        if self.max_sources < 2:
            raise ValueError('max_sources must accommodate the observed pair')
        self.namespace = None
        self.entries = {}
        self.events = []

    def begin(self, *, scene, gauge, model):
        namespace = (str(scene), str(gauge), str(model))
        if namespace != self.namespace:
            self.namespace = namespace
            self.entries = {}
            self.events = []

    def prepare(self, gaussians, sources, intrinsics, size):
        if self.namespace is None:
            raise RuntimeError('Begin a scene/gauge/model namespace before rendering')
        ids = [int(s['frame_id']) for s in sources]
        if len(ids) < 2 or len(set(ids)) != len(ids):
            raise ValueError('Expected at least two uniquely identified source views')
        if len(set(self.entries) | set(ids)) > self.max_sources:
            raise RuntimeError('Persistent source budget reached; start a new session or increase its explicit budget')
        total = gaussians.means.shape[1]
        if gaussians.means.shape[0] != 1 or total % len(ids):
            raise ValueError('Expected one source-major Gaussian batch')
        fields = {}
        for field in dataclasses.fields(gaussians):
            value = getattr(gaussians, field.name)
            if value is None:
                continue
            if not isinstance(value, torch.Tensor) or value.ndim < 2 or value.shape[:2] != (1, total):
                raise ValueError(f'Unsupported non-primitive Gaussian metadata: {field.name}')
            fields[field.name] = value
        required = {'means', 'covariances', 'harmonics', 'opacities', 'scales', 'rotations', 'rotations_unnorm'}
        if not required.issubset(fields):
            raise ValueError('Incomplete Gaussian parameterization')
        width = total // len(ids)
        staged = dict(self.entries)
        blocks = []
        event = dict(source_ids=ids, reused=[], inserted=[], revised=[], points_per_source=width)
        for index, source in enumerate(sources):
            frame_id = ids[index]
            fingerprint = _fingerprint(source, intrinsics, size)
            previous = self.entries.get(frame_id)
            if previous is not None and previous['fingerprint'] == fingerprint:
                block = previous['block']
                if set(block) != set(fields) or any(block[k].shape != v[:,index*width:(index+1)*width].shape
                                                   or block[k].dtype != v.dtype for k,v in fields.items()):
                    raise ValueError('Cached Gaussian layout changed within a model namespace')
                event['reused'].append(frame_id)
            else:
                block = {k:v[:,index*width:(index+1)*width].detach().cpu().clone() for k,v in fields.items()}
                staged[frame_id] = dict(fingerprint=fingerprint, block=block)
                event['revised' if previous is not None else 'inserted'].append(frame_id)
            blocks.append(block)
        merged = dataclasses.replace(gaussians, **{k:torch.cat([b[k].to(device=v.device) for b in blocks],dim=1)
                                                   for k,v in fields.items()})
        event.update(cached_sources=len(staged), cached_bytes=sum(t.numel()*t.element_size()
            for entry in staged.values() for t in entry['block'].values()))
        return merged, staged, event

    def commit(self, staged, event):
        self.entries = staged
        self.events.append(event)


class PersistentReSplatGeometry:
    """Wrap a frozen ReSplatCondition without modifying its reconstruction code."""

    def __init__(self, base, *, model_id, max_sources=128):
        if not base.depth_override or base.num_refine <= 0:
            raise ValueError('This adapter requires identified RGB-D views and refined ReSplat output')
        self.base = base
        self.model_id = str(model_id)
        self.cache = GaussianBlockCache(max_sources=max_sources)
        self._busy = False

    def __getattr__(self, name):
        return getattr(self.base, name)

    def begin(self, scene, gauge):
        self.cache.begin(scene=scene, gauge=gauge, model=self.model_id)

    @torch.no_grad()
    def render(self, sources, target_c2w, K, size, seed=0):
        if self._busy:
            raise RuntimeError('Concurrent reconstruction on a shared renderer is unsupported')
        original = self.base.encoder.forward_update
        pending = []

        def update(*args, **kwargs):
            output = original(*args, **kwargs)
            merged, staged, event = self.cache.prepare(output['gaussian'][-1], sources, K, size)
            output = dict(output, gaussian=[*output['gaussian'][:-1], merged])
            pending.append((staged, event))
            return output

        self._busy = True
        self.base.encoder.forward_update = update
        try:
            result = self.base.render(sources, target_c2w, K, size, seed=seed)
            if len(pending) != 1:
                raise RuntimeError('Expected exactly one reconstruction transaction')
            self.cache.commit(*pending[0])
            return result
        finally:
            self.base.encoder.forward_update = original
            self._busy = False

    def release(self):
        # Device offload must not erase the persistent world state.
        self.base.release()


class PersistentGaussianScaffold:
    """Scene-aware adapter for the existing causal scaffold interface."""

    def __init__(self, base, *, scene_id, model_id, max_sources=128):
        self.base = base
        self.scene_id = str(scene_id)
        self.geometry = PersistentReSplatGeometry(base.geometry, model_id=model_id, max_sources=max_sources)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __call__(self, **kwargs):
        self.geometry.begin(self.scene_id, kwargs['gauge'].manifest()['payload_sha256'])
        original = self.base.geometry
        self.base.geometry = self.geometry
        try:
            return self.base(**kwargs)
        finally:
            self.base.geometry = original

    def offload(self):
        self.geometry.release()
