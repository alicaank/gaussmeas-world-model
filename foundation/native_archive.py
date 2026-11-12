"""Read-only training selection from the versioned v2 camera-proxy audit.

No RGB/depth or generator is loaded. This offline selector is not a deployment
retriever: v2's depth scale uses the whole recorded camera trajectory. Indices
remain causal; useful means positive marginal sampled FOV coverage, not verified
surface support. Geometry recipes and their source indices are never rewritten.
"""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
DEFAULT_AUDIT=HERE.parents[1]/'analysis/plan_base_restart_20261008/archive_support_20261009/v2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class NativeArchiveIndex:
    def __init__(self, audit_root=DEFAULT_AUDIT, data_root=HERE/'dataset_v3', cache_streams=128):
        self.root=Path(audit_root);self.cache=OrderedDict();self.cache_streams=cache_streams
        manifest=json.loads((self.root/'manifest.json').read_text())
        progress=json.loads((self.root/'progress.json').read_text())
        self.config=manifest['config'];cfg=dict(self.config);digest=cfg.pop('digest')
        if hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()!=digest:
            raise ValueError('native audit config digest mismatch')
        if (progress['status']!='complete' or progress['completed_streams']!=manifest['planned_streams']
                or progress['statuses']!={'complete':manifest['planned_streams']}):
            raise ValueError('native audit must be complete without skipped streams')
        if (self.config['version']!='archive-frustum-v2' or self.config['history_K']!=8
                or self.config['source_stride_rgb']!=4):
            raise ValueError('unsupported native audit contract')
        for filename,key in [('corpus/streams.jsonl','streams_sha256'),
                             ('sampler/round_000_recipes.jsonl','recipes_sha256')]:
            if sha(Path(data_root)/filename)!=self.config[key]:
                raise ValueError('native audit inputs differ from training corpus: '+filename)

    def _load(self,row):
        sid=row['stream_id']
        if sid in self.cache:
            self.cache.move_to_end(sid);return self.cache[sid]
        path=self.root/'streams'/f'{sid}.json'
        meta=json.loads(path.read_text())
        for key in ('stream_id','scene_id','split','original_start','original_end_exclusive',
                    'latent_count','transforms_sha256'):
            if meta[key]!=row[key]:raise ValueError('native audit identity mismatch: '+key)
        if meta['config_digest']!=self.config['digest'] or meta['status']!='complete':
            raise ValueError('native audit provenance mismatch')
        if sha(path.with_suffix('.npz'))!=meta['npz_sha256']:
            raise ValueError('native audit shard changed')
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
            starts=z['start'].copy();selected=z['selected_latent_indices'][1].copy()
            scores=z['exclusive'][1].copy()
        if len(set(starts.tolist()))!=len(starts):raise ValueError('duplicate audited prefix')
        value={int(s):(ids,curve) for s,ids,curve in zip(starts,selected,scores)}
        self.cache[sid]=value
        if len(self.cache)>self.cache_streams:self.cache.popitem(last=False)
        return value

    def select(self,row,start,max_refs=6):
        if row['split']!='train':raise ValueError('v2 native archive index is training-only')
        if not isinstance(max_refs,int) or not 0<=max_refs<=6:raise ValueError('archive budget must be 0..6')
        if not 1<=start<=row['target_start_latent_max']:raise ValueError('invalid target prefix')
        if start<=9:
            ids=np.empty(0,dtype=np.int32);curve=np.zeros(7,dtype=np.float32)
        else:
            found=self._load(row).get(start)
            if found is None:raise ValueError('prefix absent from native audit; do not silently fall back')
            padded,curve=found;ids=padded[padded>=0]
            if len(ids)>6 or len(set(ids.tolist()))!=len(ids) or any(i<=0 or i>=start-8 for i in ids):
                raise ValueError('native source indices violate causality/uniqueness')
            gains=np.diff(curve)
            if (gains<0).any() or np.count_nonzero(gains>0)!=len(ids):
                raise ValueError('native source slots disagree with marginal gains')
        chosen=ids[:max_refs].tolist()
        return dict(policy='v2_greedy_base_positive_marginal_gain',archive_indices=sorted(chosen),
            greedy_indices=chosen,useful_count_up_to_6=len(ids),budget_max=max_refs,
            selected_count=len(chosen),sampled_exclusive_coverage=float(curve[max_refs]),
            audit_config_digest=self.config['digest'],proxy_only=True,
            deployment_validated=False,render_sources_independent=True)
