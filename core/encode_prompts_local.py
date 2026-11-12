"""Encode Memorizon's negative prompt, the empty prompt and a prefixed example with the LOCAL umT5-xxl (the same encoder
that produced the corpus captions), padded to 512 as Memorizon expects. CPU, bf16."""
import gm_env  # machine paths: gm_env.py at the repository root (scripts/env.sh puts it on PYTHONPATH)
import sys, torch
sys.path.insert(0, gm_env.MEMORIZON)
from memorizon.pipeline import NEGATIVE_PROMPT
from gmlib import _paths
from gmlib.diffsynth.core.loader.model import load_model
from gmlib.diffsynth.models.wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder
from data import pad_text
dev = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
tok = HuggingfaceTokenizer(str(_paths.WAN22_DIR / 'google/umt5-xxl'), seq_len=None, clean='whitespace', local_files_only=True)
enc = load_model(WanTextEncoder, str(_paths.WAN22_DIR / 'models_t5_umt5-xxl-enc-bf16.pth'), torch_dtype=torch.bfloat16, device=dev).eval()
out = {}
with torch.no_grad():
    for name, prompt in (('negative', NEGATIVE_PROMPT), ('null', ''), ('example_prefixed', 'First-person perspective — character not visible. A row of mattresses is displayed in a store, arranged on a carpeted floor.')):
        ids, mask = tok(prompt, return_mask=True, add_special_tokens=True); e = enc(ids.to(dev), mask.to(dev)).float().cpu()
        e = e[:, :int(mask.sum())] if e.shape[1] > int(mask.sum()) else e; out[name] = pad_text(e); torch.save(out[name], f'memorizon50/{name}_text.pt'); print(name, tuple(e.shape), '->', tuple(out[name].shape), 'std', round(float(e.std()), 4))
