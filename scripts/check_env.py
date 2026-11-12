"""Report what the pipeline can see on this machine: code roots, external components, data. Usage: source scripts/env.sh; python scripts/check_env.py"""
import importlib.util, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # works without scripts/env.sh too
import gm_env

ok = True


def line(label, path, need=True):
    global ok
    there = Path(path).exists(); ok &= there or not need
    print(f'  [{"ok" if there else ("MISSING" if need else "absent ")}] {label:34s} {path}')


print('code'); [line(n, getattr(gm_env, n)) for n in ('CORE', 'CONTRACT', 'FOUNDATION')]; line('gaussverse (vendored)', gm_env.PKG_ROOT + '/gaussverse/__init__.py')
print('external code')
line('ReSplat source', gm_env.RESPLAT); line('Depth-Anything-3 source', gm_env.DA3_SRC); line('Memorizon code', gm_env.MEMORIZON + '/memorizon'); line('Wan2.2-TI2V-5B (GAUSSVERSE_WAN22_DIR)', __import__('os').environ.get('GAUSSVERSE_WAN22_DIR', ''))
for m in ('torch', 'safetensors', 'einops', 'cv2', 'lpips', 'bitsandbytes'):
    there = importlib.util.find_spec(m) is not None; ok &= there; print(f'  [{"ok" if there else "MISSING"}] python module {m}')
print('data (built by the pipeline; docs/PIPELINE.md)')
line('raw frames 864x480', gm_env.DATA_ROOT + '/DL3DV-960P-864x480', need=False); line('448 corpus', gm_env.FOUNDATION + '/dataset_v3/corpus/streams.jsonl', need=False)
line('camera-quality manifest', gm_env.CAMERA_QUALITY + '/manifest.json', need=False); line('864 corpus', gm_env.CORE + '/data864/streams.jsonl', need=False)
line('base weights (Memorizon-50)', gm_env.CORE + '/memorizon50', need=False); line('window index', gm_env.CONTRACT + '/data_contract/index_v4.json', need=False)
line('render cache v4', gm_env.CONTRACT + '/render_cache_v4/_done', need=False); line('evaluation plans', gm_env.EVAL_BATCH + '/index.json', need=False)
print('all required code present' if ok else 'something required is missing'); sys.exit(0 if ok else 1)
