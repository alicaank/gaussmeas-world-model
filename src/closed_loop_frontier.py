"""closed_loop_864.py with the frontier mask installed (inference-only test of the observed-support render mask).
Usage: closed_loop_frontier.py --frontier support|off [--frontier_close 1] [--frontier_log path] <closed_loop_864.py args>"""
import os
import sys
from pathlib import Path

FT = '/home/hj/Alican/analysis/plan_base_restart_20261008/fresh_teacher_20261021'
HERE = str(Path(__file__).resolve().parent)
for p in (HERE, FT):
    if p not in sys.path:
        sys.path.insert(0, p)


def pop_arg(name, default, cast=str):
    if name in sys.argv:
        i = sys.argv.index(name); v = cast(sys.argv[i + 1]); del sys.argv[i:i + 2]; return v
    return default


MODE = pop_arg('--frontier', 'reproj'); CLOSE = pop_arg('--frontier_close', 1, int); LOG = pop_arg('--frontier_log', None)
if LOG is None and '--out' in sys.argv:
    out = Path(sys.argv[sys.argv.index('--out') + 1]); out.mkdir(parents=True, exist_ok=True); LOG = str(out / 'frontier_log.jsonl')
import frontier  # noqa: E402
frontier.install(MODE, CLOSE, LOG, tol=pop_arg('--frontier_tol', 0.15, float), margin=pop_arg('--frontier_margin', 2, int), dump=pop_arg('--frontier_dump', None))
import closed_loop_864  # noqa: E402

if __name__ == '__main__':
    print('frontier', frontier.STATE, flush=True)
    closed_loop_864.main(); os._exit(0)
