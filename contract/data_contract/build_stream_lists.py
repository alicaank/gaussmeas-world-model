"""Stream exclusion lists of the contract line, from their recorded rules (the committed lists were made with inline code on 11-12 Nov; this script
reproduces them and is what to run on new data).
  flag   streams whose renders say the camera labels are inconsistent: >= 2 cached windows in the render cache and at least half of them below 20 dB
         first-target PSNR (build_stats.jsonl; the first-call window counts). On render_cache_v4 this is flag_streams_v4.json (8-character prefixes).
         bad_streams_v3.json is the older list from cache v3 (continuation windows only; 119 streams; the list stage A excluded), kept as committed.
  fast   median per-call translation above 1.0 gauge units over the stream's index windows (windows.jsonl), not already in bad_streams_v3
         -> fast_streams_v4.json (training p90 is .62 units, test plans <= .27).
Usage: build_stream_lists.py [--cache DIR] [--write 1]   (CPU; without --write it only compares with the committed lists)"""
import argparse, collections, json
from pathlib import Path
import numpy as np
import gm_env

HERE = Path(__file__).resolve().parent
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--cache', type=Path, default=Path(gm_env.CONTRACT) / 'render_cache_v4'); ap.add_argument('--write', type=int, default=0); a = ap.parse_args()
    index = json.load(open(HERE / 'index_v4.json')); bad = set(json.load(open(HERE / 'bad_streams_v3.json'))['streams'])
    by = collections.defaultdict(list)
    for r in map(json.loads, (a.cache / 'build_stats.jsonl').open()):
        if r.get('psnr') is not None: by[r['stream'][:8]].append(r['psnr'])
    flag = sorted(s for s, p in by.items() if len(p) >= 2 and sum(x < 20 for x in p) * 2 >= len(p))
    trans = collections.defaultdict(dict)
    for w in map(json.loads, (HERE / 'windows.jsonl').open()): trans[w['stream']][int(w['start'])] = w['trans']
    fast = sorted(s for s, starts in index.items() if s not in bad and starts and float(np.median([trans[s][int(x)] for x in starts if int(x) in trans[s]])) > 1.0)
    for name, new in (('flag_streams_v4', flag), ('fast_streams_v4', fast)):
        f = HERE / f'{name}.json'; old = json.load(open(f)) if f.exists() else None
        print(f'{name}: {len(new)} streams' + ('' if old is None else f'; committed list {len(old["streams"])}, identical: {sorted(old["streams"]) == new}'))
        if a.write: f.write_text(json.dumps(dict(rule=(old or {}).get('rule', 'see build_stream_lists.py'), streams=new), indent=1))
