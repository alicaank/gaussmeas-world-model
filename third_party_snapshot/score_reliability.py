"""Separate diagnostic scorer; never replaces primary VIPE v2 results."""
import argparse,sys
from pathlib import Path
R=Path(__file__).resolve().parent;E=R.parents[1]/'plan_base_restart_20261008/vipe_eval_20261030'
p=argparse.ArgumentParser(add_help=False);p.add_argument('--style',default='long',choices=['long','pair']);a,rest=p.parse_known_args();sys.argv=[sys.argv[0],*rest]
source=(R/'score_snapshot.py').read_text();scope=dict(__name__='diagnostic_score',__file__=str(E/'score.py'))
exec(compile(source,str(R/'score_snapshot.py'),'exec'),scope)
if __name__=='__main__':scope['main'](a.style)
