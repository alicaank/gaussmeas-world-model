"""Causal external-model controller utilities. No reference future RGB is used."""
import gm_env  # machine paths: gm_env.py at the repository root (scripts/env.sh puts it on PYTHONPATH)
from pathlib import Path
import json,os,subprocess,time
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
R=Path(__file__).resolve().parent

def rejoin(measured,previous_command,future_plan):
 p=np.asarray(future_plan,np.float64).copy();n=len(p)
 if np.array_equal(measured,previous_command):return p
 dt=measured[:3,3]-previous_command[:3,3]
 dr=Rotation.from_matrix(measured[:3,:3]@previous_command[:3,:3].T)
 sl=Slerp([0,1],Rotation.concatenate([dr,Rotation.identity()]))
 for i in range(n):
  a=(i+1)/n;p[i,:3,3]+=(1-a)*dt;p[i,:3,:3]=sl(a).as_matrix()@p[i,:3,:3]
 return p

class PoseWorker:
 def __init__(self,out):
  out=Path(out);out.mkdir(parents=True,exist_ok=True);self.out=out
  root=R.parent/'spatia_plugins';self.log=open(out/'measurement.log','a')
  env=dict(os.environ,HF_HUB_OFFLINE='1');env.pop('LD_LIBRARY_PATH',None);env.pop('GV_DA3_FAKE',None)
  self.p=subprocess.Popen([(gm_env.BASELINES + '/venv_vmem/bin/python'),str(root/'spatia_da3_worker.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1,env=env)
  line=self.p.stdout.readline();assert line.startswith('READY '),line
  info=json.loads(line[6:]);assert not info.get('fake');self.ready=info
 def measure(self,frames,K,anchor_depth):
  req=self.out/'request.npz';out=self.out/'poses.npz';last=len(frames)-1
  np.savez(req,anchor=frames[0],frames=frames[1:],K=np.asarray(K,np.float32),query=np.array([last]),anchor_depth=np.asarray(anchor_depth,np.float32),max_views=np.int64(16),tail=np.int64(1))
  self.p.stdin.write(json.dumps({'cmd':'measure','req':str(req),'out':str(out)})+'\n');self.p.stdin.flush()
  line=self.p.stdout.readline()
  if not line:raise RuntimeError('DA3 worker exited')
  info=json.loads(line)
  if not info.get('ok'):raise RuntimeError(str(info))
  with np.load(out) as z:m=z['poses'][0].copy()
  if not np.isfinite(m).all():raise RuntimeError('Nonfinite measured endpoint')
  # Request is scratch IPC, overwritten on every call; retain poses and logs.
  # Failed measurements keep their input for diagnosis.
  req.unlink()
  return m,info
 def close(self):
  if self.p.poll() is None:
   self.p.stdin.write('{"cmd":"quit"}\n');self.p.stdin.flush()
   try:self.p.wait(timeout=30)
   except subprocess.TimeoutExpired:self.p.terminate();self.p.wait(timeout=20)
  self.log.close()
