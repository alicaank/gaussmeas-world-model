"""Opt-in memory audit. Scalars/IDs only; no images or model tensors serialized."""
import json
import os
import time

def emit(event, **values):
    path=os.environ.get("GAUSSVERSE_MEMORY_AUDIT")
    if not path:return
    record=dict(event=event,epoch=time.time(),pid=os.getpid(),
                stage_step=int(os.environ.get("GAUSSVERSE_AUDIT_STEP","-1")),
                context=os.environ.get("GAUSSVERSE_AUDIT_CONTEXT"),**values)
    with open(path,"a") as f:f.write(json.dumps(record,allow_nan=False)+"\n")
