"""Load the official Memorizon model from the local 50-step directory without HF's meta-device path."""
import glob, torch
from safetensors.torch import load_file


def load_official(directory, device='cuda', width=864, height=480):
    from memorizon.model import MemorizonModel, MemorizonConfig
    cfg = MemorizonConfig.from_pretrained(directory); cfg.image_width, cfg.image_height = width, height      # checkpoint config says 1280x704; PRoPE patch grid must match our frames
    m = MemorizonModel(cfg, _from_pretrained=True)
    sd = {}
    for f in sorted(glob.glob(f'{directory}/model-*.safetensors')):
        sd.update(load_file(f))
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith('timesteps')]
    if missing or unexpected:
        raise RuntimeError(f'official load: missing {missing[:5]} unexpected {unexpected[:5]}')
    return m.to(torch.bfloat16).to(device).eval()
