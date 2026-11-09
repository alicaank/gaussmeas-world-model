"""'Plain Memorizon' fine-tune (10-22 user request): the Memorizon-50 teacher trained with Memorizon's own recipe as far as the
32 GB card allows, same data / rows / u-table / seed / budget as our 350 run, WITHOUT our additions: no LoRA (--lora_r 0), Plucker /
metadata / role projections frozen at zero (packed_model.PLAIN), no text dropout (Memorizon prompt_dropout 0), Memorizon LRs
(backbone 2e-5, camera 1e-4). Full-backbone training (Memorizon's train_backbone) does not fit; the trainable backbone subset is
the same self-attn + patch embedding as ours. Writes plain.flag next to config.json so evals build the same trainable scope."""
import os, sys
from pathlib import Path
import packed_model
packed_model.PLAIN = True
import train

if __name__ == '__main__':
    out = Path(sys.argv[sys.argv.index('--out') + 1]); out.mkdir(parents=True, exist_ok=True); (out / 'plain.flag').write_text('memorizon recipe: no LoRA, no Plucker/metadata/roles, prompt dropout 0, lr 2e-5/1e-4\n')
    train.main(); os._exit(0)
