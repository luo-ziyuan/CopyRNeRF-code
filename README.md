# CopyRNeRF
Our model is trained on 8 GPUs.
## Quick Start
```bash
python run_copyrnerf.py --config configs/lego_8b.txt
```
## Inference
```bash
python run_copyrnerf.py --config configs/lego_8b.txt --render_only --ckpt <direction-to-ckpt>
```
