#!/usr/bin/env bash
# Camera-quality manifest (CPU): streams whose DL3DV camera labels are broken -> foundation/camera_quality/manifest.json.
# Flags a stream for: a per-latent step > 2 anchor depths (scale), a frame-to-frame step > max(.5, 20 x median step) (jump), non-finite poses.
# Needs the 448 corpus (vae.npz cameras + anchor.json gauge). On our data: 661 of 9337 streams.
set -euo pipefail; source "$(dirname "$0")/env.sh"; cd "$REPO/foundation/camera_quality"; python camera_quality.py
