#!/usr/bin/env bash
# The 864 training corpus (GPU: Wan2.2 VAE + DA3) -> core/data864/{streams.jsonl, components/<stream>/{vae.npz, meta.json}}.
# Per stream: latents [1,48,L,30,54] of the 864x480 frames, K at 864x480, cameras in the anchor gauge, metres_per_anchor_unit from DA3 metric depth
# on the anchor; text and gauge are referenced from the 448 corpus. Streams in the camera-quality manifest are skipped; 20 streams (seed 864) are
# the development split. Needs: scripts 01-03, and the 448x256 anchor frames of DL3DV-480P. Our result: 915 streams (895 train / 20 development).
set -euo pipefail; source "$(dirname "$0")/env.sh"; cd "$REPO/core"; python prep_864.py "$@"
