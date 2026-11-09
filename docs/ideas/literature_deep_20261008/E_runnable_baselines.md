# Report E — Runnable camera-controlled baselines on one RTX 5090 (subagent, 2026-10-08; V = read on repo/README/model card)
Already local with frames on all 8 held-out scenes (same trajectories): Lyra 2 (Wan2.1-14B), HY-WorldPlay-5B (distilled), SANA-WM stage-1.

A. Ready on 32 GB with single image + explicit cameras
- Wan2.2-Fun-5B-Control-Camera (alibaba-pai; VideoX-Fun; Apache-2.0): image + RE10K-style pose txt -> Plücker control video; 81 frames per clip (trained 121); chain 2 clips by re-using the last frame; no memory; same TI2V-5B backbone as ours. Entry: examples/wan2.2_fun/predict_v2v_control_camera.py. Memoryless exact-camera control baseline.
- Stable Virtual Camera / SEVA (Stability; non-commercial, gated): image + transforms.json (poses + intrinsics), 1.3B, chunked anchor-then-interpolate up to ~1000 frames; VRAM not stated. Exact-pose NVS control arm, not autoregressive.
- CameraCtrl SVD branch (Apache-2.0): image + RE10K pose txt, 14/25 frames at 320x576; chain 7-12 clips; old 2024 baseline.
B. Possible with offload
- Memorizon (TingtingLiao/memorizon, Apache-2.0, arXiv 2610.00544): Wan2.2-TI2V-5B, retrieval memory, 4-step distilled, 864x480, one pose per latent (T = 1 + 4 s), no intrinsics input; README says >= 40 GB; offload likely enough (unverified). Closest apples-to-apples memory baseline on our base.
- HY-WorldPlay WAN-5B: pose.json {extrinsic, K} per latent exists; whether the 5B pipeline honours arbitrary poses is undocumented; licence excludes EU/UK/KR. Repo no longer on disk here.
- Lyra 2.0: 14B, tested on 80 GB; our previous runs exist (how they ran on this machine is in GAUSSVERSE_DIRECT_COMPARISON.md).
- GEN3C (~43 GB with offload), Voyager (60 GB), AlayaWorld (33.6 GB weights), Warp-as-History (46-54 GB), WorldWarp (~40 GB), Uni3C (50.8 GB): VRAM-blocked as shipped. ViewCrafter fits (23.5 GB) but 25 frames/clip with d_phi/d_theta trajectories, needs a patch for explicit poses.
C. Not applicable / no weights: ReCamMaster, DepthDirector (video-to-video), Matrix-3D (panorama), WorldMem / Infinite-World (discrete actions), Context-as-Memory (no code), Mirage/LatentSpatialMemory (code, no checkpoints), GeoVerse (README only), MeRoPE ("coming soon"), h-control and CameraCtrl II (no public code found), VerseCrafter (8-GPU).
Recommendation: add Wan2.2-Fun-5B-Control-Camera (memoryless, same backbone), Memorizon (memory, same backbone), SEVA (NVS control arm); keep Lyra 2, WorldPlay-5B, SANA-WM from the existing fair-pilot frames.
