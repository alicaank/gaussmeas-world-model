# Ours vs Lyra-2 / Spatia / Matrix-Game (DL3DV, 480 frames)

Position error and endpoint in anchor-depth units; rotation in degrees; columns are means over frames 1–480 except the last-third columns (frames 321–480). "+ correction" = the paper's closed-loop controller wrapped around the released model. Bold = best.

### A. Scenes shared by all four models
Scenes: 02267acf, 0238de14, 334ff324 (seed 7, 480 frames, one scorer: `vipe_native_map_mask_diagnostic_v3`)

| Method | Position err ↓ | Endpoint ↓ | Rotation ° ↓ | LPIPS ↓ | DINO ↑ | LPIPS last⅓ ↓ | DINO last⅓ ↑ |
|---|---|---|---|---|---|---|---|
| Ours | 0.0374 | 0.134 | 2.21 | 0.366 | 0.895 | 0.525 | 0.814 |
| Ours + correction | 0.0208 | 0.055 | 0.83 | 0.333 | 0.890 | 0.492 | 0.805 |
| Lyra-2 | 0.0599 | 0.143 | 0.17 | 0.317 | 0.944 | 0.471 | 0.911 |
| Lyra-2 + correction | 0.0175 | 0.032 | **0.14** | **0.266** | **0.956** | **0.377** | **0.932** |
| Spatia | 0.0812 | 0.244 | 3.33 | 0.371 | 0.926 | 0.535 | 0.881 |
| Spatia + correction | 0.0903 | 0.282 | 4.87 | 0.381 | 0.908 | 0.543 | 0.848 |
| Matrix-Game 3.5 Distilled | 0.2218 | 0.555 | 3.58 | 0.569 | 0.805 | 0.723 | 0.640 |
| Matrix-Game 3.5 Distilled + correction | 0.0277 | 0.045 | 0.59 | 0.443 | 0.859 | 0.595 | 0.774 |
| Matrix-Game 3.5 Base | 0.0119 | 0.026 | 0.36 | 0.329 | 0.943 | 0.450 | 0.911 |
| Matrix-Game 3.5 Base + correction | **0.0106** | **0.016** | 0.41 | 0.324 | 0.949 | 0.444 | 0.924 |

### B. Scenes shared by ours, Lyra-2 and Spatia
Scenes: 02267acf, 0238de14, 334ff324, 61b27b61, a126efe8 (seed 7, 480 frames, one scorer: `vipe_native_map_mask_diagnostic_v3`)

| Method | Position err ↓ | Endpoint ↓ | Rotation ° ↓ | LPIPS ↓ | DINO ↑ | LPIPS last⅓ ↓ | DINO last⅓ ↑ |
|---|---|---|---|---|---|---|---|
| Ours | 0.0505 | 0.154 | 2.54 | 0.378 | 0.855 | 0.552 | 0.743 |
| Ours + correction | 0.0336 | 0.093 | 0.85 | 0.356 | 0.853 | 0.521 | 0.733 |
| Lyra-2 | 0.0732 | 0.187 | **0.17** | 0.333 | 0.925 | 0.484 | 0.880 |
| Lyra-2 + correction | **0.0333** | **0.068** | 0.24 | **0.297** | **0.936** | **0.416** | **0.898** |
| Spatia | 0.0749 | 0.214 | 2.60 | 0.381 | 0.906 | 0.550 | 0.837 |
| Spatia + correction | 0.0870 | 0.265 | 3.69 | 0.385 | 0.896 | 0.553 | 0.820 |
| Matrix-Game 3.5 Distilled | 0.1952 | 0.554 | 3.47 | 0.559 | 0.798 | 0.712 | 0.633 |
| Matrix-Game 3.5 Distilled + correction | 0.0540 | 0.214 | 0.55 | 0.459 | 0.832 | 0.625 | 0.715 |

Missing (row dropped if any scene missing): Matrix-Game 3.5 Base + correction 61b27b61; Matrix-Game 3.5 Base + correction a126efe8; Matrix-Game 3.5 Base 61b27b61; Matrix-Game 3.5 Base a126efe8
