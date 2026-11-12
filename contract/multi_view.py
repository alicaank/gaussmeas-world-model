"""More source views for the ReSplat render (user 10 Nov evening). The 540x960 weights were trained with 16 context views, the 512x960 ones
with 8; the rollout feeds 1 (duplicated) to 3, so surfaces stay semi-transparent (alpha .76-.81 on black -> dark) and soft.
 source_ids(s, n, mode)  latent ids for the render of the call whose first new latent is s:
     'cur'     anchor, s-1, s-2                               (stock OwnRender864 without the bank view)
     'recent'  anchor + the n-1 most recent latents            (padded with anchor copies while fewer exist)
     'spread'  anchor, s-1..s-4, the rest evenly spaced over the older latents (padded likewise)
 render_views(...)       DA3 depth once per UNIQUE frame, duplicates share it; FoundationRenderer.render with the padded list.
Returns the renderer's dict at the render size (rgb [-1,1] x valid, valid, conf = alpha x valid, depth)."""
import numpy as np, torch


def source_ids(s, n=8, mode='recent'):
    have = list(range(1, s))                                                   # generated latents 1..s-1
    if mode == 'cur':
        ids = list(dict.fromkeys([0, s - 1, max(0, s - 2)]))
        return ids
    if mode == 'recent':
        ids = [0] + have[::-1][:n - 1]
    elif mode == 'spread':
        near = have[::-1][:4]; old = [i for i in have if i not in near]; k = n - 1 - len(near)
        pick = [old[int(round(j))] for j in np.linspace(0, len(old) - 1, min(k, len(old)))] if old and k > 0 else []
        ids = [0] + near + list(dict.fromkeys(pick))[::-1]
    else:
        raise ValueError(mode)
    return ids + [0] * (n - len(ids))                                          # anchor copies fill the list (same frame, depth, pose)


def render_views(renderer, da3, frames_by_id, c2w_by_id, ids, K864, mpu, target_c2w, gauge_id, boundary, size=(540, 960)):
    import renderer864 as R
    H, W = size; uniq = list(dict.fromkeys(ids)); src = R.upsize(np.stack([frames_by_id[i] for i in uniq]), size)
    K = np.asarray(K864, np.float64).copy(); K[0] *= W / 864; K[1] *= H / 480
    depths, masks = R.da3_gauge_depths(da3, src, K, np.stack([c2w_by_id[i] for i in uniq]), mpu); at = {i: j for j, i in enumerate(uniq)}
    sources = [dict(frame_id=n, rgb=torch.from_numpy(src[at[i]].copy()).permute(2, 0, 1).float() / 127.5 - 1, depth=depths[at[i]], confidence=masks[at[i]],
                    c2w=torch.as_tensor(c2w_by_id[i]).float(), gauge_id=gauge_id) for n, i in enumerate(ids)]
    out = renderer.render(sources, torch.as_tensor(np.asarray(target_c2w)).float(), torch.as_tensor(K).float(), (H, W), seed=0, camera_units='anchor_median_depth',
                          gauge_id=gauge_id, history_boundary=max(boundary, len(ids)))
    return out, sources
