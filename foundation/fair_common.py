"""Shared helpers of the fair external-model comparison (user 10-25): batch access, depth remapping between pixel grids,
the command-free DA3 measurement used by the F' arms, and the compact run format the scorers read.

Import-light on purpose (numpy + PIL at module level, torch only inside functions): it is imported by four model venvs
(venv_memorizon / venv_worldplay / venv_vmem / venv_astronex), the wsmetrics env and the framecrafter env.

Batch (batch_ours10/, built by export_batch.py from data864, the same 10 dev videos as our eval864/c10_* runs):
  index.json                 scenes: name (= stream_id[:8], the c10 file names), image, caption, prompts per model, mpu, ...
  images/<name>.png          the dataset's 864x480 frame 0 (frames_864x480/frame_<original_start+1>.jpg), decoded, lossless PNG
  trajectories/<name>.npz    w2c [161,4,4] anchor-relative OpenCV, translation in anchor-median-depth units (batch_dev10 gauge),
                             w2c_overrun [>=189,4,4] (same path continued; only for chunk overrun beyond frame 160, never scored),
                             intrinsics [161,3,3] pixel K at 864x480, anchor_depth [480,864] f16 gauge depth (NaN = no data)
  captions/<name>.json       {"0": dataset caption} (Lyra/batch_dev10 layout)

Run format (lead spec 10-25), written by save_run():
  <out>/<arm>/<name>_s<seed>.mp4  frames 0..160 at the model's NATIVE output resolution, frame 0 = the model's own PROCESSED first
                                  frame (official resize/crop of the 864x480 frame), 1..160 = generated; H.264 libx264 crf 12,
                                  preset slow, yuv420p, 16 fps (40.5 dB on DL3DV frames), ffmpeg 7.0.2 static (FFMPEG below);
                                  the mp4 is renamed into place last, so its existence marks a finished run
  <out>/<arm>/<name>_s<seed>.json poses (planned per frame 0..160 + the harness's planned / commanded / measured, also keyed by
                                  frame index), K_out (pixel K of the mp4 frames), out_hw, preprocess (model name), events, settings
Scored frame indices (identical for every model, incl. our c10 runs): WorldScore-5 and DINO/LPIPS vs the recording: 0 + 1, 5, ...,
157 (the only generated frames our eval864/c10_* runs store); VGGT camera: 0 + latent ends 4, 8, ..., 160 (closed_loop_864).
"""
import gm_env  # machine paths: gm_env.py at the repository root (scripts/env.sh puts it on PYTHONPATH)
import io, json, os, shutil, subprocess
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
BATCH = HERE / 'batch_ours10'
N_FRAMES = 160                                                                    # generated frames scored for every model
WS_IDS = [1 + 4 * i for i in range(40)]                                          # c10 frames (closed_loop_864: gen[1::4])
VGGT_IDS = [4 * j for j in range(1, 41)]                                         # latent ends (closed_loop_864 VGGT views)
SCORE_IDS = sorted(set(WS_IDS) | set(VGGT_IDS))
FFMPEG = gm_env.FFMPEG   # framecrafter imageio_ffmpeg binary
MP4_ARGS = ['-c:v', 'libx264', '-preset', 'slow', '-crf', '12', '-pix_fmt', 'yuv420p']
MEMORIZON_PREFIX = 'First-person perspective — character not visible. '    # memorizon/caption.py required starting text


def index():
    return json.load(open(BATCH / 'index.json'))


def index_scene(name):
    return [x for x in index()['scenes'] if x['name'] == name][0]


def free_gib(path=gm_env.HOME):
    return shutil.disk_usage(path).free / 2 ** 30


def scene(name):
    """Everything a harness needs for one batch video (poses float64, c2w = inv(w2c))."""
    idx = index(); s = [x for x in idx['scenes'] if x['name'] == name][0]
    with np.load(BATCH / 'trajectories' / f'{name}.npz') as z:
        w2c = z['w2c'].astype(np.float64); ext = z['w2c_overrun'].astype(np.float64)
        K = z['intrinsics'][0].astype(np.float64); depth = z['anchor_depth'].astype(np.float32)
    c2w = np.linalg.inv(w2c); c2w_ext = np.linalg.inv(ext)
    c2w_ext[:len(c2w)] = c2w                                                       # identical by construction; scored frames use w2c
    return dict(s, w2c=w2c, c2w=c2w, c2w_ext=c2w_ext, K=K, depth=depth, image_path=str(BATCH / s['image']),
                anchor=np.asarray(Image.open(BATCH / s['image']).convert('RGB')))


def remap_depth(depth, K_src, K_dst, hw_dst):
    """Resample a per-pixel depth map of a pinhole view onto another pixel grid of the SAME camera (resize / crop / stretch:
    only K changes). Bilinear; NaN where the destination ray falls outside the source image or hits a NaN."""
    H, W = hw_dst; h, w = depth.shape
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    us = (u - K_dst[0, 2]) / K_dst[0, 0] * K_src[0, 0] + K_src[0, 2]
    vs = (v - K_dst[1, 2]) / K_dst[1, 1] * K_src[1, 1] + K_src[1, 2]
    inside = (us >= 0) & (us <= w - 1) & (vs >= 0) & (vs <= h - 1)
    us = np.clip(us, 0, w - 1); vs = np.clip(vs, 0, h - 1)
    u0 = np.minimum(np.floor(us).astype(int), w - 2); v0 = np.minimum(np.floor(vs).astype(int), h - 2); a = us - u0; b = vs - v0
    d = depth.astype(np.float64); out = np.zeros((H, W)); bad = ~inside
    for dv, du, wt in ((0, 0, (1 - a) * (1 - b)), (0, 1, a * (1 - b)), (1, 0, (1 - a) * b), (1, 1, a * b)):
        x = d[v0 + dv, u0 + du]; use = wt > 0
        out += np.where(use, wt * np.nan_to_num(x), 0.); bad |= use & ~np.isfinite(x)           # zero-weight NaN neighbours do not count
    out[bad] = np.nan
    return out.astype(np.float32)


def scale_crop_K(K, sx, sy, left=0., top=0.):
    """Pixel K after resizing by (sx, sy) and cropping (left, top) pixels of the resized image."""
    K2 = np.asarray(K, np.float64).copy(); K2[0, 0] *= sx; K2[0, 2] = K2[0, 2] * sx - left; K2[1, 1] *= sy; K2[1, 2] = K2[1, 2] * sy - top
    return K2


def k_memorizon(K, src_wh, out_wh):
    """memorizon/pipeline.py encode_image: centre-crop to the output aspect, then LANCZOS resize (864x480 -> 864x480: identity)."""
    (iw, ih), (w, h) = src_wh, out_wh
    if iw / ih > w / h:
        nw = int(ih * w / h); left, top, cw, ch = (iw - nw) // 2, 0, nw, ih
    else:
        nh = int(iw / (w / h)); left, top, cw, ch = 0, (ih - nh) // 2, iw, nh
    return scale_crop_K(scale_crop_K(K, 1., 1., left, top), w / cw, h / ch), (left, top, left + cw, top + ch)


def k_worldplay(K, src_wh, out_wh):
    """wan/inference/pipeline_wan_w_mem_relative_rope.py resize_center_crop_to_tensor: proportional LANCZOS resize to cover
    (int() truncation), centre crop. 864x480 -> 979x544 -> crop x 9..969."""
    (ow, oh), (tw, th) = src_wh, out_wh
    s = max(tw / ow, th / oh); nw, nh = int(ow * s), int(oh * s); left, top = (nw - tw) // 2, (nh - th) // 2
    return scale_crop_K(K, nw / ow, nh / oh, left, top), (nw, nh, left, top)


def k_vmem(K, src_wh, out_wh=(576, 576)):
    """VMem utils/util.py transform_img_and_K(mode='crop') K branch: cover factor, ceil sizes, area resize, centre crop.
    864x480 -> 1037x576 -> crop x 230..806."""
    import math
    (w, h), (W, H) = src_wh, out_wh
    rb, ar = W / H, w / h
    if rb >= 1.0:
        f = min(H, W) / min(h, w) if ar >= rb else (max(H, W) / min(h, w) if ar < 1.0 else max(H, W) / max(h, w))
    else:
        raise NotImplementedError('portrait targets are not used')
    rh, rw = int(math.ceil(f * h)), int(math.ceil(f * w))
    ct = min(max(0, int(0.5 * rh) - H // 2), rh - H); cl = min(max(0, int(0.5 * rw) - W // 2), rw - W)
    return scale_crop_K(K, rw / w, rh / h, cl, ct), (rw, rh, cl, ct)


def k_astronex(K, src_wh, out_wh=(832, 480)):
    """inference/sample.py I2V transform: torchvision Resize((480, 832)) = anisotropic stretch, no crop."""
    (w, h), (W, H) = src_wh, out_wh
    return scale_crop_K(K, W / w, H / h), (W, H)


def se3_interp(delta, w):
    """verbatim memorizon_plugins/gv_memorizon.py se3_interp (F' replanning: geodesic rotation + linear translation)."""
    from scipy.spatial.transform import Rotation, Slerp
    rot = Rotation.from_matrix(delta[:3, :3])
    r = Slerp([0., 1.], Rotation.concatenate([Rotation.identity(), rot]))(w).as_matrix()
    out = np.eye(4); out[:3, :3] = r; out[:3, 3] = w * delta[:3, 3]
    return out


def _pad44(e):
    e = np.asarray(e, dtype=np.float64)
    if e.shape[-2] == 3:
        o = np.tile(np.eye(4), (e.shape[0], 1, 1)); o[:, :3, :4] = e; return o
    return e


def da3_window(da3, anchor, anchor_depth, frames, K, max_views=20, tail=16):
    """memorizon_plugins/gv_memorizon2.py da3_window (poses only), unchanged except that NaN anchor-depth pixels (outside the
    stored 448x256 field of view) are excluded from the scale median: command-free DA3 on [anchor + frames], per-frame c2w
    relative to the anchor view, ONE scale from DA3's anchor depth vs the stored gauge depth (median 1 = anchor-depth units)."""
    import torch
    with torch.no_grad():
        views = [anchor] + list(frames)
        if len(views) > max_views:
            tl = list(frames[-tail:]); ctx = list(frames[:-tail])
            nctx = max(0, max_views - 1 - len(tl))
            pick = np.linspace(0, len(ctx) - 1, nctx).round().astype(int) if ctx and nctx else []
            views = [anchor] + [ctx[i] for i in pick] + tl
            idx = [None] * (len(frames) - tail) + list(range(len(views) - tail, len(views)))
        else:
            idx = list(range(1, len(views)))
        Ks = np.repeat(np.asarray(K, dtype=np.float32)[None], len(views), 0)
        pred = da3.inference(image=views, intrinsics=Ks, extrinsics=None, align_to_input_extrinsics=False,
                             align_to_input_ext_scale=False, infer_gs=False, process_res=504, export_dir=None)
        w2c = _pad44(pred.extrinsics); rel = np.stack([w2c[0] @ np.linalg.inv(w2c[i]) for i in range(len(views))])
        d0 = np.asarray(pred.depth[0], dtype=np.float64)
        ours = torch.nn.functional.interpolate(torch.as_tensor(anchor_depth)[None, None].float(), size=d0.shape,     # NaN propagates ->
                                               mode='bilinear', align_corners=False)[0, 0].numpy()                     # excluded below
        ok = np.isfinite(d0) & (d0 > 0) & np.isfinite(ours) & (ours > 0); scale = float(np.median(ours[ok] / d0[ok]))
        rel[:, :3, 3] *= scale
        return dict(poses=[None if j is None else rel[j] for j in idx], scale=scale, valid_px=int(ok.sum()))


def _jdefault(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def write_mp4(path, frames, fps=16):
    """uint8 [T,H,W,3] (or list) -> H.264 crf 12 / preset slow / yuv420p mp4 through the static ffmpeg binary."""
    frames = [np.ascontiguousarray(f, dtype=np.uint8) for f in frames]; h, w = frames[0].shape[:2]
    assert h % 2 == 0 and w % 2 == 0, f'yuv420p needs even sizes, got {w}x{h}'
    p = subprocess.Popen([FFMPEG, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
                          *MP4_ARGS, '-threads', '4', '-movflags', '+faststart', '-f', 'mp4', str(path)], stdin=subprocess.PIPE)
    for f in frames:
        assert f.shape[:2] == (h, w)
        p.stdin.write(f.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f'ffmpeg failed writing {path}')


def read_mp4(path, hw=None):
    """-> uint8 [T,H,W,3] (all frames; the same binary decodes, so the rgb<->yuv matrix is the encoder's)."""
    if hw is None:
        import re
        info = subprocess.run([FFMPEG, '-hide_banner', '-i', str(path)], capture_output=True, text=True).stderr
        w, h = map(int, re.search(r'Video: .*?, (\d{2,5})x(\d{2,5})', info).groups())
    else:
        h, w = hw
    raw = subprocess.run([FFMPEG, '-loglevel', 'error', '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def per_frame_poses(log, name):
    """Adds planned_per_frame (0..160) and commanded / measured keyed by frame index (pose_frame_ids: 4j for latent models)."""
    out = dict(planned_per_frame=planned_c2w(name)[:N_FRAMES + 1].tolist())
    ids = log.get('pose_frame_ids')
    if ids is not None:
        for k in ('planned', 'commanded', 'measured'):
            if k in log:
                out[k + '_by_frame'] = {str(f): log[k][i] for i, f in enumerate(ids) if f <= N_FRAMES and i < len(log[k])}
    return out


def save_run(dst, frames, frame0, K, log, preprocess=None):
    """dst = <out>/<arm>/<name>_s<seed>.mp4; frames[f-1] = generated frame f (f = 1..>=160), uint8 HxWx3 native size;
    frame0 = the model's processed first frame (same size); K = pixel K of these frames."""
    dst = Path(dst).with_suffix('.mp4'); dst.parent.mkdir(parents=True, exist_ok=True)
    assert len(frames) >= N_FRAMES, f'{len(frames)} frames < {N_FRAMES}'
    seq = [np.asarray(frame0)] + [np.asarray(frames[f]) for f in range(N_FRAMES)]
    hw = list(seq[1].shape[:2]); assert list(seq[0].shape[:2]) == hw, f'frame 0 {seq[0].shape} vs generated {seq[1].shape}'
    tmp = dst.with_name(dst.stem + '.tmp.mp4'); write_mp4(tmp, seq)
    extra = per_frame_poses(log, log['scene']) if 'scene' in log else {}
    with open(dst.with_suffix('.json.tmp'), 'w') as f:
        json.dump(dict(log, **extra, K_out=np.asarray(K).tolist(), out_hw=hw, frames_in_mp4=len(seq), preprocess=preprocess or log.get('model'),
                       mp4=dict(codec='libx264', crf=12, preset='slow', pix_fmt='yuv420p', fps=16, frame0='processed input frame')), f, default=_jdefault)
    os.replace(dst.with_suffix('.json.tmp'), dst.with_suffix('.json'))
    os.replace(tmp, dst)                                                              # mp4 last: its existence marks a finished run
    return dst


def load_run(mp4):
    """Fair-format run -> frame0 (uint8), {frame index: uint8 frame} for 0..160, K (3x3), (H, W), log."""
    log = json.load(open(Path(mp4).with_suffix('.json'))); fr = read_mp4(mp4, log.get('out_hw'))
    K = np.asarray(log['K_out'], np.float64)
    return fr[0], {i: fr[i] for i in range(len(fr))}, K, tuple(fr.shape[1:3]), log


def load_lyra(frames_dir, name):
    """Lyra-2 run (lyra2_custom_traj_inference --save_png_frames: <out>/<arm>/<name>_frames/frame_%05d.png, 161 frames, frame 0 =
    Lyra's decode of the anchor). Frame 0 is replaced by Lyra's processed input (lyra2_custom_traj_inference.py:418-424 ->
    cv2.resize INTER_LINEAR stretch to the output size); K = batch K rescaled by (W/864, H/480) as its load_trajectory does."""
    files = sorted(Path(frames_dir).glob('frame_*.png'))
    fr = {i: np.asarray(Image.open(f).convert('RGB')) for i, f in enumerate(files)}
    H, W = fr[0].shape[:2]; sc = index_scene(name)
    with np.load(BATCH / 'trajectories' / f'{name}.npz') as z:
        K = z['intrinsics'][0].astype(np.float64); h0, w0 = int(z['image_height']), int(z['image_width'])
    fr[0] = preprocess('lyra', np.asarray(Image.open(BATCH / sc['image']).convert('RGB')), (W, H))
    return fr[0], fr, scale_crop_K(K, W / w0, H / h0), (H, W), dict(model='lyra', preprocess='lyra')


def preprocess(model, img, out_wh=None):
    """Each model's OFFICIAL first-frame preprocessing applied to an 864x480 uint8 frame (used for frame 0 of Lyra runs and for
    the recording frames the DINO/LPIPS references are made from, so generated and reference frames share one geometry).
    Geometry = fair_common.k_<model> (verify_batch.py checks both against the official code paths)."""
    pil = Image.fromarray(np.asarray(img, np.uint8)); w, h = pil.size
    if model == 'memorizon':                                                         # pipeline.encode_image: crop to aspect + LANCZOS
        W, H = out_wh or (864, 480); _, (l, t, r, b) = k_memorizon(np.eye(3), (w, h), (W, H))
        return np.asarray(pil.crop((l, t, r, b)).resize((W, H), Image.LANCZOS))
    if model == 'worldplay':                                                         # resize_center_crop_to_tensor
        W, H = out_wh or (960, 544); _, (nw, nh, l, t) = k_worldplay(np.eye(3), (w, h), (W, H))
        return np.asarray(pil.resize((nw, nh), Image.LANCZOS).crop((l, t, l + W, t + H)))
    if model == 'astronex':                                                          # torchvision Resize((480, 832)) on PIL = BILINEAR
        W, H = out_wh or (832, 480); return np.asarray(pil.resize((W, H), Image.BILINEAR))
    if model == 'spatia':                                                            # spatia_fprime: PIL .resize((W, H)) default filter
        W, H = out_wh or (1248, 704); return np.asarray(pil.resize((W, H)))
    if model == 'lyra':                                                              # cv2.resize INTER_LINEAR stretch
        import cv2
        W, H = out_wh or (832, 480); return cv2.resize(np.asarray(img, np.uint8), (W, H), interpolation=cv2.INTER_LINEAR)
    if model == 'matrixgame':                                                        # MG3.5 unified_dataset._video_frames_uint8_to_normalized_tensor:
        import torch                                                                 # centre crop only if larger, then bilinear antialias stretch
        W, H = out_wh or (1280, 704); a = np.asarray(img, np.uint8); h0, w0 = a.shape[:2]
        ch, cw = min(h0, H), min(w0, W); t, l = max((h0 - ch) // 2, 0), max((w0 - cw) // 2, 0); a = a[t:t + ch, l:l + cw]
        x = torch.from_numpy(np.array(a)).permute(2, 0, 1)[None].float()
        x = torch.nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=False, antialias=True)
        return x[0].permute(1, 2, 0).round().clamp(0, 255).byte().numpy()
    if model == 'vmem':                                                              # load_img_and_K + transform_img_and_K(crop) + tensor_to_pil
        import torch
        W, H = out_wh or (576, 576); _, (rw, rh, cl, ct) = k_vmem(np.eye(3), (w, h), (W, H))
        x = torch.from_numpy(np.asarray(img, np.float32) / 255.).permute(2, 0, 1)[None]
        x = torch.nn.functional.interpolate(x, (rh, rw), mode='area')[0, :, ct:ct + H, cl:cl + W]
        return np.clip(x.permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)
    raise ValueError(model)


def iter_runs(kind, root):
    """-> (arm, scene, loader) for every finished run under root: fair mp4 runs (<root>/<arm>/<name>_s<seed>.mp4 + .json); for
    kind 'lyra' the Lyra-2 PNG runs (<root>/<arm>/<name>_frames/frame_%05d.png, complete = >= 161 frames) first, then fair mp4
    runs made from them by lyra_to_mp4() for scenes whose PNG folder is gone."""
    root = Path(root); names = {s['name'] for s in index()['scenes']}; seen = set()
    if kind == 'lyra':
        for d in sorted(root.glob('*/*_frames')):
            name = d.name[:-len('_frames')]
            if name in names and len(list(d.glob('frame_*.png'))) >= N_FRAMES + 1:
                seen.add((d.parent.name, name)); yield d.parent.name, name, (lambda d=d, name=name: load_lyra(d, name))
    for f in sorted(root.glob('*/*_s*.mp4')):
        arm, name = f.parent.name, f.stem.rsplit('_s', 1)[0]
        if not f.name.endswith('.tmp.mp4') and f.with_suffix('.json').exists() and (arm, name) not in seen:
            yield arm, name, (lambda f=f: load_run(f))


def lyra_to_mp4(root, seed=7):
    """Write the fair mp4 + json next to every complete Lyra-2 PNG folder (frame 0 = Lyra's processed input, K as load_lyra);
    the PNG folder is left in place (its owner may remove it afterwards to save ~75 MB per run)."""
    for d in sorted(Path(root).glob('*/*_frames')):
        name = d.name[:-len('_frames')]; dst = d.parent / f'{name}_s{seed}.mp4'
        if dst.exists() or len(list(d.glob('frame_*.png'))) < N_FRAMES + 1:
            continue
        f0, fr, K, hw, _ = load_lyra(d, name)
        save_run(dst, [fr[i] for i in range(1, N_FRAMES + 1)], f0, K, dict(model='lyra', scene=name, arm=dict(name=d.parent.name), seed=seed,
                 source_png_dir=str(d), planned=planned_c2w(name)[:N_FRAMES + 1].tolist(), pose_frame_ids=list(range(N_FRAMES + 1))), preprocess='lyra')
        print('wrote', dst, flush=True)


def planned_c2w(name):
    """Per-frame planned path 0..160 (= our recorded c2w, the camera truth of the c10 runs)."""
    with np.load(BATCH / 'trajectories' / f'{name}.npz') as z:
        return np.linalg.inv(z['w2c'].astype(np.float64))


if __name__ == '__main__':                                                            # python fair_common.py lyra2mp4 RUNS/lyra_dmd
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == 'lyra2mp4':
        lyra_to_mp4(sys.argv[2])
    else:
        sys.exit('usage: fair_common.py lyra2mp4 ROOT')
