"""CPU-only full-rate visual evidence; does not alter model inputs or scoring."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import imageio_ffmpeg


def save_comparison(path,recorded,generated,playback_fps=16,chunk_frames=16):
    path=Path(path)
    if recorded.shape!=generated.shape or generated.ndim!=4 or generated.shape[-1]!=3:
        raise ValueError('comparison requires matched RGB sequences')
    if recorded.dtype!=np.uint8 or generated.dtype!=np.uint8:
        raise ValueError('comparison requires uint8 RGB')
    n,h,w,_=generated.shape
    if n<1 or h%2 or w%2:raise ValueError('invalid video dimensions')
    if path.exists():raise FileExistsError(path)
    tmp=path.with_name(path.stem+'.partial.mp4')
    writer=imageio_ffmpeg.write_frames(str(tmp),(2*w,h+32),fps=playback_fps,
        codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',
        output_params=['-crf','18','-preset','fast','-threads','2','-movflags','+faststart'])
    writer.send(None)
    try:
        for i in range(n):
            board=Image.new('RGB',(2*w,h+32),'white')
            draw=ImageDraw.Draw(board)
            boundary=i>0 and (i-1)%chunk_frames==0
            label='anchor' if i==0 else f'call {(i-1)//chunk_frames+1}'
            draw.text((6,8),f'Recorded | RGB {i} | {label}',fill='black')
            draw.text((w+6,8),f'Generated | RGB {i} | {label}',fill='black')
            if boundary:draw.rectangle((0,28,2*w-1,31),fill=(220,70,30))
            board.paste(Image.fromarray(recorded[i]),(0,32))
            board.paste(Image.fromarray(generated[i]),(w,32))
            writer.send(np.asarray(board))
    finally:
        writer.close()
    if not tmp.exists() or tmp.stat().st_size==0:raise RuntimeError('video encoder produced no artifact')
    tmp.replace(path)
    report=dict(path=str(path),frames=n,rgb_indices=[0,n-1],playback_fps=playback_fps,
        dimensions=[2*w,h+32],chunk_frames=chunk_frames,
        note='All RGB frames retained in compressed visualization; orange line marks first frame of each generated chunk. Playback rate is presentation only. Metrics use original arrays.')
    path.with_suffix('.json').write_text(json.dumps(report,indent=2))
    return report
