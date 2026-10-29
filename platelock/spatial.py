"""Optional conservative whole-clip translation; never invents border pixels."""
import json
import shutil
import time
from fractions import Fraction
from pathlib import Path
import cv2
import numpy as np
from .video import FrameReader, FrameWriter, probe, decode_small_rgb
from .drift import measure, summary
from .plan import Op
from .compare import compare_video
from .viewer import export


def framing_only(plate_path, gen_path, out_dir, analysis, codec='h264', mask=None):
    """Correct same-time framing while preserving every generated frame and its FPS.

    Temporal correspondence is deliberately not taken from an uncertain retiming
    plan. This mode requires matching timebases; the spatial fit must independently
    pass the existing shared-feature and withheld-frame checks.
    """
    from comfy_api.input_impl import VideoFromFile
    from .plan import plan_to_json
    pi, gi = probe(plate_path), probe(gen_path)
    if pi.fps != gi.fps:
        raise ValueError('Framing-only repair needs matching input frame rates; it never changes the timebase.')
    common = min(pi.frames, gi.frames)
    ops = [Op(i, 'copy', i) for i in range(common)]
    summary = dict(plate_frames=pi.frames, gen_frames=gi.frames, copied=gi.frames,
                   interpolated=[], duplicates_in_gen=[], dropped_gen_frames=[],
                   missing_plate_frames=[], segments=[], offset_median=0,
                   complete=True, compared_frames=common, timing_verified=False)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{Path(gen_path).stem}_framing_only_{time.time_ns()}'
    # Stage the original so correct_framing writes into the output directory,
    # never next to or over the user's input file.
    staged = out_dir / (stem + ('.mov' if codec == 'prores' else '.mp4'))
    shutil.copyfile(gen_path, staged)
    viewer = out_dir / (stem + '_viewer')
    manifest = dict(name=Path(gen_path).stem, fps=str(gi.fps), label='OUTPUT',
                    analysis_only=False, ops=plan_to_json(ops), generated=[],
                    drift=[None]*common, exactness={}, summary=summary,
                    timing_preserved=True, timing_analysis=analysis.get('summary', {}),
                    preserved_frames=gi.frames, compared_frames=common,
                    repair_task='Correct framing',
                    repair_scope='Fixed framing only; original timing preserved and not verified',
                    timebase=analysis.get('timebase'))
    export(viewer, plate_path, gen_path, None, manifest)
    sidecar = out_dir / (stem + '.json')
    source = dict(plate=pi.to_dict(), generated=gi.to_dict(), ops=plan_to_json(ops),
                  summary=summary, timing_preserved=True,
                  timing_analysis=analysis.get('summary', {}))
    sidecar.write_text(json.dumps(source, indent=2))
    state = dict(sidecar=str(sidecar), viewer=str(viewer), codec=codec,
                 analysis_only=False, fps=str(gi.fps), frames=gi.frames,
                 missing=[], filled=[], generate=False, work=str(out_dir/stem), summary=summary)
    report = f'Framing only: all {gi.frames} AI frames preserved at {gi.fps} fps. Timing was not repaired or verified.'
    try:
        video, comparison, report = correct_framing(VideoFromFile(str(staged)), state, manifest, report, mask)
    finally:
        staged.unlink(missing_ok=True)
    source.update(output=str(video.get_stream_source()), output_frames=gi.frames,
                  compared_frames=common, fill_method='none; original sequence preserved')
    sidecar.write_text(json.dumps(source, indent=2))
    state.update(clip=str(video.get_stream_source()), side=str(comparison.get_stream_source()), report=report)
    return video, comparison, report, state


def correct_constant_shift(video, fill, manifest, report, mask=None):
    from comfy_api.input_impl import VideoFromFile
    measured=[d for d in manifest['drift'] if d and d.get('shift') is not None and d.get('sure') and not d.get('dt')]
    if len(measured)<max(8,len(manifest['ops'])*.5):
        raise ValueError('Constant-shift correction needs reliable timing and position on at least half the clip. Use Measure only.')
    offsets=np.array([[d['dx'],d['dy']] for d in measured])
    shift=np.median(offsets,axis=0)
    if np.percentile(np.linalg.norm(offsets-shift,axis=1),95)>2:
        raise ValueError('Position varies by more than 2 plate pixels. A constant correction would not lock this shot.')
    source=json.loads(Path(fill['sidecar']).read_text())
    path=Path(video.get_stream_source()); info=probe(path)
    # Analysis displacement is in plate pixels; rendering uses generated pixels.
    ratio=info.width/source['plate']['width']
    m=np.float32([[1,0,-shift[0]*ratio],[0,1,-shift[1]*ratio]])
    target=path.with_name(path.stem+'_position'+path.suffix)
    reader=FrameReader(path); writer=FrameWriter(target,info.width,info.height,info.fps,fill['codec'])
    try:
        while True:
            f=reader.read()
            if f is None: break
            writer.write(cv2.warpAffine(f,m,(info.width,info.height),borderMode=cv2.BORDER_CONSTANT))
    finally:
        reader.close(); writer.close()
    plate=decode_small_rgb(source['plate']['path']); gen=decode_small_rgb(source['generated']['path']); final=decode_small_rgb(target)
    ops=[Op(**o) for o in source['ops']]
    d=measure(plate,final,[True]*len(ops),source['plate']['width'],(mask.cpu().numpy()[0] if mask.ndim==3 else mask.cpu().numpy()) if mask is not None else None)
    side=target.with_name(target.stem+'_3up.mp4')
    compare_video(plate,gen,ops,source['summary'],side,info.fps,final=final)
    previous=json.loads((Path(fill['viewer'])/'manifest.json').read_text())
    export(Path(fill['viewer']),source['plate']['path'],source['generated']['path'],target,
           {**previous,'drift':d,'exactness':summary(d)},sources=False)
    return VideoFromFile(str(target)),VideoFromFile(str(side)),report+f'\nConstant translation corrected: {-shift[0]:+.2f}, {-shift[1]:+.2f} plate pixels. Exposed borders are black.'


def correct_framing(video, fill, manifest, report, mask=None):
    """Apply a validated single transform, then measure the decoded render again."""
    from comfy_api.input_impl import VideoFromFile
    from .framing import analyse_framing, correction_matrix, correspondences, motion_verdict
    source=json.loads(Path(fill['sidecar']).read_text())
    roi=(mask.cpu().numpy()[0] if mask.ndim==3 else mask.cpu().numpy()) if mask is not None else None
    plate=decode_small_rgb(source['plate']['path'],640)
    gen=decode_small_rgb(source['generated']['path'],640)
    ops=[Op(**o) for o in source['ops']]
    fit=analyse_framing(plate,gen,ops,source['plate']['width'],roi)
    path=Path(video.get_stream_source()); info=probe(path)
    matrix=correction_matrix(fit,info.width,info.height,source['plate']['width']/source['plate']['height'])
    target=path.with_name(path.stem+'_framing'+path.suffix)
    reader=FrameReader(path); writer=FrameWriter(target,info.width,info.height,info.fps,fill['codec'])
    try:
        while True:
            frame=reader.read()
            if frame is None: break
            writer.write(cv2.warpAffine(frame,matrix,(info.width,info.height),borderMode=cv2.BORDER_CONSTANT))
    finally:
        reader.close(); writer.close()
    final=decode_small_rgb(target,640)
    residuals=[];errors=[]
    for sample in fit['series']:
        i=sample['frame'];match=correspondences(plate[i],final[i],roi)
        if match is None: continue
        p,q,_,_,noise=match
        delta=np.median(q-p,axis=0)*source['plate']['width']
        residuals.append({'frame':i,'dx':float(delta[0]),'dy':float(delta[1]),'error_px':noise*source['plate']['width']})
        errors.append(float(np.median(np.linalg.norm(q-p,axis=1)))*source['plate']['width'])
    fit['rendered_motion']=motion_verdict([s['frame'] for s in residuals],[[s['dx'],s['dy']] for s in residuals],len(ops),
                                         float(np.median([s['error_px'] for s in residuals])) if residuals else 2)
    fit['rendered_error_px']=round(float(np.median(errors)),2) if errors else None
    fit['rendered_samples']=len(errors);fit['rendered_series']=residuals;fit['applied']=True
    d=measure(plate,final,[True]*len(ops),source['plate']['width'],roi)
    side=target.with_name(target.stem+'_3up.mp4')
    compare_video(plate,gen,ops,source['summary'],side,info.fps,final=final,result_label="OUTPUT")
    previous=json.loads((Path(fill['viewer'])/'manifest.json').read_text())
    export(Path(fill['viewer']),source['plate']['path'],source['generated']['path'],target,
           {**previous,'label':'OUTPUT','framing':fit,'drift':d,'exactness':summary(d)},sources=False)
    return VideoFromFile(str(target)),VideoFromFile(str(side)),report+f"\nFixed framing corrected. Held-out shared-feature error: {fit['before_px']} → {fit['after_px']} plate pixels. {fit['rendered_motion']['headline']}. Exposed borders are black."
