"""Verify the three portable examples on decoded video, without ComfyUI or models."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from platelock.core import analyze_arrays
from platelock.framing import analyse_framing
from platelock.plan import Op
from platelock.video import decode_small_rgb, decode_gray

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--inputs',type=Path,default=Path('demo/inputs'))
p.add_argument('--report',type=Path,default=Path('demo/verification.json'))
a=p.parse_args();root=a.inputs
plate=decode_small_rgb(root/'platelock_demo_plate.mp4',320)
fixed=decode_small_rgb(root/'platelock_demo_framing.mp4',320)
fit=analyse_framing(plate,fixed,[Op(i,'copy',i) for i in range(len(plate))],320,max_samples=30)
assert fit['after_px']<fit['before_px']*.25 and fit['motion']['state']=='stable',fit
fixed_plan=analyze_arrays(decode_gray(root/'platelock_demo_framing.mp4'),decode_gray(root/'platelock_demo_plate.mp4'))
assert all(o.op=='copy' and o.plate==o.gen for o in fixed_plan.ops),fixed_plan.summary
truth=json.loads((root/'truth.json').read_text())
r=analyze_arrays(decode_gray(root/'platelock_demo_timing.mp4'),decode_gray(root/'platelock_demo_plate.mp4'))
wrong=[o.plate for o in r.ops if o.op=='copy' and truth['timing']['source_indices'][o.gen]!=o.plate]
assert not wrong,wrong
assert set(truth['timing']['missing'])<=set(r.summary['interpolated']),r.summary
try:
    analyse_framing(plate,decode_small_rgb(root/'platelock_demo_unknown.mp4',320),[Op(i,'copy',i) for i in range(len(plate))],320,max_samples=24)
except ValueError as exc: refusal=str(exc)
else: raise AssertionError('Unmeasurable input was accepted')
report={'fixture':'Procedural controlled defects, not a real-world accuracy benchmark',
        'framing':{k:v for k,v in fit.items() if k!='series'},
        'timing':{'wrong_copy_frames':wrong,'gap_frames':r.summary['interpolated'],'expected_gaps':truth['timing']['missing']},
        'unknown':{'refused':True,'reason':refusal}}
a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
