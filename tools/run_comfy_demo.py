"""Exercise real ComfyUI execution, including native RIFE, on the portable examples.

Generate the fixtures into that server's input folder first. Does not download models.
"""
import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--url',default='http://127.0.0.1:8189')
p.add_argument('--report',type=Path,default=Path('demo/comfy-verification.json'))
a=p.parse_args()

def request(path,payload=None):
    req=Request(a.url+path,data=json.dumps(payload).encode() if payload is not None else None,
                headers={'Content-Type':'application/json'})
    with urlopen(req,timeout=30) as response:return json.load(response)

results=[]
for case,mode,task in [('framing','Analyse','Correct framing'),('framing','Repair','Correct framing'),
                       ('timing','Analyse','Repair timing'),('timing','Repair','Repair timing'),
                       ('unknown','Analyse','Advanced settings')]:
    prompt={
        '1':{'class_type':'LoadVideo','inputs':{'file':'platelock_demo_plate.mp4'}},
        '2':{'class_type':'LoadVideo','inputs':{'file':f'platelock_demo_{case}.mp4'}},
        '3':{'class_type':'PlateLockStudio','inputs':{'plate':['1',0],'generated':['2',0],
             'mode':mode,'method':'RIFE','repair_task':task}},
    }
    started=time.monotonic();job=request('/prompt',{'prompt':prompt})
    if job.get('node_errors'):raise RuntimeError(job['node_errors'])
    key=job['prompt_id'];result=None
    while time.monotonic()-started<600:
        history=request('/history/'+key)
        if key in history:result=history[key];break
        time.sleep(.5)
    if not result or result['status']['status_str']!='success':raise RuntimeError(result or 'Timed out')
    viewer=result['outputs']['3']['platelock_viewer'][0]['subfolder']
    from urllib.parse import urlencode
    manifest=request('/view?'+urlencode({'type':'output','subfolder':viewer,'filename':'manifest.json'}))
    if case=='unknown':assert manifest['framing']['motion']['state']=='unknown'
    if case=='framing' and mode=='Repair':
        f=manifest['framing'];assert f['applied'] and f['rendered_error_px']<f['before_px']*.25
    if case=='timing':assert set(manifest['summary']['interpolated'])=={20,40}
    record={'case':case,'mode':mode,'task':task,'seconds':round(time.monotonic()-started,2),'viewer':viewer,
            'summary':manifest.get('summary'),'framing':{k:v for k,v in manifest.get('framing',{}).items() if k not in ('series','rendered_series')}}
    results.append(record)
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(results,indent=2))
    print(f'{case} / {mode}: PASS ({record["seconds"]}s)',flush=True)
