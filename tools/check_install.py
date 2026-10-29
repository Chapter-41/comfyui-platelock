"""Check local prerequisites without loading models or changing the installation."""
import argparse
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--comfy-root',type=Path)
args=parser.parse_args()
checks={name:bool(shutil.which(name)) for name in ('ffmpeg','ffprobe')}
versions={'python':sys.version.split()[0]}
for package in ('numpy','opencv-python-headless','torch'):
    try: versions[package]=importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError: versions[package]=None
try:
    import cv2
    checks['opencv_sift']=hasattr(cv2,'SIFT_create')
    versions['opencv_runtime']=cv2.__version__
    import numpy
    checks['numpy']=True
except ImportError as e:
    checks['python_dependencies']=False
    versions['import_error']=str(e)
if args.comfy_root:
    root=args.comfy_root.resolve()
    checks['comfy_main']=(root/'main.py').is_file()
    checks['native_video_nodes']=(root/'comfy_extras/nodes_video.py').is_file()
    checks['native_interpolation_nodes']=(root/'comfy_extras/nodes_frame_interpolation.py').is_file()
    checks['torch']=versions['torch'] is not None
    versions['rife_checkpoints']=[p.name for p in (root/'models/frame_interpolation').glob('*') if 'rife' in p.name.lower()]
print(json.dumps({'ok':all(checks.values()),'checks':checks,'versions':versions,
                  'note':'RIFE checkpoints are needed only for gap repair. LTX is experimental and not checked here.'},indent=2))
raise SystemExit(0 if all(checks.values()) else 1)
