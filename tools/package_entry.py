"""Bundle source, workflows and procedural demo inputs; exclude client media and environments."""
import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
files=[root/name for name in ('README.md','LICENSE','requirements.txt','pyproject.toml','__init__.py','nodes.py')]
for folder,pattern in [('platelock','*.py'),('web','*.js'),('web','*.mjs'),('web','*.css'),
                       ('tests','*.py'),('tests','*.mjs'),('tools','*.py'),('docs','*.md'),('example_workflows','*.json')]:
    files.extend((root/folder).glob(pattern))
files.extend((root/'demo/inputs').glob('platelock_demo_*.mp4'))
truth=root/'demo/inputs/truth.json'
if truth.is_file():files.append(truth)
a.output.parent.mkdir(parents=True,exist_ok=True)
with ZipFile(a.output,'w',ZIP_DEFLATED) as archive:
    for path in sorted(set(files)):
        archive.write(path,Path('comfyui-platelock')/path.relative_to(root))
with ZipFile(a.output) as archive:
    assert archive.testzip() is None
    print(f'{a.output.resolve()} ({len(archive.namelist())} files, {a.output.stat().st_size/1024:.0f} KiB)')
