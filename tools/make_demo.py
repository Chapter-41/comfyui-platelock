"""Generate the portable demo clips into ComfyUI/input or another chosen directory."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from platelock.demo import write_examples
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
print(write_examples(args.output).resolve())
