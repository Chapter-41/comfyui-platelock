# Demo guide

The [current demo](https://chapter-41.github.io/comfyui-platelock/) is 65 seconds at 1920 × 1080. It uses full-screen footage, A/B comparisons, brief app captures and one closing poster, with recorded narration and a face-cam overlay.

| Time | Picture |
| --- | --- |
| 0–8 s | Original car → AI replacement → actual framing repair |
| 8–13 s | Magnified framing A/B with source overlay |
| 13–19 s | Load, repair and review in ComfyUI |
| 19–27 s | LTX matte and selective-compositing reveal |
| 27–32 s | Second AI car replacement on a moving road shot |
| 32–36 s | Actual ComfyUI repair of a real AI timing mismatch |
| 36–46 s | Real Seedance take versus PlateLock/RIFE, including a slowed detail |
| 46–55 s | Complete AI take extended from 193 to 216 frames with LTX |
| 55–65 s | Finished garage shot and poster |

The public demo contains no deliberately damaged-stock timing test. The timing example is a contiguous real AI excerpt; its one-frame mismatch also appears in the full untouched take. The extension adds footage because the edit needs a longer shot; no existing AI tail was cut off. Fixed framing can leave residual movement. LTX masks/compositing and tail continuation use separate advanced workflows. [Sources, methods and credits](media/README.md) document these distinctions.

To reproduce the app's portable controlled examples, generate the fixtures with `tools/make_demo.py`, then load an example workflow. Use `tools/run_comfy_demo.py` against an idle ComfyUI test instance to check native repair execution. Detailed commands are in the [usage guide](usage.md).

`tools/package_entry.py` creates a local source bundle with the procedural fixtures. It does not submit an entry. The published media and repository are ready for review; competition submission remains a separate action.
