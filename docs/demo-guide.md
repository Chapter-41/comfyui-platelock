# Demo guide

The [current demo](media/PlateLock-Demo-1080p.mp4) is 58 seconds at 1920 × 1080. It uses full-screen footage, A/B comparisons, brief app captures and one closing poster.

| Time | Picture |
| --- | --- |
| 0–8 s | Original car → AI replacement → actual framing repair |
| 8–13 s | Magnified framing A/B with source overlay |
| 13–19 s | Load, repair and review in ComfyUI |
| 19–27 s | LTX matte and selective-compositing reveal |
| 27–32 s | Second AI car replacement on a moving road shot |
| 32–48 s | Timing controls and actual PlateLock/RIFE A/B |
| 48–58 s | Finished garage shot and poster |

The timing test is labeled as controlled. The framing correction is fixed, and residual drift remains. LTX masks and compositing are a separate finishing workflow. [Sources, methods and credits](media/README.md) document these distinctions.

To reproduce the app's portable controlled examples, generate the fixtures with `tools/make_demo.py`, then load an example workflow. Use `tools/run_comfy_demo.py` against an idle ComfyUI test instance to check native repair execution. Detailed commands are in the [usage guide](usage.md).

`tools/package_entry.py` creates a local source bundle with the procedural fixtures. It does not submit an entry. The published media and repository are ready for review; competition submission remains a separate action.
