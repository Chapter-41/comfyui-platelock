# PlateLock for ComfyUI

Bring an AI video back into alignment with its source plate. PlateLock compares frame correspondence and shared image detail, explains what it can measure, and repairs supported timing and framing problems.

**Plate → As delivered → Output.** Start with the shot verdict, inspect the evidence, then choose what to fix. PlateLock does not promise pixel identity or automatically repair local deformation.

## Install

Use an existing ComfyUI installation with native **Load Video**, **Save Video** and frame interpolation nodes. FFmpeg and ffprobe must be available on the PATH used to launch ComfyUI.

From your ComfyUI root:

```sh
git clone https://github.com/Chapter-41/comfyui-platelock.git custom_nodes/comfyui-platelock
# Use ComfyUI's Python, not an unrelated system environment:
.venv/bin/python -m pip install -r custom_nodes/comfyui-platelock/requirements.txt
.venv/bin/python custom_nodes/comfyui-platelock/tools/check_install.py --comfy-root .
```

For Windows portable, use its `python_embeded/python.exe` instead of `.venv/bin/python` and adjust paths to your ComfyUI directory. Restart ComfyUI after installation; reload the browser after frontend updates. The nodes appear under the PlateLock category. No external service or API key is required.

Analysis and fixed framing correction need no model weights. For skipped-frame repair, install a native-compatible RIFE checkpoint in `ComfyUI/models/frame_interpolation`. The tested checkpoint is `rife_v4.26.safetensors`; alternatively connect an interpolation model. Model weights are not included or automatically downloaded.

## Try the portable examples

From the pack directory, using ComfyUI's Python:

```sh
../../.venv/bin/python tools/make_demo.py --output ../../input
```

Load one of these workflows in ComfyUI and click Run:

| Workflow | What to expect |
| --- | --- |
| `example_workflows/demo_framing.json` | A moving, recolored clip with a known fixed scale/position error. Analyse, prepare **Correct framing**, run Repair, inspect Overlay. |
| `example_workflows/demo_timing.json` | Frames 20 and 40 were removed. Analyse, then **Repair timing** with RIFE. |
| `example_workflows/demo_unknown.json` | No shared picture detail. The result must say **Insufficient evidence**, not “clean.” |

These fixtures are generated procedurally, covered by the repository's MIT license, and contain no client footage or third-party imagery. They demonstrate controlled defects, not general accuracy on AI footage. No large media download is required.

## Read the result

**Open Review** shows a shot verdict and a next action. Detailed measurements are collapsed underneath it. The residual movement graph lets you jump to measured samples. Use shared zoom/pan, Wipe, Flicker, Overlay and Edges to inspect the picture.

- **Fixed framing mismatch:** one constant scale and translation improves shared-feature alignment on withheld frames.
- **Fixed framing corrected:** that transform was applied. Inspect remaining differences; this is not a claim that the shot is perfect.
- **Possible progressive drift / wobble:** movement remains after the fixed transform. Review the trend; no automatic per-frame stabilisation is performed.
- **Insufficient evidence:** the shot cannot be judged reliably. Select a mask over shared subject detail and analyse again.

Frame numbers are zero-based. Measurements use original plate pixels. Unsampled frames are not counted as passing a spatial check. Appearance-difference boxes can show intended restyling, not just errors.

**Prepare recommended repair** changes the node settings; it does not start rendering. You can inspect the settings before clicking Run.

## Choose the repair task

| Task | Behaviour |
| --- | --- |
| Correct framing | One validated fixed scale/position correction using same-time shared detail. Preserves every AI frame and its frame rate; leaves timing unresolved. Requires matching input frame rates. |
| Repair timing | Aligns frame correspondence; uses RIFE only for short bounded gaps. |
| Repair timing + framing | Aligns timing, fills supported gaps and applies a validated fixed framing correction. |
| Reconstruct missing footage (experimental) | Local LTX reconstruction for unbounded missing footage; RIFE for short gaps. End-to-end LTX quality is not yet validated. |
| Advanced settings | Preserves older workflows and exposes backend, position settings and the ambiguous-timing override. |

**Analyse** never loads a generative model or produces an Output video. **Repair** exposes `output` and `comparison_3up`; connect these to Save Video. Existing output slot ordering is preserved. Weak sequence evidence blocks timing edits unless explicitly overridden in Advanced settings. Missing footage is never silently repaired with held or black frames by Studio.

## Limits that matter

- No explicit letterbox/active-picture detection yet. Mixed crops and bars need inspection; do not assume the framing verdict is reliable for these inputs.
- Fixed framing correction cannot fix local shape changes, perspective changes or independently drifting objects. Exposed image borders are black.
- Current rendering uses 8-bit RGB. A ProRes container does **not** preserve the source's original bit depth, color management, timecode or metadata.
- The node does not preserve audio. Remux audio separately when appropriate. This is a review/correction candidate, not an untouched finishing master.
- A steady pan can make temporal and spatial displacement ambiguous. Changed backgrounds and occlusion can leave too little shared detail. A mask limits evidence to the selected area.
- Timing checks search nearby frames; thresholds are tested heuristics, not calibrated probabilities. Spatial drift uses five temporal windows and a threshold tied to feature-fit noise. “No clear drift” does not rule out subtle or local drift.
- On Apple MPS, RIFE runs on an isolated CPU model due to unsupported border sampling; large repairs can be slow.

## Experimental LTX

Framing correction does not need LTX. The public demo now includes a separately executed native LTX continuation of a real AI take: 23 additional frames for a longer edit. That advanced workflow is documented in the [demo evidence](media/real-ai-evidence.json). Studio’s automatic reconstruction integration still has contract tests rather than a validated full execution. It expects these local checkpoints:

- `diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors`
- `vae/ltx-2.5-video-vae-bf16.safetensors`
- `vae/ltx-2.5-audio-vae-bf16.safetensors`
- `text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors`

The older Fill Window / Fill Merge nodes remain available for advanced workflows.

## Validation and contribution

```sh
python -m unittest discover -s tests -q
node --test tests/review.test.mjs
python tools/make_demo.py --output demo/inputs
python tools/verify_demo.py --inputs demo/inputs --report demo/verification.json
```

[Validation notes](validation.md) separate controlled tests from real-footage evidence and untested cases. [Demo guide](demo-guide.md) describes a short, reproducible presentation. Code and procedural fixtures are MIT licensed; model weights retain their own licenses.

For a real-model integration check, start a separate local ComfyUI instance with RIFE installed, generate the fixtures in its input folder, then run:

```sh
python tools/run_comfy_demo.py --url http://127.0.0.1:8189
```

This queues five executions on that server, including actual RIFE repair. Use an idle test instance; no LTX run is included.
