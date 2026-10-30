# PlateLock media

**Keep the change. Keep the shot.**

Align AI video edits to the original plate. Repair timing and framing mismatches, then composite only the intended change.

## Downloads

- [Final demo with face cam and recorded voice — 1080p](PlateLock-Demo-Final-1080p.mp4)
- [Final English captions](PlateLock-Demo-Final.en.srt)
- [Earlier demo with synthetic narration — 1080p](PlateLock-Demo-1080p.mp4)
- [Composite refinement — previous / refined](PlateLock-Composite-Refinement.mp4)
- [Earlier English captions](PlateLock-Demo.en.srt)
- [Landscape poster — 1920 × 1080](platelock-poster-16x9.jpg)
- [Portrait poster — 1080 × 1920](platelock-poster-9x16.jpg)
- [Square poster — 1080 × 1080](platelock-poster-square.jpg)

The original generated PNGs are also included: [landscape](platelock-poster-16x9-source.png), [portrait](platelock-poster-9x16-source.png), [square](platelock-poster-square-source.png). JPEGs are resized delivery exports. Poster photography is generated marketing artwork, not a repair screenshot.

The final 65-second cut uses the latest approved webcam and wireless-microphone take, with a circular face cam in the upper-right corner. The countdown is removed, voice loudness is leveled, and the timing-repair explanation is aligned to its demonstration by moving a quiet interval together with the face-cam footage. Captions follow the recorded delivery. The original take remains in local production files. [Final export verification](final-video-verification.json)

## What the demo shows

The car edits were generated at native 1920 × 1080 with Seedance 2.0. Their stock references were supplied to the model as 720p proxies; PlateLock uses the original 1080p plates. The demo is delivered at 30 fps; source and repair files retain their own frame rates before presentation conforming.

The garage comparison uses the actual exported PlateLock correction. Median shared-feature displacement on withheld frames falls from **12.4 to 1.9 pixels**, measured again after decoding the output. This measures shared detail, not every pixel. The correction is a single fixed scale and translation; some residual movement remains.

The garage finishing example uses native **LTX 2.5 Alpha Gen on Windows** to mask the old and replacement cars. The car matte is tightened, with small contact-shadow and reflection patches. A local background-only colour match removes much of the warm spill introduced by the AI edit before linear-light compositing; the new car retains its generated colour. Protected source pixels are copied exactly before delivery compression. This is an external finishing experiment, not an automatic change-mask feature in PlateLock.

The road reveal is an additional AI replacement with the actual PlateLock framing output. It is not a selective composite: the Alpha Gen mask lost the receding car, so that mask result was rejected. Progressive residual movement remains in this shot.

The timing comparison uses a **real 1080p Seedance output**, not damaged stock. A contiguous six-second excerpt contains a frame-correspondence mismatch also found at plate frame 96 in the full unmodified take. No internal frames were manually removed or repeated. Native PlateLock/RIFE preserves the order of all 143 excerpt frames, reconstructs one frame, and applies one fixed framing correction. Shared-feature displacement on 40 withheld frames falls from **12.93 to 1.25 pixels** after decoding. Later ambiguous portions of the full take are excluded; this is not independent proof that Seedance literally deleted a source frame.

The LTX continuation uses the **complete 193-frame AI take**. A nine-second edit needs 216 frames, so native LTX 2.5 generates the extra 23 frames using the last 17 original frames as context. No AI tail was artificially removed. The output uses the original decoded pixels for all 193 existing frames before delivery encoding. This is a separate advanced workflow using PlateLock’s patch/colour-match helpers, not validation of Studio’s automatic missing-footage execution. It generates a plausible continuation, not the exact original plate trajectory. [Evidence](real-ai-evidence.json) · [LTX API workflow](ltx-tail-workflow-api.json)

The app footage is a real ComfyUI capture. Execution waits are omitted. Optional captions are not burned into the picture.

[Generation recipes](generation-recipes.json) · [Verification record](verification.json)

## Sources and credits

- Garage plate: [Mixkit #66](https://mixkit.co/free-stock-video/parked-red-sports-car-66/), under the [Mixkit Stock Video Free License](https://mixkit.co/license/#videoFree).
- Road plate: [Mixkit #52427](https://mixkit.co/free-stock-video/a-red-sports-car-traveling-along-a-curvy-asphalt-road-52427/), under the same stock video license.
- Car replacements: Seedance 2.0 through fal.ai, endpoint `bytedance/seedance-2.0/us/reference-to-video`.
- Garage continuation: native Windows ComfyUI, LTX 2.5 distilled, seed 41; 1920 × 1088 generation with padding cropped to 1080p.
- Garage mattes: [LTX 2.5 Alpha Gen](https://docs.ltx.io/open-source-model/vfx-post-production/alpha-gen), generated locally in Windows ComfyUI.
- Poster artwork and format adaptations: OpenAI built-in image generation; prompts included in [poster-prompts.txt](poster-prompts.txt).
- Final narration and face cam: user-recorded performance, latest approved take from 30 October 2026; no synthetic voice in the final export.
- Earlier synthetic narration: ElevenLabs / Chris through fal.ai, endpoint `elevenlabs/tts/eleven-v4`.
- Ambient bed: original synthesized composition; no sampled recording.

No client footage is included. The repository's MIT license applies to application code; it does not relicense stock footage, model weights or third-party media. Provider terms apply to generated assets. This repository publication does not submit the project to the competition.
