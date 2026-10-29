# PlateLock media

**Keep the change. Keep the shot.**

Align AI video edits to the original plate. Repair timing and framing mismatches, then composite only the intended change.

## Downloads

- [Demo video — 1080p](PlateLock-Demo-1080p.mp4)
- [English captions](PlateLock-Demo.en.srt)
- [Landscape poster — 1920 × 1080](platelock-poster-16x9.jpg)
- [Portrait poster — 1080 × 1920](platelock-poster-9x16.jpg)
- [Square poster — 1080 × 1080](platelock-poster-square.jpg)

The original generated PNGs are also included: [landscape](platelock-poster-16x9-source.png), [portrait](platelock-poster-9x16-source.png), [square](platelock-poster-square-source.png). JPEGs are resized delivery exports. Poster photography is generated marketing artwork, not a repair screenshot.

## What the demo shows

The car edits were generated at native 1920 × 1080 with Seedance 2.0. Their stock references were supplied to the model as 720p proxies; PlateLock uses the original 1080p plates. The demo is delivered at 30 fps; source and repair files retain their own frame rates before presentation conforming.

The garage comparison uses the actual exported PlateLock correction. Median shared-feature displacement on withheld frames falls from **12.4 to 1.9 pixels**, measured again after decoding the output. This measures shared detail, not every pixel. The correction is a single fixed scale and translation; some residual movement remains.

The garage finishing example uses native **LTX 2.5 Alpha Gen on Windows** to mask the old and replacement cars. Their masks are combined, holes filled, and edges and the shadow/reflection region refined before linear-light compositing. Protected source pixels are copied exactly before delivery compression. This is an external finishing experiment, not an automatic change-mask feature in PlateLock.

The road reveal is an additional AI replacement with the actual PlateLock framing output. It is not a selective composite: the Alpha Gen mask lost the receding car, so that mask result was rejected. Progressive residual movement remains in this shot.

The train comparison is explicitly a **controlled timing test**: nine known missing frames and two repeats were introduced into licensed stock. Actual PlateLock/RIFE repair copies 231 frames and reconstructs nine. Copied-frame correspondence was checked against the known source map. Reconstructed frames are estimates.

The app footage is a real ComfyUI capture. Execution waits are omitted. Optional captions are not burned into the picture.

[Generation recipes](generation-recipes.json) · [Verification record](verification.json)

## Sources and credits

- Garage plate: [Mixkit #66](https://mixkit.co/free-stock-video/parked-red-sports-car-66/), under the [Mixkit Stock Video Free License](https://mixkit.co/license/#videoFree).
- Road plate: [Mixkit #52427](https://mixkit.co/free-stock-video/a-red-sports-car-traveling-along-a-curvy-asphalt-road-52427/), under the same stock video license.
- Train plate: [smile / Pexels #5593088](https://www.pexels.com/video/a-train-passing-by-the-railway-station-5593088/), under the [Pexels License](https://www.pexels.com/license/).
- Car replacements: Seedance 2.0 through fal.ai, endpoint `bytedance/seedance-2.0/us/reference-to-video`.
- Garage mattes: [LTX 2.5 Alpha Gen](https://docs.ltx.io/open-source-model/vfx-post-production/alpha-gen), generated locally in Windows ComfyUI.
- Poster artwork and format adaptations: OpenAI built-in image generation; prompts included in [poster-prompts.txt](poster-prompts.txt).
- Synthetic narration: ElevenLabs / Chris through fal.ai, endpoint `elevenlabs/tts/eleven-v4`.
- Ambient bed: original synthesized composition; no sampled recording.

No client footage is included. The repository's MIT license applies to application code; it does not relicense stock footage, model weights or third-party media. Provider terms apply to generated assets. This repository publication does not submit the project to the competition.
