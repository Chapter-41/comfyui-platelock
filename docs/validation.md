# Validation scope

This document describes evidence, not a general accuracy or production-readiness claim.

## Tested environment

Observed on 2026-10-28: macOS, Python 3.13.12, ComfyUI 0.27.1, commit `c2638ce6`, NumPy 2.4.4, OpenCV 5.0.0 (`opencv-python-headless` 5.0.0.93), PyTorch 2.10.0. RIFE checkpoint: `rife_v4.26.safetensors`. Analysis also runs in the repository's separate Python environment. Run `tools/check_install.py` to record your environment.

A fresh Python 3.12.9 environment installed the declared requirements (NumPy 2.5.4 and OpenCV 5.0.0.93) and passed all three decoded-video fixture checks. This verifies the standalone dependencies on this Mac, not a fresh ComfyUI installation on every OS.

The package minimums express dependency compatibility intent, not proof of every supported version. Native ComfyUI node APIs must be present. Windows/Linux and a fresh machine still need independent installation verification.

## Controlled evidence

`tools/make_demo.py` creates moving procedural footage and three known-defect cases. `tools/verify_demo.py` decodes the actual MP4s and checks:

- Fixed scale 0.975 and translation (+7,-3) pixels, with changed color: alignment improves on frames withheld from fitting.
- Removed frames 20 and 40: both gaps are found and copied frames retain correct correspondence.
- Flat unrelated input: spatial fitting refuses insufficient shared detail.

Tests additionally cover camera motion, independently moving detail, progressive translation that remains after fixed correction, unrelated moving imagery, natural pauses, offsets, repeats, missing tails, weak evidence, and backward-compatible repair settings. Some node tests stub interpolation to check orchestration; those are not model-quality tests.

The original repeated-image fit test remains a unit test only. The moving-scene tests provide broader coverage, but synthetic success does not establish real-world accuracy.

## Real-footage evidence

The public garage example uses licensed Mixkit stock and a native 1080p Seedance car replacement. Actual PlateLock framing repair reduces shared-feature displacement on withheld frames from 12.41 to 1.93 plate pixels after decoding the rendered output. All 193 AI frames are retained. A residual movement trend remains; this is fixed framing correction, not general drift removal.

The public timing example introduces nine isolated gaps and two repeats into licensed Pexels stock. Native RIFE reconstructs nine frames; 231 copied-frame correspondences match the known source map. This is a controlled test, not a naturally occurring AI timing failure.

The road replacement still has progressive residual movement after its fixed correction. A coffee replacement was rejected for insufficient shared detail. These are useful limits, not passing examples of arbitrary deformation repair. See [demo evidence and sources](media/README.md).

## Known unverified areas

Explicit letterbox handling; color-managed high-bit-depth finishing; automatic audio/metadata preservation; local deformation repair; calibrated error probabilities; broad testing across extreme restyling, rolling shutter, motion blur and occlusion; end-to-end local LTX generation quality.

## Native ComfyUI execution

On the isolated local ComfyUI instance, all five executions passed: framing Analyse, framing Repair, timing Analyse, timing Repair and unknown Analyse. Timing Repair loaded the real native RIFE IFNet and reconstructed both missing frames; it did not use the interpolation stub. The framing output was decoded and measured again. `tools/run_comfy_demo.py` reproduces these checks through ComfyUI's local API after the fixtures are installed.

On 2026-10-29, all 43 Python tests and nine JavaScript review-logic tests passed. The expanded review was inspected against actual decoded outputs. Native Windows LTX Alpha Gen also produced the garage mattes used in the separate demo finishing workflow; this does not validate PlateLock’s experimental LTX missing-footage reconstruction.
