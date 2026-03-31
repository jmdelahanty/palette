# Testing TODO

Focus areas remaining after schema correctness tests.

## Refinement selection logic

- Preferred resolution order: detect → filtered → interpolated → manual.
- Manual override: `manual_review_latest` should select manual when present.
- Review status override: `detect_review_status.resolved_group` should win if set.
- Mixed availability: ensure resolution is stable when some groups are missing.

## Coverage accounting

- Full imports: coverage percent based on total frames.
- Sampled imports: coverage percent based on sampled universe; store `coverage_frames_full`.
- Passthrough refine: no filtering/interpolation labels when no action taken.
- Training datasets: verify coverage reports use sampled indices when present.

## Pytest Decord Import Instability

- Some pytest collectors that import `fisheye.core.pipeline` transitively import `capture/import_video.py`, which imports `decord` at module import time.
- On some workstations, pytest then fails during collection with an FFmpeg shared-library mismatch even though the pipeline may otherwise run normally in the shell.
- Observed 2026-03-30 symptom:
  - `OSError: /opt/orange/lib/ffmpeg-nvidia/lib/libavfilter.so.7: undefined symbol: av_gcd_q, version LIBAVUTIL_56`
- This is easy to misread as a regression in unrelated pipeline-adjacent tests, including refined-subject stage tests. It is an environment/import-chain problem, not necessarily a functional regression in the code under test.
- Short-term testing guidance:
  - Prefer narrower pytest subsets when validating code that does not need `fisheye.core.pipeline`.
  - If a change is isolated to a non-pipeline module, do not block on pipeline collector failures caused by `decord` import-time linkage.
  - Record the exact command that passed and call out the skipped pipeline collector explicitly in handoff notes.
- Medium-term fix candidates:
  - Lazy-import `decord` inside `capture/import_video.py` call sites instead of at module import time.
  - Reduce top-level imports in `fisheye.core.pipeline` so non-import-stage tests do not pull in the video decode stack during collection.
  - Add a stable test seam or monkeypatch strategy for pipeline tests that should not depend on live decode libraries.
