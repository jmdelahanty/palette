# Testing TODO

Focus areas remaining after schema correctness tests.

## Refinement selection logic

- Current curated detect read order: `refined_detect_runs/<latest>/instances`
  → legacy sparse refined fallback
  (`manual_review_latest`, `manual`, `interpolated`, `filtered`) → raw detect.
- Manual override: `manual_review_latest` should still select manual when the
  resolver is operating on legacy sparse runs.
- Review status override: `detect_review_status.resolved_group` should win if
  set for legacy subgroup fallback.
- Mixed availability: ensure resolution is stable when `instances/` or legacy
  sparse groups are missing.

## Detect Review

- Add more arena-aware `detect_review` coverage around the interactive refined
  path:
  - multi-arena targeting with `--frames` should review all slots for the
    selected frames
  - manual box rejection should be covered when the drawn box center falls
    outside the active arena ROI
  - status-only approval should be exercised for arena-aware refined runs
- Add a focused non-interactive regression for `review_axis="frame_arena"` save
  metadata so `manual_review_slots` stays consistent.
- Retune still follows the older single-instance refined path; add explicit
  tests or defer until arena-aware retune is implemented.

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

## Codex Sandbox AppArmor Restriction

- On Ubuntu 24.04 workstations, Codex sandbox startup may fail before the requested command runs if `bubblewrap` cannot create the expected user namespace.
- Observed 2026-03-31 symptom outside the repo:
  - `bwrap: setting up uid map: Permission denied`
  - `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`
- The confirmed machine-level cause on `delahantyj-ws1` was `kernel.apparmor_restrict_unprivileged_userns=1`.
- A temporary diagnostic override:
  - `sudo sysctl kernel.apparmor_restrict_unprivileged_userns=0`
  restored `unshare -Ur ...`, `bwrap ...`, and normal Codex sandbox command execution.
- This is not a Palette code regression. It is a host AppArmor/user-namespace policy issue that can surface as unexplained Codex sandbox failures during otherwise ordinary repo work.
- Preferred workstation fix:
  - keep the global AppArmor restriction enabled when possible,
  - add a targeted AppArmor allowance for the specific Codex/bubblewrap launcher that needs `userns`,
  - use the global sysctl override only as a temporary diagnostic or personal-workstation fallback.
