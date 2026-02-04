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
