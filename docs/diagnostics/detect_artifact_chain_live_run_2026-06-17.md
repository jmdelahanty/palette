# Detect Artifact Chain Live Run 2026-06-17

Status: in progress.

Purpose: capture the first registry-scoped GoodCopBadCop run through the
scratch-artifact detect -> import -> validate -> detect-quality -> refined-detect
chain without prematurely marking the workflow as fully promoted.

## Run

- Branch/commit on cluster deployment: `sun` at `229bb8b`.
- Submitter: `scripts/submit_detect_artifact_quality_refine_bsub.sh`.
- Run id: `goodcopbadcop_detect_artifact_refine_20260617T025729Z`.
- Run directory:
  `/groups/johnson/johnsonlab/jeremy/recordings/logs/detect_artifact_quality_refine_bsub/detect_artifact_quality_refine_goodcopbadcop_detect_artifact_refine_20260617T025729Z`.
- Target count: 12 GoodCopBadCop recordings.
- Registry: PRFS-visible snapshot under
  `/groups/johnson/johnsonlab/jeremy/registries/`.
- Model resolution: registry-selected detect model, after model artifact paths
  were mirrored from `/nvme1/models` to `/groups/johnson/johnsonlab/jeremy/models`.

## Verified So Far

- The LSF environment smoke on `gpu_l4` completed successfully with CUDA,
  PyTorch, Decord GPU decode, and `PyNvVideoCodec` import available.
- The submitted detect jobs are using the intended backend and input size:
  `pynvvc_nv12_rgb`, `resize_dims=640x640`, `batch_size=16`.
- Live logs showed L4 throughput around near realtime after warmup instead of
  the bad full-frame tensor path.
- Detect jobs write to node-local scratch first:
  `/scratch/$USER/$LSB_JOBID/work/detect_output.zarr`.
- Dependent CPU postprocess jobs are submitted with LSF dependencies and should
  import the packaged detect run, validate it, run detect quality, and run
  refinement with explicit run names.

## Pending Audit

- Confirm all 12 GPU artifact jobs completed with `DONE`.
- Confirm all 12 dependent postprocess jobs completed with `DONE`.
- Inspect non-empty `.err` files for real errors versus normal progress output.
- Count imported `detect_runs`, nested quality reports, and `refined_detect_runs`
  in the target analysis Zarrs.
- Verify each imported detect run has the expected run-completion marker,
  provenance, `latest_policy=set_latest_explicit`, decode backend, model path,
  and resize dimensions.
- Verify detect quality and refinement used the deterministic detect and quality
  run names from the submission manifest rather than mutable `latest`.
- Refresh or validate registry projections after successful imports.

## Documentation Outcome

The general operator docs now require a PRFS-visible registry snapshot for LSF
submission. The previous examples using `/nvme1/palette_registry.sqlite` were
valid only on the workstation and were stale for login/compute-node submission.
