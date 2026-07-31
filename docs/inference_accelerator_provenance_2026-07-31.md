# Inference accelerator provenance — 2026-07-31

## Decision

Every maintained inference writer must persist one versioned accelerator
snapshot. The shared document is `palette.accelerator_runtime` version 1 and is
captured by `get_environment_info()` exactly once per writer invocation.

The same document is available at `env_info["gpu"]` and is embedded in the
environment document that stage writers already persist:

```text
provenance.environment.accelerator
```

When a final `run_provenance` is derived from that stage record, the already
captured document is reused at:

```text
run_provenance.system.gpu
```

The derivation must not probe the device a second time. This keeps stage and
run provenance identical even for a short-lived worker or changing scheduler
allocation.

## Required contents

- capture timestamp and capture semantics;
- accelerator availability and backend;
- `CUDA_VISIBLE_DEVICES`/LSF allocation values when present;
- driver version when `nvidia-smi` is available;
- PyTorch, CUDA, and cuDNN runtime versions;
- cuDNN benchmark/deterministic flags and float32/TF32 policy;
- each visible device's index, name, UUID when exposed by PyTorch, compute
  capability, memory, and multiprocessor count;
- best-effort point-in-time NVML telemetry when that library is available.

Transient utilization, temperature, clocks, and power are runtime diagnostics.
They must not enter scientific content identity or array digests.

## Maintained producer census

| Producer | Persisted owner | Accelerator path after this change |
|---|---|---|
| YOLO canonical detections | `detect_runs` or detection artifact run | stage environment and run system |
| YOLO keypoints | `keypoints_runs` or raw keypoint artifact | stage environment and run system |
| U-Net subject masks | `subject_mask_runs` or shard run | stage environment and derived run system |
| SAM subject masks | subject-mask run | stage environment and run system |
| Training-Zarr detection prediction | detection artifact run | stage environment and derived run system |
| Traditional detection/keypoint/segmentation | existing stage run | shared environment, including unavailable accelerator when CPU-only |
| Model-backed analysis classifiers | existing stage run | shared environment accelerator |

Pure prediction helpers that return in-memory values do not own persisted
provenance. Their enclosing writer is responsible for the stage/run record.

## Publication and registry boundary

Detection, keypoint, and direct SAM writers persist their inference run. U-Net
subject-mask inference may first produce a temporary shard, so the final raw
publication receipt must bind both the inference stage provenance and the
inference run provenance before node-local scratch is removed.

The performance registry currently projects rates and selected source fields,
not the complete accelerator document. The persisted Zarr provenance remains
authoritative; adding registry projection is a query optimization, not a
capture prerequisite.

## Compatibility

Historical runs remain valid without the accelerator document. New inference
runs are expected to carry it. This change does not rewrite historical archives
and does not alter scientific array identity.

The subject-mask reference job `153236891` is bound to commit `0c4f3962`, which
predates this checkpoint. Its stage record still captures host, queue, PyTorch,
CUDA, and model identity, but it is not evidence for the complete v1
accelerator document.

## Deferred L4/A6000 performance checkpoint

The 22,926-row subject-mask reference job `153236891` measured `269.3 s` for
U-Net inference, or `85.1 ROI/s`, on an L4 node after the ROI cache and model
were staged on node-local storage. Historical workstation runs of the same
model family reported approximately `195–214 ROI/s` on the A6000 host.

The current profile attributes `186.56 s` (`8.137 ms/ROI`) to synchronization
after the model forward call. A representative historical profile attributed
approximately `3.09 ms/ROI` to that phase. Node-local staging therefore does
not explain the observed difference: it removes remote input latency, while
the measured gap is primarily inside the accelerator/runtime boundary. The
new sharded write and exact reread-validation phases add end-to-end time but do
not explain the forward synchronization gap.

Some difference between L4 and A6000 execution is plausible, but the observed
gap is not accepted as a hardware-only expectation. It remains a deferred
performance investigation. Run a controlled comparison only after the mask
publication and Crimson demo gates are complete, using:

- the same committed Palette code, model digest, and node-local ROI cache;
- fixed batch size `128`, precision policy, preprocessing, and warmup;
- an inference-only measurement with publication and validation timed
  separately;
- at least three fresh repetitions on both L4 and A6000;
- the complete `palette.accelerator_runtime` document plus clock/power and
  utilization telemetry as non-scientific benchmark evidence; and
- separate throughput for ROI reading, host-to-device transfer, forward
  enqueue/synchronization, device-to-host transfer, output writing, and
  validation.

This checkpoint must not block the selector-ineligible mask fixture or the
Crimson presentation demo. Its purpose is later capacity planning and runtime
tuning, not scientific or storage-contract acceptance.

## Checklist

- [x] Version the accelerator document.
- [x] Capture driver, framework/runtime, precision policy, and visible devices.
- [x] Embed the document in every shared environment record.
- [x] Reuse it when deriving final run provenance.
- [x] Preserve U-Net inference provenance through raw publication receipts.
- [ ] Validate one new CUDA detection, keypoint, and mask artifact outside the
  sandbox.
- [ ] After the Crimson mask demo, run the controlled current-code L4/A6000
  inference-only comparison described above.
- [ ] Optionally project accelerator identity into performance-registry query
  columns without duplicating scientific authority.
