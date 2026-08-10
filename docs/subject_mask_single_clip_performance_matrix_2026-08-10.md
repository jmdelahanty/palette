# Subject-mask single-clip performance matrix — 2026-08-10

Status: implemented benchmark-only workflow; no profile, writer, selector,
registry, or production archive is promoted by this matrix.

## Question

Current full-duration L4 canaries report approximately `82–85 ROI/s`. Historical
workstation runs of the same U-Net model family reported approximately
`195–214 ROI/s` on an RTX A6000. The comparable historical gap is therefore
about `2.3–2.6x`, not `4x`; the approximately `370 rows/s` result was keypoint
inference and is not a subject-mask baseline.

The first controlled question is whether the current per-batch synchronized
timing profiler materially changes throughput. A later L4/A6000 comparison is
useful only after this observer effect is measured.

## Frozen matrix

`fisheye.diagnostics.benchmark_subject_mask_single_clip_inference` prepares six
fresh L4 processes for one exact acquisition clip:

- three `async_no_synchronized_stage_profile` repetitions;
- three `async_synchronized_stage_profile` repetitions;
- alternating candidate order across repetitions;
- one active process at a time;
- identical model digest, crop-v2 and refined-keypoint-v2 references, video
  window, ROI rows, batch size, precision, probability encoding, chunks,
  shards, asynchronous writer, and validation behavior.

Each trial stages its own video, model, maintained reference archive, and crop
pixel work package on node-local scratch. It atomically publishes only an
immutable selector-ineligible bundle below `.palette_benchmarks`.

## Measurement

Every trial records:

- inference and whole-worker ROI/s;
- unsynchronized `perf_counter` durations for reference staging, video copy,
  model copy, crop-pixel materialization, inference CLI, local proof, and total
  pre-bundle work;
- atomic bundle-copy time;
- continuous `nvidia-smi` samples and summaries for SM/memory/decoder
  utilization, device memory, power, temperature, and clocks;
- the synchronized internal stage profile only for the explicitly profiled
  candidate; and
- exact per-array decoded row-unit receipt digests.

The aggregate fails closed unless all six trials complete, every GPU telemetry
sidecar is valid, and every decoded array receipt signature is identical.
Performance telemetry is explicitly excluded from scientific identity and
array-content digests.

## Scheduler boundary

The trial array is submitted with an external LSF condition on the existing
full-duration inference array:

```text
done(153303424)
```

This allows preparation and submission now while preventing the benchmark from
competing with the correctness canary. It does not wait for CPU refinement or
recording-level publication. The six GPU trials are serialized; the aggregate
runs only after all six succeed.

## Interpretation

This first matrix can establish synchronized-profiler overhead and locate
end-to-end time among staging, pixel materialization, inference, proofing, and
atomic publication. It does not by itself attribute device-kernel time or
promote a production configuration.

After it completes:

1. If unsynchronized L4 throughput returns near the historical range, remove
   synchronized profiling from ordinary inference and retain it only for
   diagnostics.
2. If inference remains near `85 ROI/s` with healthy GPU utilization, run the
   identical frozen package on the A6000 and L4 using the same commit.
3. If device utilization is low, inspect CPU preparation, D2H, writer
   backpressure, and validation before changing the model.
4. If L4 and A6000 both regress on the current commit, replay one historically
   identified commit/runtime against the same pixels and model digest.
5. Keep every follow-up benchmark selector-ineligible and separate scientific
   correctness from performance evidence.

