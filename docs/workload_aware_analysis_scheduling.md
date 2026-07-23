# Workload-aware analysis scheduling

## Decision

Cluster allocations for Palette analysis must be selected from the persisted
workload of the stage to be run, not from nominal recording duration alone.

Recording duration is not a stable proxy for computational cost. A 30-minute
recording contains approximately 180,000 source frames at 100 Hz but
1,260,000 source frames at 700 Hz. Conversely, a high-frame-rate recording can
have a sparse persisted analysis rowset and should not be charged as though
every source frame requires processing.

## Stage-specific workload authorities

The planning preflight should resolve an immutable input run, measure its
available persisted surfaces, and use the relevant quantity for each stage.

| Stage family | Primary scheduling work unit | Additional cost factors |
| --- | --- | --- |
| Subject shape | refined-subject-mask rows | ROI shape, component count, mask encoding |
| Track kinematics | selected track-sample/frame-axis rows | selected track count, temporal operators |
| Swim-bout detection | selected track-sample rows | detector levels; normally comparatively serial |
| Eye angles | aligned subject-shape and keypoint rows | requested angle channels and physical column layout |
| Tail kinematics | subject-shape rows | spline-point count and requested derived surfaces |
| Bout kinematics | detected bout rows | selected per-bout temporal windows and optional gaze joins |

The preflight must distinguish source-video frame count from retained analysis
rows. It should retain both in provenance, but allocate from the authoritative
stage input rows and explicit complexity factors.

## Initial allocation policy

Until empirical timing models replace these tiers, use an explicit,
provenance-recorded policy for dense full-recording materialization:

| Effective stage rows | Initial CPU allocation |
| ---: | ---: |
| up to 100,000 | 8 cores |
| 100,001–500,000 | 16 cores |
| over 500,000 | 32 cores |

These are starting allocations, not scientific parameters. The planner must
record the chosen tier, measured work units, relevant complexity factors, and
policy version. Operators may still explicitly override an allocation; that
override and its reason must also be recorded.

For reference, a 30-minute recording at 30 Hz has about 54,000 frames and
typically falls in the 8-core tier. The same duration at 100 Hz has about
180,000 frames, while 700 Hz has about 1.26 million frames; their allocation
must follow retained stage rows rather than the shared duration.

## Workflow shape

The current workflow reserves one allocation for a serial dependency chain.
That is operationally simple but can reserve 32 cores while a later stage is
effectively serial. The target design is resource-aware stage grouping:

1. Run high-throughput materializers such as subject shape and eye angles in
   appropriately sized allocations.
2. Run serial or light stages, including ordinary bout detection, in smaller
   allocations when they are not bundled for convenience.
3. Preserve exact run pinning, completion-last publication, and fail-closed
   dependency checks across job boundaries.

This grouping is a scheduling optimization only: it must not change source
selection, detector parameters, numerical outputs, or publication contracts.

## Evidence and review

The Sleepyfish subject-shape benchmark documents why worker count matters for
large dense recordings: a 1.17 million-row canary took roughly 466 seconds of
compute with 32 single-threaded workers. See
[`subject_shape_performance_benchmark.md`](subject_shape_performance_benchmark.md).

Before making automatic tiers the default, validate them against a matrix of
row counts, mask complexity, hosts, and stage types. The scheduler should then
replace coarse row bands with versioned throughput estimates and a target
wall-time policy.
