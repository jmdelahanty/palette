# Canonical Detection Storage Reduction Contract

Status: frozen before repetition 5

This contract reduces balanced 200,000-frame canonical-detection benchmark
evidence. It selects candidates for the next benchmark stage only. It does not
promote a production storage profile, update a selector or registry, or create
a training artifact.

## Evidence Preconditions

The reducer fails closed unless every source workflow has:

- a complete schema-v2 aggregate and valid matrix manifest;
- the same seed, scales, workloads, candidate plans, correctness gates, and
  performance tolerances;
- the same Palette commit, frozen fixture ID, and fixture-manifest digest;
- nonoverlapping `(scale_id, repetition_index)` blocks;
- complete candidate coverage for every repetition at its scale; and
- block and local-write evidence paths contained in that workflow.

Candidate gates are evaluated only after at least five balanced repetitions.
Four repetitions may be reduced as a preview, but cannot produce a selection.

## Control And Gates

The control is the regular, unsharded 1 MiB target-chunk candidate
`regular__chunk_1048576`. A candidate must pass every check below:

| Dimension | Reduced statistic | Maximum ratio to control |
| --- | --- | ---: |
| local write pipeline | median | 1.25 |
| publication total | median | 1.25 |
| write-phase peak RSS | median | 1.25 |
| each required PRFS latency metric | median | 1.10 |
| each required PRFS latency metric | cross-repetition p95 | 1.20 |

The required PRFS latency metrics are:

- fresh reader-subprocess wall time;
- eager `frame_row_offsets` first-pass and warm-pass consumer time;
- random frame-slice first-pass and warm-pass per-frame p95 latency;
- random observation-range first-pass and warm-pass per-range p95 latency; and
- sequential 700-frame-window first-pass and warm-pass consumer time.

Direct and consolidated metadata-open timings remain in the report but are not
performance gates in this filesystem reduction. Request-count and true
remote-open behavior require the later HTTP Range and Crimson stage.

Correctness is not traded against performance. Every source block must already
have exact decoded arrays, exact consumer reads, direct and consolidated opens,
an unchanged fixture, and isolated fresh destinations before it reaches this
reducer.

## Next-Stage Selection

Among candidates passing all gates, select the lowest median payload-object
count. Break an object-count tie with median fresh PRFS reader-subprocess wall
time, then deterministic candidate ID order.

The selected candidate is only the preferred layout to carry into
full-duration, HTTP Range, and Crimson validation. A production profile still
requires those consumer-side results.
