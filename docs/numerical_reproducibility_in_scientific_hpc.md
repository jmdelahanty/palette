# Numerical Reproducibility in Scientific HPC

## Purpose

Scientific provenance should make it possible to explain where a result came
from, reproduce the calculation at an appropriate level, and distinguish a
meaningful scientific disagreement from harmless numerical roundoff.

That does not require saving every temporary array or demanding that every
floating-point bit remain identical on every computer. It requires an explicit
policy for data authority, numerical representation, derivation, and
validation.

This guide describes that policy and uses behavioral kinematics as a running
example.

## Three Levels of Reproducibility

It is useful to separate three goals that are often all called
"reproducibility."

| Level | Question | Typical criterion |
| --- | --- | --- |
| Scientific reproducibility | Does the analysis support the same biological conclusion? | Agreement within a domain-justified scientific tolerance |
| Numerical reproducibility | Do repeated computations agree within a defined numerical error envelope? | Absolute/relative tolerance, ULP limit, or a versioned deterministic kernel |
| Bitwise reproducibility | Are all output bits identical? | Exact byte or array equality |

Most scientific and HPC workflows should target scientific and numerical
reproducibility. Bitwise reproducibility is especially useful for:

- identifiers, frame indices, labels, masks, and other discrete values;
- hashes and file-transfer integrity;
- literal aliases that claim to be exact copies;
- small regression canaries;
- algorithms deliberately designed for deterministic bitwise execution.

Bitwise equality is usually too strong as a universal requirement for derived
floating-point data.

## Why Floating-Point Results Can Differ

Floating-point numbers represent a finite subset of real numbers. Most decimal
fractions and many mathematical results cannot be represented exactly.
Operations therefore round to a nearby representable value.

Floating-point addition is not associative:

```text
(a + b) + c may not equal a + (b + c)
```

This matters in reductions such as sums, means, standard deviations, and dot
products. Parallel workers can combine partial results in a different order
from a serial process. Low-order bits can consequently depend on:

- CPU model and vector instruction set;
- GPU kernel and device architecture;
- compiler and optimization flags;
- numerical library versions;
- thread or worker count;
- chunk boundaries and reduction-tree shape;
- input, compute, accumulator, and output dtypes.

These differences are not automatically errors. Their meaning depends on the
declared numerical and scientific contract.

## Storage Dtype, Compute Dtype, and Accumulator Dtype

A single derived array can involve several representations:

- **Input dtype:** representation of the authoritative parent data.
- **Compute dtype:** representation used by elementwise or transformation
  kernels.
- **Accumulator dtype:** representation used by reductions.
- **Output dtype:** representation persisted as the public result.

For example:

```text
source heading dtype: float32
smoothing compute dtype: float64
persisted smoothed heading dtype: float32
turning parent: persisted smoothed heading
turning compute dtype: float64
turning output dtype: float32
angle wrap: [-180, 180)
```

The rounding point is part of the algorithm. These two graphs are not
bitwise-equivalent:

```text
float64 parent -> downstream calculation -> float32 output
float64 parent -> float32 persisted parent -> downstream calculation -> float32 output
```

If the public derivation metadata says that a downstream array depends on the
persisted parent, the writer and validator should both use that persisted
parent. They should not independently use an unrecorded higher-precision
intermediate.

## What Provenance Should Record

A practical provenance record should include:

- authoritative input paths, identities, row axes, and hashes;
- code commit and container or software environment;
- algorithm and derivation-contract versions;
- parameters, units, calibration, coordinate frame, and angle convention;
- input, compute, accumulator, and output dtypes where they affect semantics;
- authoritative parent arrays for each public derived surface;
- rounding, clipping, missing-value, and overflow policies;
- random seeds and deterministic-mode settings;
- backend and hardware information when they can materially affect results;
- requested and effective worker/chunk topology for parallel reductions or
  writes.

It is normally unnecessary to persist every ephemeral float64 intermediate.
The versioned code plus the numerical policy should define those intermediates
well enough to explain or reproduce the calculation.

## Storage Contract Versus Derivation Contract

The two contracts answer different questions.

### Storage contract

The storage contract describes what is persisted:

- path and logical name;
- dtype and shape;
- axis identity and units;
- missing-value representation;
- valid storage range;
- chunks, shards, and compression;
- whether the array is authoritative, derived, or an exact alias.

### Numerical derivation contract

The derivation contract describes how a value is calculated:

- authoritative input references;
- operation order;
- compute and accumulator precision;
- angle wrapping or coordinate convention;
- smoothing alignment and edge behavior;
- clipping or saturation behavior;
- output rounding point;
- algorithm version.

For example:

```text
delta_heading_smoothed_degrees =
    wrap_difference(
        persisted_float32(smoothed_heading_degrees),
        interval=[-180, 180)
    )
```

This is more reproducible than saying only that both arrays came from the same
tracking run.

## Choosing an Equality Rule

Validation should match the semantics of the surface.

| Surface type | Recommended validation |
| --- | --- |
| IDs, frame indices, row keys, labels | Exact equality |
| Masks and discrete segmentations | Exact equality when representations match |
| File copies and content-addressed objects | Cryptographic digest |
| Literal compatibility aliases | Exact dtype, shape, NaN mask, and values |
| Deterministic arrays derived from a persisted public parent | Exact result from one shared, versioned kernel when practical |
| General floating-point transformations | Explicit absolute/relative or ULP envelope |
| Bounded values such as probabilities or resultants | Defined rounding/clipping policy plus a representational boundary rule |
| Biological endpoints | Separately justified scientific acceptance threshold |

Do not use an arbitrary tolerance merely to make tests pass. A tolerance should
come from a numerical error model, a representational bound, or a scientific
resolution requirement.

## ULPs and Boundary Values

An ULP is a "unit in the last place": the spacing between adjacent
floating-point values at a given magnitude.

A circular resultant is mathematically in `[0, 1]`. A floating-point reduction
can nevertheless produce the next representable float32 value above 1:

```text
1.0000001192092896
```

There are two defensible policies:

1. Clamp the computed mathematical result to `[0, 1]` before persistence.
2. Preserve a historical writer and accept only a precisely defined one-ULP
   storage envelope, while still requiring exact recomputation.

The first is preferable for a new versioned writer. The second can be useful
for compatibility with already persisted data. An open-ended tolerance is not
equivalent to either policy.

## Case Study: Track-Kinematics Publication

The long Sleepyfish recording exposed three numerical-contract issues.

### Acceleration summary reduction

The writer computed the mean and standard deviation from the persisted float32
acceleration array. The validator promoted the array to float64 before reducing
it. Both summarized the same data, but the low bits differed.

Resolution: writer and validator use one shared summary kernel in the declared
persisted float32 reduction domain.

### Circular resultant boundary

Two of 39,563 per-second heading resultants were exactly one float32 ULP above
1. The arrays otherwise exactly matched recomputation.

Resolution: the compatibility validator accepts only the one-step float32
storage envelope of mathematical `[0, 1]`. A future versioned writer may clamp
before persistence.

### Smoothed turning parent

Three arrays differed at one sample each:

- smoothed heading delta;
- smoothed angular velocity;
- smoothed angular speed.

The largest difference was approximately `5.82e-11` degrees. The persisted
smoothed heading itself was correct. The downstream writer and validator were
crossing a rounding boundary through a higher-precision intermediate.

Resolution: the downstream turning kernel now explicitly uses the persisted
float32 smoothed-heading surface named by its derivation metadata.

None of these differences were biologically meaningful. They were still useful
because they revealed underspecified numerical contracts.

## A Layered Validation Strategy

Palette should use layered assurance rather than applying maximum-cost checks
to every surface equally.

### Every production run

- Verify archive and recording identity.
- Verify axes, row keys, units, calibration, and coordinate lineage.
- Verify required arrays, shapes, dtypes, and completion state.
- Verify algorithm version and parameters.
- Publish atomically so incomplete runs never become selector-eligible.

### Numerical contract validation

- Use shared writer/validator kernels.
- Recompute from declared authoritative parents.
- Apply exactness only where the contract promises exactness.
- Apply explicit ULP or tolerance envelopes elsewhere.
- Keep cached summaries subordinate to their authoritative arrays.

### Canary and release validation

- Run stronger full-content and cross-surface audits on small reference data.
- Compare scientific endpoints against expected ranges.
- Exercise multiple worker counts or hardware backends when portability matters.
- Record benchmark and environment information.

This preserves strong scientific safeguards without forcing every routine run
to pay for repeated full-data validation.

## Parallel Zarr Writes Are a Separate Exactness Problem

Floating-point tolerance does not make overlapping parallel writes safe. Zarr
workers must own whole, non-overlapping physical chunks for every written
array, or write separate temporary outputs and merge deterministically.

That is a storage-concurrency requirement, not a numerical-tolerance question.
Logical row slices that share a physical chunk can overwrite one another even
when the computed values are individually valid.

See `docs/dask_zarr_write_safety.md` for Palette's write-safety policy.

## Practical Review Checklist

Before publishing a new floating-point analysis surface, answer:

1. What is the authoritative parent array?
2. What are the input, compute, accumulator, and output dtypes?
3. At what point is rounding applied?
4. Are bounds mathematical, representational, or biological?
5. Is clipping allowed, and at which step?
6. How are NaN, infinity, gaps, and invalid samples handled?
7. Does parallel execution change reduction order?
8. Is exact equality scientifically required, numerically intended, or merely
   convenient for a test?
9. What validation envelope is justified?
10. Which algorithm or derivation version records this policy?
11. Can a small canary exercise boundary and long-array reduction behavior?
12. Is an expensive full audit needed for every run or only for releases?

## Guiding Principle

Record enough to explain and reproduce the computation, and validate strongly
enough to detect scientifically meaningful corruption. Do not equate bitwise
identity with scientific truth, and do not use scientific tolerance to excuse
broken identity, units, axes, or provenance.
