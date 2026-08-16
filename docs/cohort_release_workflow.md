# Registry-defined cohort releases

Palette cohort releases separate a reusable scientific query from its immutable
evaluation. The query says which recordings qualify; the frozen cohort manifest
records exactly which active source analysis Zarrs qualified against one
read-only registry snapshot.

## Selection semantics

The v1 query schema is `palette.cohort_query`, version 1. Supported selectors
are:

- normalized stimulus modes, such as `CHASER`;
- exact protocol names and exact 64-character protocol hashes;
- exact days post fertilization values or an inclusive DPF range;
- exact line/strain, genotype, and cross identifiers;
- required registry pipeline steps whose latest status must be `ok`;
- dataset status, Zarr use, and Zarr origin.

Repeated values within one field are OR. Different fields are AND. There is no
implicit limit or sampling: every matching active source analysis dataset is
included. Ordering is deterministic.

Subject metadata defaults to `unambiguous_recording`. Each selected biological
field must be populated for every normalized subject and have exactly one
distinct value for the recording. This is the safe default for recordings that
may eventually contain one, two, or more subjects. `any_subject` intentionally
allows a subset to match; `all_subjects` requires complete values and every
subject to match. These alternatives are available only when the scientific
query explicitly needs those meanings.

Missing normalized metadata for an active biological selector is a blocker by
default. `missing_selected_metadata: exclude` is available, but using it means
that omission is part of the cohort definition. Multiple active candidate
analysis Zarrs for one recording, multiple stimulus runs marked latest, and
non-`ok` prerequisites are also blockers.

Protocol selectors operate over known normalized protocol values. A missing
protocol value is a recorded exclusion, not a match. The plan keeps all blocker
and exclusion reasons; reason counts are non-exclusive because a row can fail
more than one predicate.

## Plan, coverage, and freeze

Use the example query directly:

```bash
PYTHONPATH=src scripts/py -m fisheye.cohorts plan \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --spec examples/cohorts/redscare_exact_protocol_v1.yaml \
  --output /tmp/redscare_cohort_plan.json

PYTHONPATH=src scripts/py -m fisheye.cohorts coverage \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --spec examples/cohorts/redscare_exact_protocol_v1.yaml \
  --output /tmp/redscare_subject_metadata_coverage.json

PYTHONPATH=src scripts/py -m fisheye.cohorts freeze \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --spec examples/cohorts/redscare_exact_protocol_v1.yaml \
  --plan-output /tmp/redscare_cohort_plan.json \
  --output /tmp/redscare_frozen_cohort.json
```

The frozen `palette.frozen_cohort_manifest` contains the normalized query and
its SHA-256, the immutable registry UUID, a hash of every registry row consulted,
exact dataset/recording/Zarr membership, protocol context, subject context,
prerequisite statuses, and its own canonical manifest hash. Freezing refuses
blockers and an empty result. New v2 manifests fail before submission when the
supplied registry UUID differs from the frozen identity. Historical v1 manifests
remain readable but do not claim this identity guarantee.

The coverage report distinguishes normalized values from scalar legacy
provenance candidates. Legacy candidates are diagnostic only; they are not used
to satisfy a biological selector. This prevents a scalar compatibility field
from being mistaken for authoritative multi-subject metadata. Review the report,
then use the established registry maintenance workflow to normalize suitable
legacy provenance:

```bash
# Read-only preview first.
scripts/py -m fisheye.registry.maintenance \
  --registry /path/to/palette_registry.sqlite \
  --backfill-subject-dish-cross \
  --dry-run
```

Registry mutation remains a separately approved, serialized maintenance action;
the cohort command never performs this backfill automatically.

## One guarded release command

Render the complete release DAG without submitting it:

```bash
scripts/submit_cohort_release_bsub.sh \
  --spec examples/cohorts/redscare_exact_protocol_v1.yaml \
  --release-id redscare_protocol_578a2c_v1 \
  --chaser-authority-manifest /path/to/chaser_authority.json \
  --speed-level smoothed \
  --queue short
```

After inspecting `release_submission.json`, add `--submit`. Submission requires
the workstation and cluster-visible Palette checkouts to be clean and at the
same commit.

`--speed-level` is a required scientific choice for authoritative epoch
analytics and must name the physical track-speed representation (`raw`,
`filtered`, `smoothed`, or `averaged`). Palette records and forwards the exact
choice; the release front end does not infer or silently default it.

The command freezes membership before any LSF submission and then creates this
dependency chain:

```text
recording analytics array
  -> final per-recording run binding / virtual collection
  -> Parquet export + statistics + serialized export registry index
  -> semantic PNG montages + immutable report + serialized report index/check
```

Every downstream LSF dependency uses `done(job_id)`, so a failed stage prevents
publication stages from starting. Workers validate the captured Palette commit,
future inputs after dependencies complete, manifest hashes, exported Parquet
parts, and final report files. Existing immutable destinations and submission
run directories are refused.

The release directory contains:

- `cohort_query.json`;
- `cohort_plan.json` with every exclusion and blocker;
- `metadata_coverage.json`;
- `frozen_cohort_manifest.json`;
- `zarr_paths.txt` with its recorded SHA-256;
- `release_submission.json` with job IDs, commands, commit, and expected outputs;
- per-stage job scripts, logs, status, and validation records.

The current release implementation runs the generic chaser analysis family and
therefore requires `CHASER` as a stimulus mode. The query contract itself is
protocol-neutral and can support other analysis-family release drivers later.

## Direct typed selectors

A checked-in YAML/JSON query is preferred for durable releases. For a new
exploratory definition, the same command can create the versioned query from
typed flags:

```bash
scripts/submit_cohort_release_bsub.sh \
  --direct-selectors \
  --release-id chaser_dpf7_ab_v1 \
  --cohort-id chaser_dpf7_ab \
  --cohort-name "CHASER, DPF 7, AB background" \
  --chaser-authority-manifest /path/to/chaser_authority.json \
  --speed-level smoothed \
  --stimulus-mode CHASER \
  --protocol-hash 578a2cd8b3aa5762994b61a2405b94e1cf5012d68c1fa6bfcb76a5e04eb45492 \
  --dpf 7 \
  --strain "AB [AB IC] SEPT25" \
  --subject-match-policy unambiguous_recording \
  --missing-selected-metadata error
```

Useful repeated flags include `--protocol-hash`, `--protocol-name`,
`--stimulus-mode`, `--dpf`, `--strain`, `--genotype`, `--cross-id`, and
`--require-step-ok`. Inclusive ranges use `--dpf-min` and `--dpf-max`.

## Registry update serialization

The release never asks two jobs to write the registry concurrently. Recording
analytics and collection binding do not register the cohort. The export job
indexes only after validation; the report job depends on that export and then
indexes the verified report. Registry maintenance/backfill stays outside the
release DAG and must be run separately.
