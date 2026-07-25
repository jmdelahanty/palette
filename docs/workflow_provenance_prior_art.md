# Workflow & provenance engines: prior art, and what palette reinvented

<!-- contract-meta
status: reference
last_verified: 2026-07-24
authority: background and recommendations only; not a migration decision
-->

A reference for the class of tools that already solve parts of the problem
palette solves by hand. Written 2026-07-24 as a companion to
`docs/diagnostics/provenance_chain_audit_2026-07-24.md`, which found ~60 issues
that reduce to four root causes — several of which are non-issues in systems
built on the ideas below.

This is a learning document, not a migration proposal. The recommendation at the
end is to steal three mechanisms, not to adopt any of these tools. There is also
an honest section on **what these tools do not give you**, because a meaningful
part of palette is not reinvention.

---

## The one idea

Every tool here is a variation on a single mechanism:

> **Name a computation's output by a hash of everything that went into it.**

"Everything" means the code, the resolved parameters, the *content* of the
inputs, and the software environment. Call that the **task hash**.

Once outputs are named by task hash, four things you currently implement
separately fall out as consequences of one decision:

| You get | Because |
|---|---|
| **Idempotent retry** | Re-running computes the same hash, lands in the same place. No orphans, no collisions. |
| **Automatic staleness** | Change an input's bytes → its hash changes → every consumer's hash changes → everything downstream reruns, transitively. Nobody writes a cascade. |
| **Free caching** | If a directory with this hash exists and is complete, skip the work. |
| **Provenance** | The hash *is* the provenance record. The inputs are recoverable from it. |

The second-order effect is the important one, and it is the reason these systems
stay correct while palette's provenance quietly decayed:

> **Provenance and caching become the same mechanism.** You depend on the cache
> daily because recomputation is expensive. If the provenance were wrong, you
> would notice within the hour — as a spurious rerun, or a cache hit on data you
> can see is stale.

That is a *forcing function*. Palette has no equivalent: nothing anyone does
daily gets worse when provenance is absent or lying, which is why the strict
completion epoch sat disarmed for five weeks and nobody noticed.

### The ancestor, and the original sin

All of this descends from **`make`** (1976). Make had the DAG, the declarative
rules, and the incremental rebuild. What it did not have was content addressing —
it decided staleness by comparing **mtimes**. That works on one machine with one
clock and breaks on a cluster with NFS, coarse mtime granularity, and clock
skew.

Palette independently arrived at mtime comparison in several places
(`cli/palette.py:733` compares `palette_run_completed_at_utc` stamped on
different compute nodes; `cluster/keypoints/clipped_collection.py:149` selects a
pipeline input by `max(paths, key=st_mtime)`). Everyone starts there. The
history of this tool class is largely the story of moving off it.

---

## Nextflow

**What it is.** A dataflow workflow engine from CRG Barcelona, dominant in
bioinformatics. `nf-core` is its community pipeline collection. Apache-licensed,
very much alive.

**Why it is the closest match to palette.** It runs on **LSF natively**. The
same pipeline script runs under `local`, `lsf`, `slurm`, `sge`, Kubernetes, AWS
Batch, Google Batch — you change one config line. `src/fisheye/cluster/lsf/`
(~12.6k LOC) is a hand-built version of one of its executor backends.

### The model

You declare `process` blocks — black boxes with declared inputs and outputs —
and connect them with **channels**. You never write a scheduler call. You
declare the shape of the computation; the engine decides what can run when.

```groovy
process DETECT {
    container 'palette/detect:2026.07'
    cpus 4
    memory '32 GB'
    accelerator 1, type: 'nvidia-a100'

    input:
    tuple val(recording_id), path(video), path(weights)

    output:
    tuple val(recording_id), path("detections.zarr")

    script:
    """
    palette detect --video ${video} --model ${weights} --out detections.zarr
    """
}

process REFINE {
    input:
    tuple val(recording_id), path(detections)

    output:
    tuple val(recording_id), path("refined.zarr")

    script:
    """
    palette refine-detect --in ${detections} --out refined.zarr
    """
}

workflow {
    recordings = Channel.fromPath('recordings/*/video.mp4')
        .map { v -> tuple(v.parent.name, v, file(params.weights)) }

    DETECT(recordings) | REFINE
}
```

Run it:

```bash
nextflow run main.nf -profile lsf -resume -with-report -with-trace -with-dag dag.html
```

### The mechanism that matters

Every task execution gets its own working directory named by a **128-bit hash**:

```
work/
  ab/
    cd1234ef567890abcdef1234567890/
      .command.sh        # the exact rendered script
      .command.run       # the wrapper actually submitted to LSF
      .command.log       # stdout+stderr
      .exitcode
      video.mp4 -> /nvme1/recordings/.../video.mp4    # inputs, staged as symlinks
      detections.zarr                                  # outputs, produced in place
```

The hash covers the script text, the input files, the resolved parameter values,
and the container image. Change any one and you get a different directory.

`-resume` then means: recompute every task hash; for each one, if a completed
directory with that hash exists, skip and reuse it; otherwise run. There is no
"is this stale?" logic anywhere — staleness is *the hash not matching*.

### The nuance you should notice

Nextflow's **default caching mode hashes file metadata** (path, size, last
modified), not file contents, because content-hashing terabytes on every resume
is slow. You opt into real content addressing per process:

```groovy
process DETECT {
    cache 'deep'      // hash file CONTENT, not metadata
    // 'lenient'      // even weaker: size + timestamp only, for network filesystems
}
```

This is worth internalizing: **even the best-in-class tool ships the pragmatic
approximation as the default and makes the rigorous version opt-in.** Palette's
`stat_v1` fingerprint (`shared/import_source_fingerprint.py:12,42-58`), which
hashes `{path, size, mtime}` rather than bytes, is the same tradeoff — arrived at
independently, and defensible. The difference is that Nextflow tells you which
mode you are in, and palette does not.

### What else it gives you

- `-with-trace` — per-task TSV: exit status, duration, **max RSS, CPU%, queue
  wait, LSF exit reason**. This is precisely the `bjobs`/`bacct` data palette
  never collects (audit finding F1/F2, where an OOM-killed job leaves a status
  file saying `"running"` forever).
- `-with-report` / `-with-timeline` — HTML execution reports.
- `-with-dag` — renders the actual DAG.
- `nf-prov` plugin — emits W3C PROV and RO-Crate provenance bundles.
- Container integration: Docker / Singularity / **Apptainer** (the one you'd use
  on an HPC cluster), with the image digest as part of the task hash. Palette
  captures no container digest anywhere.

### Honest limits

- Groovy DSL. It is not Python and it is a real learning curve.
- File-granular. A process's inputs and outputs are *files*. It has no concept of
  "row 4711 of this array changed" — see the limits section below.
- The `work/` directory grows without bound until you `nextflow clean`.

---

## Snakemake

**What it is.** The Python-native, `make`-like alternative. You write **rules**
with input/output file patterns; Snakemake builds the DAG backward from the
outputs you request. Big in bioinformatics alongside Nextflow; the two are the
main rivalry in the field.

```python
rule detect:
    input:
        video   = "recordings/{rec}/video.mp4",
        weights = config["detect_weights"],
    output:
        "recordings/{rec}/detections.zarr"
    threads: 4
    resources:
        mem_mb = 32000, gpu = 1
    container:
        "docker://palette/detect:2026.07"
    shell:
        "palette detect --video {input.video} --model {input.weights} --out {output}"

rule all:
    input: expand("recordings/{rec}/detections.zarr", rec=RECORDINGS)
```

**Why it is in this document.** Snakemake's default staleness check is
**mtime-based** — the `make` inheritance, and exactly the approach the audit
flagged in `cli/palette.py:733`. The community hit the same wall you are on, and
the fix was to make the trigger configurable:

```bash
snakemake --rerun-triggers mtime params input software-env code
```

That flag list is a good checklist in its own right. It enumerates the five
things that *should* invalidate a result, and palette currently tracks the
first weakly, the second partially, the third nominally, the fourth not at all
(no lockfile hash, no container digest), and the fifth via a git SHA that is
`dirty=true` on 75% of runs with the diff unrecoverable.

Snakemake also has `--cache` for content-addressed shared caching between
workflows, `--report` for self-contained HTML reports with embedded provenance,
and cluster executors including LSF.

**Compared to Nextflow:** friendlier if you think in Python and in files;
weaker at dynamic, data-dependent DAGs; the pull model (declare the outputs you
want) versus Nextflow's push model (declare the dataflow) is a genuine
philosophical fork, not a quality difference.

---

## DVC (Data Version Control)

**The lightest-weight thing on this list, and probably the most immediately
applicable to palette.**

DVC is `git` for data plus a `make` for pipelines, with content addressing
throughout and **no daemon, no cluster, no Kubernetes**. It is a pip install.

```yaml
# dvc.yaml
stages:
  detect:
    cmd: palette detect --video data/video.mp4 --model models/best.pt --out out/detections.zarr
    deps:
      - data/video.mp4
      - models/best.pt
      - src/fisheye/detection/detect_yolo.py
    params:
      - detect.conf_threshold
      - detect.imgsz
    outs:
      - out/detections.zarr
```

```bash
dvc repro          # runs only what changed
dvc dag            # print the DAG
dvc metrics diff   # compare metrics across git commits
```

The state lives in `dvc.lock`, which records an **md5 for every dep and out**.
`dvc repro` compares current hashes against the lock and reruns only genuinely
changed stages. Large files live in a content-addressed cache
(`.dvc/cache/<first two hex>/<rest>`) with the working tree holding links, and
`dvc push`/`pull` moves them to remote storage.

**Why this one is worth a serious look.** It gives you, for a small fraction of
the adoption cost of Nextflow or Pachyderm:

- Content-addressed staleness (`dvc repro` is the working version of
  `audit_analysis_staleness`, which the audit found *structurally incapable* of
  ever returning `stale`).
- Data versioning tied to git commits — so "show me the dataset as of date D"
  becomes `git checkout <sha> && dvc checkout`. That is undefendable-claim #9 in
  the audit, closed.
- `params` tracking, which is the thing palette records as a hand-curated subset
  (`parameters` in stage provenance carries 5 keys for detect; Ultralytics
  inference defaults and `cudnn.benchmark=True` are recorded nowhere).

**Limits.** File-granular like the rest. No scheduler integration — it does not
submit to LSF, so it complements rather than replaces `cluster/lsf/`. And the
cache can get large.

---

## Pachyderm

**What it is.** Data versioning plus automatic pipeline triggering, on
Kubernetes. Think "git for data" fused with a reactive scheduler. (Acquired by
HPE in 2023 — worth knowing before betting on it long-term.)

**The model.** The core abstraction is a **versioned data repository** with
commits and branches. You commit data into a repo; a pipeline declares that it
subscribes to that repo; when a commit lands, Pachyderm runs the pipeline and
produces an **output commit** in an output repo.

```yaml
pipeline:
  name: detect
input:
  pfs:
    repo: recordings
    glob: "/*"          # each top-level entry is one "datum" = one parallel unit
transform:
  image: palette/detect:2026.07
  cmd: ["palette", "detect", "--in", "/pfs/recordings", "--out", "/pfs/out"]
```

```bash
pachctl inspect commit detect@master          # what produced this?
pachctl list commit detect@master --origin=auto
```

### The two things to steal

**1. Provenance is a first-class, indexed, queryable relation.** Every output
commit permanently records exactly which input commits it derived from.
"What produced this?" and "what is downstream of this?" are both database
queries returning in milliseconds — Pachyderm calls the two directions
*provenance* and *subvenance*.

Compare palette: `build_run_lineage_graph`
(`utils/inspect_run_lineage_graph.py:285`) recursive-descends a single archive,
there is **no cross-archive traversal at all**, and answering "which analyses
used the superseded mask run?" means N separate NFS walks with no index and no
incremental refresh.

**2. There is no side door.** You cannot produce output except through a commit
that records its provenance. This is the structural answer to palette's "three
review front-ends, one audit log" problem — `tune/detect_review_web.py` writing
canonical zarr with no audit event is simply not expressible in that model.

It is also genuinely incremental: **datums** (granularity set by the input glob)
are the reprocessing unit, so a changed subset reprocesses alone. That is
row-level invalidation, which palette cannot do even in principle because every
node in its DAG is a whole run
(`utils/inspect_run_lineage_graph.py:35-51` — no row fields anywhere).

**Limits.** It wants Kubernetes and a real ops story. For one person, that is
the whole objection and it is sufficient.

---

## DataLad

**What it is.** `git` + `git-annex` for datasets, from the neuroimaging
community (Halchenko, Hanke — the same world as BIDS and OpenNeuro). Relevant to
you specifically because it is the provenance tool neuroscience actually adopted.

```bash
datalad run \
  --input  'recordings/rec001/video.mp4' \
  --output 'recordings/rec001/detections.zarr' \
  "palette detect --video {inputs} --out {outputs}"
```

That records the command, its declared inputs and outputs, and the environment
into the **git commit message as machine-readable JSON**. Then:

```bash
datalad rerun <commit>              # re-execute from the record
datalad containers-run ...          # same, pinned to a container image
datalad diff --from <sha> --to HEAD
```

Content lives in git-annex, addressed by hash. History is git history.

**What it gets you cheaply:** "reconstruct the dataset as it stood on date D" is
free — it is a git checkout. That is one of the ten claims the audit found
palette cannot defend. And the run record is close to a well-formed PROV
Activity, which is more than palette's review path produces (grep for
`tool_version|code_version|git_commit` across `src/fisheye/labeling/` and
`src/fisheye/tune/` returns **nothing**).

**Limits.** git-annex has a learning curve and is not fond of millions of small
files. No scheduler integration.

---

## MLflow — the model-side equivalent

Everything above is about data pipelines. The ML side has its own lineage stack,
and palette reinvented parts of it in `registry/`.

**MLflow Tracking** logs params, metrics, and artifacts per run.
**MLflow Model Registry** versions models with stage transitions and, crucially,
records the **artifact hash** and the **model signature** (input/output schema).
Autologging hooks into PyTorch/Ultralytics to capture hyperparameters and
metrics without manual instrumentation.

Two audit findings map straight onto features that exist here:

- `training_sets` stores `dataset_ids_json` — a list of **mutable** dataset IDs
  with no content fingerprint, no `set_version`, no `parent_set_id`
  (`registry/migration_bodies.py:516-526`). The standard answer is a dataset
  hash logged as a run input, which MLflow's dataset tracking and DVC both do.
- `shared/pose_model_schema_binding.py` — which the audit called the exemplary
  surface in the repo — is an independently-derived, arguably stricter version
  of an MLflow model signature plus artifact hashing. **This one you reinvented
  well.**

Weights & Biases occupies similar ground with better visualization and a hosted
model.

---

## W3C PROV — the vocabulary

Not a tool. The **standard data model** underneath most of this, and worth
learning because it names your gaps precisely.

Three node types:

| Type | Meaning | palette example |
|---|---|---|
| **Entity** | a thing | a `detect_run` group, a video file, a model checkpoint |
| **Activity** | something that happened | a detect execution, a human review session |
| **Agent** | who or what is responsible | the annotator, the LSF job, the software version |

Core edges: `wasGeneratedBy` (Entity ← Activity), `used` (Activity → Entity),
`wasDerivedFrom` (Entity → Entity), `wasAttributedTo` (Entity → Agent),
`wasAssociatedWith` (Activity → Agent), `wasInformedBy` (Activity → Activity).

**Why this matters for palette.** Map the audit onto it:

- **Entity** coverage: decent. Runs, arrays, and rows are all identified, some
  with real content hashes.
- **Activity** coverage: partial. `stage_provenance` records the stage, the
  parameters, and the git commit.
- **Agent** coverage: **almost nothing.** `reviewer` is optional in the canonical
  payload (`docs/review_status_schema_unification_contract.md:51-53`);
  `approved_by` falls back to `"unknown"`
  (`tune/detect_review.py:1513`); identity resolves from a bearer link
  (`labeling/web_auth.py:96-111`) or `os.environ.get("USER")`; and no tool or
  code version is recorded anywhere in the edit path.

In PROV terms, **palette's derivations are largely unattributed.** That is not
an implementation oversight — it is a category that was never modeled. Having
the word for it is most of the fix.

**RO-Crate** is the packaging layer above PROV: a JSON-LD manifest describing a
dataset and its provenance, with a Workflow RO-Crate profile for exactly this
use case. It is what you would emit if you wanted a published dataset to carry
machine-readable lineage. **CWL** (Common Workflow Language) is the
interoperable workflow description standard, with CWLProv for its provenance.

---

## What palette reinvented

The honest mapping. "Quality" is my assessment from the audit, not a value
judgment about effort.

| palette mechanism | Prior art | Quality of the reinvention |
|---|---|---|
| Timestamped run names + `_refuse_output_collisions` + 4 manual `*_recovery.py` | Nextflow task hash; DVC content-addressed cache | **Worse.** The timestamp is the root of the TOCTOU race, the orphan accumulation, and the non-idempotent retry. |
| `shared/run_lineage_fingerprint.py` | Nextflow task hash; `dvc.lock` | **Partial.** Right idea; excludes model identity by construction (`TRANSIENT_LINEAGE_KEYS` strips `artifacts`), and `source_fingerprints` is empty in production. |
| `utils/audit_analysis_staleness.py` | `dvc repro`; Snakemake DAG; `nextflow -resume` | **Non-functional.** Structurally cannot return `stale`, and has no callers. |
| `registry/step_cascade.py` + `*_stale.py` | Transitive invalidation — automatic in all four engines | **Partial.** One hop, one edge, new-runs-only. |
| `utils/inspect_run_lineage_graph.py` | Pachyderm provenance/subvenance; PROV graph | **Unindexed.** Reconstructed by walking attrs; no cross-archive traversal. |
| `src/fisheye/cluster/lsf/` (~12.6k LOC) | Nextflow / Snakemake LSF executors | **Good but redundant.** Genuinely well-built kernel; ~all of it is a solved problem elsewhere. |
| `analysis_workflows/materializers/atomic_run_publisher.py` | Pachyderm commits; DVC cache atomicity | **Well done, wrongly scoped.** Real commit protocol; covers ~8 analysis materializers and none of the perception pipeline. |
| `shared/tabular_deltas.py` | git-annex / DataLad history; Pachyderm commits | **Well designed, zero callers.** |
| `registry/` SQLite catalog | Pachyderm PFS metadata; MLflow tracking store | **Reinvented, with an authority problem** the originals avoid by having exactly one writer. |
| `training_sets` / `training_runs` | MLflow Model Registry + DVC data versioning | **Weaker.** Mutable ID list, no dataset fingerprint, no immutability. |
| `shared/system_metadata.py:703-747` `build_invocation_record` | DataLad run record; PROV Activity | **Well done, unwired to the edit path.** |
| `shared/pose_model_schema_binding.py` | MLflow model signature + artifact hashing | **Better than the original.** Fails closed, re-hashes from disk, emits a binding digest. |
| `run_provenance` / `stage_provenance` schemas | W3C PROV; RO-Crate | **Reinvented without interop.** No standard vocabulary, so nothing external can read it. |
| Review front-ends with no audit trail | PROV Agent / attribution | **Not modeled at all.** |

---

## What is *not* reinvention

This section matters as much as the one above, and it is why "you should have
used Nextflow" is not the right conclusion.

**Every tool on this list is file-granular.** A Nextflow process consumes and
produces *files*. A DVC stage's `deps` and `outs` are *files*. A Pachyderm datum
is a file or directory. None of them has any concept of "row 4711 of this array
changed, and here is which downstream rows depend on it."

Palette's data is a 200,000-frame video with per-frame detections, keypoints,
masks, and derived kinematics, stored as chunked arrays. Treating each frame as
a file is not viable at that scale. So:

- **`shared/row_source_signature.py`, `shared/instance_keys.py`,
  `shared/rowset_fingerprint.py`, `shared/row_lineage.py`** — row-level content
  identity inside array storage — have **no equivalent in any tool on this
  list.** This is genuinely novel work and the audit rated
  `row_source_signature.py` the most rigorous scheme in the repo.
- **The human-in-the-loop correction model** — a reviewer edits one instance in
  one frame, and that must propagate — is not something any of these engines
  models. Pachyderm comes closest and is still file-granular.
- **Coordinate frame contracts, pixel authority, arena geometry, the
  acquisition→analysis frame mapping** are domain work with no prior art to
  borrow.
- **Real-time closed-loop acquisition** (the TensorRT path) is out of scope for
  every one of these tools.

So the accurate framing is narrower than "you reinvented the wheel":

> **The orchestration and lineage *plumbing* is reinvented. The array-native,
> row-level provenance *data model* is not — nobody else has built that.**

And the reinvention was not obviously avoidable at the start, because the data
model genuinely doesn't fit these tools. The thing worth doing differently is
not "adopt Nextflow"; it is **read what Nextflow does once the orchestration
problem becomes real**, and steal the mechanism rather than the system. That is
the step that got skipped.

---

## What to steal, concretely

Three mechanisms, in order of payoff. None requires adopting a tool.

### 1. Content-hash the run names

Replace `detect_2026-02-09_10-10-20` with a digest over
`(model_sha256, resolved_params, input_rowset_fingerprint, code_version)`.

This is generalizing a mechanism palette has already proven —
`shared/rowset_fingerprint.py` plus `assert_rowset_fingerprint_matches` already
does exactly this across refined_detect → arena_assignment → tracking → crop,
and it works. Extending it to name runs collapses a large fraction of the audit
at once:

- Retry becomes idempotent; the four `*_recovery.py` scripts become unnecessary.
- The `detect_yolo.py:539-547` TOCTOU race becomes structurally impossible.
- Orphaned partial runs stop accumulating.
- Staleness becomes computable rather than declarable, which is what
  `audit_analysis_staleness` was trying and failing to be.

### 2. Make provenance load-bearing for something you want

The Nextflow lesson, applied locally. The cheapest version: make run resolution
work **only** through the strict path, with no legacy fallback. One bad week,
then a system that surfaces a regression in minutes instead of five weeks.

The general principle: if a correctness mechanism has no daily consumer, it will
decay and you will not find out. Give it a consumer.

### 3. Give the lineage graph an index

A single SQLite table written at commit time:

```sql
CREATE TABLE run_lineage_edge (
    output_run_uid  TEXT NOT NULL,
    input_run_uid   TEXT NOT NULL,
    edge_kind       TEXT NOT NULL,   -- 'source_detect_run', 'model', 'video', ...
    input_digest    TEXT,            -- content hash at the time of consumption
    recorded_at_utc TEXT NOT NULL,
    PRIMARY KEY (output_run_uid, input_run_uid, edge_kind)
);
```

That is Pachyderm's provenance/subvenance property without Kubernetes. It turns
store-wide questions from N filesystem walks into a query, and — because
`input_digest` records the hash *at consumption time* — it makes "did my upstream
change after I consumed it?" answerable, which is the single thing the current
staleness system cannot do.

---

## Further reading

- **Nextflow** — https://nextflow.io ; `nf-core` pipelines at https://nf-co.re ;
  the caching/resume documentation is the part to read first.
- **Snakemake** — https://snakemake.readthedocs.io ; read the `--rerun-triggers`
  and `--cache` sections.
- **DVC** — https://dvc.org ; start with "Get Started: Data Pipelines".
- **Pachyderm** — https://docs.pachyderm.com ; read the "Provenance" concept page
  even if you never install it.
- **DataLad** — https://handbook.datalad.org ; the DataLad Handbook is genuinely
  well written and neuroscience-flavored.
- **MLflow** — https://mlflow.org ; Model Registry and Dataset Tracking.
- **W3C PROV** — the PROV Primer (https://www.w3.org/TR/prov-primer/) is short
  and is the highest value-per-page item on this list.
- **RO-Crate** — https://www.researchobject.org/ro-crate/ ; Workflow RO-Crate
  profile.
- **CWL** — https://www.commonwl.org ; CWLProv for the provenance profile.
