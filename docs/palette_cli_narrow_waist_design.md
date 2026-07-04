# Palette narrow-waist API design

<!-- contract-meta
status: proposed
created: 2026-07-01
owner: jeremy
supersedes: none
-->

## Goal

An operational API for Palette that a small-model agent (Haiku-class; Sonnet at worst)
can execute end-to-end workflows against — take a fresh recording through
import → detect → crop → keypoints → subject masks → analysis without a human or a
frontier model holding the workflow knowledge. Secondary goal: a clean seam for
language-agnostic access later (MCP, other languages).

The design principle throughout: **move the reasoning burden from the model into the
system.** Small models fail at *selection* and *recovery*, not execution. The API's job
is to make selection trivial (few verbs, one oracle) and recovery mechanical (errors
that are instructions).

## Concepts primer (read this first)

This design leans on three ideas — **verb**, **envelope**, **accessor** — plus one
principle that ties them together. If you're new to building this kind of system, this
section is the on-ramp; the rest of the doc assumes it.

### The one principle: the narrow waist

*One thing does one job, with one implementation, reachable from everywhere.* You see it
first in the CLI: **12 verbs instead of 270 `python -m` modules.** The three concepts
below are that same move applied to **actions**, **results**, and **reads**. Whenever you
catch yourself about to write "just open the zarr inline here" or "just parse the args and
do the work in the handler," you're about to duplicate one of these — and the duplicate is
where the next drift bug will live.

### Verb — the unit of *doing* (one what, many hows)

A verb is a named workflow operation: "run detect," "approve this run." The rule: **there
is exactly one implementation of a verb, and it does not know how it was invoked** — CLI,
notebook, agent-over-MCP, or the webKnossos bridge all call the same function.

The enabler is a **typed request** as input, instead of an `argparse.Namespace`:

```python
@dataclass
class DetectRequest:
    recording: str
    model: str | None = None
    apply: bool = False
    force: bool = False

def detect(request: DetectRequest) -> Envelope:   # the WHAT — one implementation
    ...
```

- CLI = a shell: parse argv → build `DetectRequest` → call `detect()` → render the result.
- Notebook/bridge = `detect(DetectRequest("rec_042", apply=True))`.

Why a typed request, not a `Namespace` or loose kwargs? The request object **is the
contract** — it states exactly what the operation needs, it autocompletes and type-checks,
and any caller can build one. A `Namespace` is opaque and only argparse can build it.
Principle name: **one what, many hows.** The verb is the *what*; front-ends are the *hows*;
you never duplicate the *what*.

### Envelope — the uniform *shape of the answer* (HTTP for your pipeline)

Every verb returns the **same structure**, regardless of what it did:

```python
{
  "command": "detect",
  "status": "ok | blocked | failed | dry_run",
  "reason_code": "DETECT_COMPLETE",
  "recording": "...", "run": "...",
  "artifacts": [...], "metrics": {...},
  "next_hints": ["palette crop ..."],
  "provenance": {"git_sha": ..., "config_hash": ...},
}
```

`detect`, `approve`, and `status` all return this shape — not three different return
types. Three payoffs:

1. **Uniform handling.** Any caller processes any verb's result the same way: check
   `status`, read `reason_code`, follow `next_hints`. No per-verb parsing.
2. **Errors are data, not exceptions.** A blocked operation *returns* `status: "blocked"`
   with `next_hints` — it does not throw. The failure is a value you can inspect, and it
   **carries the instruction to fix itself**. This is what lets a small model recover: it
   can't act on a Python traceback, but it can follow a `next_hint`.
3. **Stable, extensible contract.** Adding a field (we added `provenance`,
   `run_resolution`) breaks no caller — they read the fields they know.

The analogy: **the envelope is HTTP for your pipeline.** Every HTTP response has the same
envelope (status + headers + body) whether it's a GET or a 404; nobody writes per-endpoint
response parsing. `status: blocked` is your `409`; `next_hints` is the response body
telling the client what to do.

### Accessor — the unit of *reading* (resolution in one place)

Verbs *do*; accessors *read*. The `Recording` accessor is a handle that hides the storage:

```python
rec   = open_recording("2026-06-23T16-01-09Z_arena_1_RedScare")
masks = rec.subject_masks()      # RLE-decoded, layout hidden, authoritative run resolved
kps   = rec.keypoints(run="...") # or an explicit run
```

Why not open the zarr directly (as most current code does)? Three reasons, each mapping to
a bug class already seen in this repo:

1. **Hides layout.** The consumer needn't know the zarr group structure or RLE encoding.
   Today every consumer re-derives it, so the knowledge is duplicated and drifts.
2. **Answers "which run?" once, correctly.** Run resolution lives in *one* place
   (authoritative-first), so every reader agrees on "current" — and agrees with what verbs
   write. Re-implementing this per consumer is exactly what produced the false-stale bug.
3. **One place for correctness.** Decode contract, RLE decode, run resolution — fix once,
   every consumer benefits (vs. the silent-wrong-data fix, which had to touch seven files).

Underneath is **Command-Query Separation**: operations that *change* state (commands =
verbs, with dry-run/apply/provenance/blocking) stay separate from operations that *read*
state (queries = accessors, always pure and safe). Don't conflate them — a `get_or_create`
that sometimes writes is a classic bug nest.

### How they compose — a self-guiding loop

```python
rec  = open_recording(rec_id)                     # ACCESSOR: read state
plan = plan_verb(PlanRequest(rec_id))             # VERB → ENVELOPE: what's next?
env  = detect(DetectRequest(rec_id, apply=True))  # VERB → ENVELOPE: do it
# env.next_hints -> ["palette crop ..."]           # the envelope names the next verb
masks = rec.subject_masks()                       # ACCESSOR: read the result
```

You **read** with the accessor, **act** with a verb, get an **envelope** whose
`next_hints` points to the next verb. Internally a verb *uses the accessor* to read and
resolve runs, so verbs and readers never disagree about "current." And this exact loop is
identical whether the driver is a human at the CLI, a notebook, an agent over MCP, or the
bridge — **one set of verbs, one envelope, one accessor; many drivers.**

Mental model to keep: **verb = do (one impl, many callers) · envelope = the uniform answer
(HTTP-response for the pipeline) · accessor = read (resolution in one place).**

## The abstraction-layer map

Palette's surfaces, graded as of 2026-07-01 (see
`docs/diagnostics/codebase_review_2026-07-01.md` for evidence):

| Layer | What it is in Palette | State |
|---|---|---|
| **L0 — byte-level specs** | zarr layout (`zarr_structure.md`, `shared/zarr/stage_arrays.py`), RLE mask store, flat ROI cache manifest, registry SQLite schema | **Strong, under-enforced.** Already language-agnostic (zarr/SQLite readable from anything). Keep hardening write-time enforcement; the design is done. |
| **L1 — library API** | Public Python functions in `fisheye` | **Effectively nonexistent** (cross-module `_private` imports, no `__all__` discipline, logic buried in `utils/`). Deliberately deprioritized: agents run commands, they don't call functions. |
| **L2 — operational API** | ~270 `python -m` entry points; 167 incantations in `docs/operator_guide/` alone | **Right idiom, wrong cardinality.** The idiom (shared args, JSON event logs, dry-run/`--apply` gates, reason codes, bounded-apply guards) is agent-grade. 270 verbs is a vocabulary, not an API. |
| **L3 — workflow API** | "What do I run next": `registry/stage_catalog.py` (DAG), zarr completion markers (state), registry (inventory) | **All the parts exist, nothing composes them.** The answer currently lives in the maintainer's head and prose docs. |

Conclusion: L0 and the L2 *idiom* — the hard parts — are built. The missing piece is a
**narrow waist**: a single entry point with ~12 verbs and a next-action oracle, thin
shims over what already works.

## Requirements for small-model executability

1. **Few verbs.** ~12 subcommands under one `palette` entry point. Everything else
   becomes internal.
2. **One envelope.** Every command accepts/emits the same shape (`--json`), uniform exit
   codes. The existing JSON-event-log + reason-code idiom, codified.
3. **Errors are instructions.** Precondition failures name the missing prerequisite
   *and the command that satisfies it*. A frontier model debugs; a small model follows.
4. **A next-action oracle.** `palette status` / `palette plan` read catalog + completion
   markers + registry and state what is valid next. This is the highest-leverage piece.

## The verb set (v1)

Extracted from what `docs/operator_guide/` actually instructs. Each verb is a **thin
shim** over the existing module — routing plus envelope normalization, no logic rewrite.
(Shim targets marked *resolve* need confirmation during implementation; do not trust
this table over the operator guide.)

| Verb | Shims to | Notes |
|---|---|---|
| `palette status <rec>` | new (read-only) | Per-stage state from completion markers + registry. |
| `palette plan <rec>` | new (read-only) | Valid next actions from `stage_catalog` deps × current state. Recommends; never executes. |
| `palette import <src>` | sampled/analysis import runners (*resolve*: two flavors — expose as `--profile training\|analysis`, not two verbs) | |
| `palette detect <rec>` | `fisheye.utils.run_detect_with_registry_model` | |
| `palette crop <rec>` | crop runner (*resolve*) | |
| `palette keypoints <rec>` | `fisheye.utils.run_keypoints_with_registry_model` | |
| `palette subject-masks <rec>` | `fisheye.utils.run_sam_subject_masks` (SAM3) / UNet inference (*resolve*: `--method sam3\|unet`) | |
| `palette refine-subject-masks <rec>` | `fisheye.refinement.refine_subject_masks` | |
| `palette register <path>` | `Registry.register_from_root` via the maintenance CLI | |
| `palette submit <verb> <rec>` | the `scripts/submit_*_bsub.sh` wrappers | Cluster execution of any run-verb; keeps bounded-apply guards. |
| `palette verify <rec>` | contract/QC checks (*resolve*: which `check_*` scripts graduate) | |
| `palette export-training <rec>` | training-zarr exporters (*resolve*) | |

Rules:
- **The stage catalog is the only DAG.** `plan` derives from `stage_catalog.py` +
  completion markers; no hand-maintained stage table may be added (the graph is already
  defined three times — this must not become a fourth).
- Every run-verb supports `--dry-run` (default) and `--apply`, inheriting the existing
  gating idiom.
- Run-verbs route to the **live path** (the same runners the bsub scripts call), never
  to the legacy `core/pipeline.py` orchestrator.

## The envelope

Every command with `--json` emits exactly one JSON object on stdout:

```json
{
  "command": "detect",
  "status": "ok | failed | blocked | dry_run",
  "reason_code": "DETECT_COMPLETE",
  "recording": "2026-06-23T16-01-09Z_arena_1_RedScare",
  "run": "detect_registry_20260701_01",
  "artifacts": ["zarr/..._training.zarr/detect_runs/detect_registry_20260701_01"],
  "metrics": {"rows": 18234, "duration_s": 412.0},
  "next_hints": ["palette crop 2026-06-23T16-01-09Z_arena_1_RedScare"],
  "provenance": {"git_sha": "...", "fisheye_version": "...", "config_hash": "...",
                  "params": {"model": "...", "set_id": "..."}}
}
```

- Exit codes: `0` ok/dry_run, `1` failed, `2` blocked (precondition unmet), `3` usage.
- `status: blocked` **must** include the unmet precondition and the exact command that
  satisfies it in `reason_code` + `next_hints`. This is the errors-are-instructions rule.
- **The `provenance` block is mandatory on every run-verb** and is stamped into the run's
  zarr attrs at write time. The waist is the natural enforcement point for the
  code-version/config-hash provenance gap (review finding #3): a run created through
  `palette` cannot lack code identity.

## `palette plan` example

```json
{
  "recording": "2026-06-23T16-01-09Z_arena_1_RedScare",
  "stages": [
    {"stage": "import",   "state": "complete", "run": "import_20260623_01"},
    {"stage": "detect",   "state": "complete", "run": "detect_registry_20260701_01"},
    {"stage": "crop",     "state": "missing"},
    {"stage": "keypoints","state": "blocked_by: crop"}
  ],
  "next": [
    {"action": "palette crop 2026-06-23T16-01-09Z_arena_1_RedScare",
     "cluster": "palette submit crop 2026-06-23T16-01-09Z_arena_1_RedScare"}
  ],
  "stale": []
}
```

`plan` also surfaces staleness (a stage whose upstream was invalidated per the catalog's
`invalidates` edges) — recommending re-runs, never performing them.

## Acceptance: the operator-guide test

An agent given **only** `palette --help` output and `palette plan` responses — no repo
access, no operator guide — can take a fresh recording to analysis. Every failure of
this test identifies workflow knowledge still trapped in prose. The operator guide
shrinks as the API grows; `pipeline_workflow.md` at ~3 lines is the done signal.

Practical benchmark: run the test with Haiku as the driver. Where Haiku fails but
Sonnet succeeds, the gap is usually an error message that describes instead of
instructs — fix the message, not the model.

## Non-goals and constraints

- **Not a new orchestrator.** The waist routes and reports; `plan` recommends. No
  auto-execution of chains. Palette's real orchestration (per-stage LSF jobs coordinated
  by completion markers) is a sound cluster pattern and stays. If/when the deferred
  orchestrator-convergence decision is made, the waist is where it lands — until then
  the waist must not quietly become orchestrator #4.
- **No logic migration.** v1 shims call existing runners. Rewrites happen behind the
  waist later, invisibly to callers.
- **L1 (library API) stays deprioritized.** Public-function discipline is a separate,
  later effort behind import-linter contracts.
- **MCP comes after the waist, not instead of it.** Wrapping 12 enveloped verbs as MCP
  tools is a weekend; wrapping 270 modules is the same problem over a new protocol.

## Sequencing

1. **`pyproject.toml` console script** (`palette = fisheye.cli.palette:main`) — lands
   with the packaging work already in flight.
2. **`status` + `plan`** — read-only, new code, immediately useful to humans and agents
   alike; also the proof that catalog + markers + registry actually compose.
3. **Envelope helper + first three run-verbs** (`detect`, `crop`, `keypoints`) with
   provenance stamping; iterate the errors-are-instructions wording against a real
   small-model driver.
4. **Remaining verbs; operator guide rewritten to reference `palette`**; guide shrinks.
5. **MCP server over the waist** — language/model-agnostic access.

## Risks

- **Fourth-orchestrator drift** — mitigated by the route-and-report constraint above;
  review any `plan` change that adds execution.
- **Shim rot** — if runners' flags drift, shims break silently. Mitigation: each verb
  gets a smoke test in CI asserting the shim still constructs a valid invocation.
- **Two doc sources of truth during transition** — operator guide vs `--help`. Rule:
  when a verb lands, its guide section is replaced by the `palette` command in the same
  commit.

---

## Extension (2026-07-03): the library API — programmatic invoke

The waist so far is a *CLI* for agents and operators. A second consumer class needs the
same workflows **programmatically** — the review-backend→`palette approve` wiring, the
webKnossos bridge, the marimo notebooks, and eventually colleagues. Same philosophy, new
front door: **one thin, typed, contract-carrying surface over the stable internals** —
now for `import fisheye` as well as `palette`.

### Current state (updated 2026-07-04 — see the scoreboard for the source data)

Status snapshot below is a pointer, not a duplicate ledger — for the live count of
what's landed, check the "Remediation delta" entries in
docs/diagnostics/codebase_review_2026-07-01.md rather than this section.

- **Move #1 (`verb(request) -> Envelope`) is done, verified directly in
  `src/fisheye/cli/palette.py`.** `status`, `plan`, `approve`, `detect`, `keypoints`,
  and `crop` are all `def verb(request: <Verb>Request) -> dict[str, Any]`, each backed
  by a typed request dataclass (`StatusRequest`, `PlanRequest`, `ApproveRequest`,
  `DetectRequest`, `KeypointsRequest`, `CropRequest`) and returning the
  `build_envelope(...)` payload. No verb takes `argparse.Namespace` anymore — the CLI
  parser builds a request object and calls the same function the library calls.
- `fisheye/__init__.py` is still ~10 lines: **no curated public surface yet**
  (`fisheye.api` does not exist).
- **No `Recording` accessor merged to `sun` yet** — a `Recording` accessor and a
  `RunResolution` resolver exist on branch `agent/run-resolution-accessor` but are not
  an ancestor of `sun` HEAD (verified via `git merge-base --is-ancestor`); every
  consumer on `sun` still opens raw zarr and re-derives run resolution + RLE decode
  itself.

So the remaining gap is the `Recording` accessor and the curated `fisheye.api`
surface, not the request/verb decoupling — that part is done.

### The design: promote the waist to be the library (do not build a parallel API)

1. **Decouple verbs from argparse — `verb(request) -> Envelope`.** Lift the input side
   off `Namespace` onto a typed request object (dataclass per verb, or one request type
   with per-verb fields). The CLI becomes: parse argv → request → verb → envelope →
   render. The library becomes: build request → verb → envelope. One implementation, two
   front-ends. Small, because the envelope-return half already exists. This is the
   highest-leverage move.

2. **A `Recording` accessor.** `fisheye.open_recording(path_or_id)` → a read handle with
   `.detections(run=…)`, `.subject_masks(…)`, `.keypoints(…)`, resolving
   **authoritative-first** (composes with the authoritative-run pointer work) and hiding
   zarr layout + RLE decode. Biggest ergonomics win for programmatic consumers. Read-only
   first; a mutating counterpart later if needed.

3. **A curated `fisheye.api` surface** re-exporting the ~15–20 things a consumer needs:
   `open_recording`, the verbs, the request/envelope types, the stage catalog. The thin
   public front door — the `import`-side equivalent of `palette --help`.

### Explicitly deferred (needs layering debt paid first)

Broad public-API hygiene — `__all__` across ~790 files, eliminating the ~61 `_private`
cross-module imports, the `Any`→typed sweep. A public surface laid over a tangled
internal graph is a facade that leaks (consumers reach past it). Do the thin curated
surface now; the deep cleanup rides on the import-linter/layering work later.

### Sequencing and the forcing function

The review-backend→`approve` wiring was the **first real programmatic consumer of a
verb**. It hit the argparse coupling directly, which drove move #1 for one verb
(`approve`), then the `verb(request) -> Envelope` pattern generalized to the others.
Order:

1. **DELIVERED.** Extract `approve` to a callable `approve(request) -> Envelope`; CLI
   wraps it. (Driven by the review-backend wiring.)
2. **DELIVERED.** Generalize the request/verb split to
   `detect`/`crop`/`keypoints`/`status`/`plan` — all six verbs now take typed request
   dataclasses (verified in `src/fisheye/cli/palette.py`).
3. **In flight, not merged to `sun`.** Add `Recording` accessor (authoritative-first,
   read-only) — exists on branch `agent/run-resolution-accessor`, not yet on `sun`.
4. **Not started.** Curated `fisheye.api` re-export surface + a programmatic-usage doc
   section.
5. (Later, gated on layering work) public-API hygiene sweep.

### Risks

- **Two invocation paths diverging** — mitigated because CLI and library call the *same*
  verb function; the CLI is only a parse+render shell. Never let a verb grow
  CLI-only logic.
- **Premature public surface** — exporting internals before they're stable re-creates the
  `_private`-import problem at the package boundary. Keep `fisheye.api` deliberately
  small; add only what a consumer demonstrably needs.
