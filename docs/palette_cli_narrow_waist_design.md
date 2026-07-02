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
