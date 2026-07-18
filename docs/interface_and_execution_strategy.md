# Interface & execution strategy

<!-- contract-meta
status: proposed
created: 2026-07-03
owner: jeremy
related: docs/palette_cli_narrow_waist_design.md,
         docs/archive/provenance_enforcement_roadmap.md,
         docs/labeling_platform_build_vs_adopt.md,
         docs/orchestration (core/pipeline.py legacy; per-stage bsub live path)
-->

## The core decision

**The primary mode of operation is autonomous execution.** When a recording transfers
from the acquisition rig, the pipeline should run itself to completion with no human in
the loop. **Most "users" therefore never interface with the pipeline at all** — they
acquire data and consume results. Every *human* interface is **supervisory** (monitor,
debug, approve, ask), never the routine path.

This is the opposite of how the system started (a human driving `python -m` incantations,
or the interactive TUI launcher as the front door). The interface question is not "TUI vs
webapp" — it is "what supervises an autonomous pipeline, for whom."

## The unifying principle: everything is a driver over the waist

Because the pipeline is built as `verb(request) -> Envelope` verbs, an oracle
(`palette plan`), and (soon) a `Recording` accessor, **every mode of operation is a thin
driver over the same verbs** — including the autonomous one:

- The **autonomous executor** is a *headless* driver: a watcher triggers on transfer,
  asks the `plan` oracle "what's valid next for this recording," runs the verbs to
  completion, and stops. No new pipeline logic — it drives the same verbs a human would.
- The **human interfaces** are *presentation* drivers: they render the envelope and
  invoke verbs, and exist for the cases the autonomous path can't handle alone (failures,
  approvals, exploration).

None of these grow their own "run detect" logic. That is the whole point of the waist:
the pipeline has one implementation, and autonomous execution is just another face on it.

## The interface tiers

| Tier | Audience | Interface | Purpose | Status |
|---|---|---|---|---|
| **Autonomous executor** | none (the system) | headless watcher over `plan` + verbs | run every transferred recording to completion, unattended | to build — the primary mode |
| **Power-user** | maintainer (you), now | TUI / CLI (`palette`) | direct operation, debugging, one-off runs, exception handling | exists (CLI landed; TUI lazy-imported) |
| **Conversational** | other researchers | MCP server / skill over the waist | "what's the status of my recording?", occasional triggers, natural-language driving | sequenced (narrow-waist doc step 5) |
| **Operator dashboard** | future operators | thin web status/plan dashboard (reuse labeling-web survivors) | fleet monitoring + approval, click-to-run | future |
| **Labeling / annotation** | annotators | webKnossos (adopt) | mask/bbox/keypoint annotation | see build-vs-adopt doc |

Notes per tier:
- **TUI stays — it's *your* interface, and it's cheap.** Now lazy-imported and thin over
  verbs, it costs almost nothing to keep. Do not invest in making the TUI nicer *for
  other people*; that effort belongs to the waist and the thin layers below.
- **Conversational (MCP/skill) is the highest-leverage human interface for non-terminal
  users** — the payoff of the `plan` oracle + errors-as-instructions work. A researcher
  describes what they want; an agent (even Haiku-class) drives the verbs via `next_hints`.
  The "GUI" for someone who doesn't want the terminal is a conversation.
- **Operator dashboard is *thin and reused*, not a new app.** Build it over the labeling
  web app's surviving dashboard routes (the strangler survivors): read `plan`, render the
  stage grid, buttons POST to verbs and show the returned envelope. Zero pipeline logic.
- **Do not build a bespoke desktop GUI or a fourth full web app.** As a solo maintainer
  the failure mode is maintaining N interfaces, not picking the wrong one. Consolidate:
  autonomous by default, TUI for you, MCP for researchers, one thin dashboard for
  operators, webKnossos for labeling — all over the same verbs.

## What autonomous execution requires (and what it raises the stakes on)

The autonomous executor needs the pieces the waist work has been building — plus two new
ones, and it *sharpens the urgency* of provenance enforcement:

- **The `plan` oracle** (what's valid next) — ✅ exists. This is the executor's brain: it
  asks `plan`, runs the returned next verb(s), repeats until done or blocked.
- **Fail-closed completion markers** (know when a stage is truly done) — ✅ exists. What
  makes it safe to run unattended and resume after interruption.
- **The verbs** (executable operations with dry-run/apply, blocked/failed envelopes) —
  ✅ mostly exists. `status: blocked/failed` is how the executor knows to stop and
  escalate instead of charging ahead.
- **Provenance enforcement at finalization** — ⚠️ *this becomes critical, not optional.*
  A human running `palette detect` gets stamping via the waist; an unattended daemon has
  no one to remember. If provenance is not **enforced at the finalization layer** (see
  `docs/archive/provenance_enforcement_roadmap.md` Slice 2), autonomous runs will silently lack
  code identity. **Autonomous execution is the strongest argument for that enforcement
  slice** — it removes the human backstop entirely.
- **A transfer trigger / watcher** — 🆕 to build. A daemon (or cron, or an inotify/queue
  watcher) on the acquisition transfer location that registers the new recording and hands
  it to the executor. Keep it dumb: detect-new → register → hand to executor; all the
  "what to run" logic stays in `plan`.
- **Failure surfacing** — 🆕 to build. When an autonomous run returns `blocked`/`failed`,
  it must reach a human — a notification, a review queue, a dashboard row. This is where
  the supervisory interfaces earn their place: they are the "something needs attention"
  surface for an otherwise hands-off pipeline. The envelope already carries the reason and
  `next_hints`; the executor just needs to route them.

## Where the orchestrator convergence lands

The deferred orchestrator decision (three coexisting orchestrators: legacy
`core/pipeline.py`, the subprocess `run_recording_analysis_pipeline.py`, and the per-stage
bsub submitters) **resolves here.** The autonomous executor *is* the orchestrator: a
`plan`-driven loop that dispatches verbs — locally or via bsub for GPU stages — to
completion. Building it is the forcing function to retire the legacy in-process
orchestrator and unify on "the stage catalog + `plan` decide; the executor dispatches."
Do not build the executor as a fourth orchestrator; build it as the *convergence* of the
existing ones, driving the waist.

## Discipline

- Every interface — autonomous, TUI, MCP, dashboard — is a **thin driver over the verbs**
  and renders the **envelope**. The moment one grows its own pipeline logic, the system
  has forked into N implementations.
- The `plan` oracle is the single source of "what runs next," used identically by the
  autonomous executor and every human interface. There is no second copy of the DAG.
- Provenance and completion markers are what make unattended execution *safe*; enforce
  them at finalization so no path — human or daemon — can skip them.

## Sequencing (not urgent; forward-looking)

1. **Provenance enforcement at finalization** (`archive/provenance_enforcement_roadmap.md`
   Slice 2) — precondition for trustworthy unattended runs; do before autonomous execution
   goes live.
2. **The headless executor** — a `plan`-driven loop over the verbs, resumable via
   completion markers. Start by wrapping the existing live path; it becomes the
   orchestrator convergence.
3. **The transfer watcher** — detect-new-recording → register → hand to executor.
4. **Failure surfacing** — route `blocked`/`failed` envelopes to a human (queue /
   notification), reusing the dashboard/MCP surfaces.
5. **MCP server over the waist** — conversational supervision for researchers.
6. **Thin operator dashboard** — over the labeling-web survivors, when operators need a
   visual surface.

## Non-goals

- Not making the TUI a general-user interface (it's the maintainer's power tool).
- Not a bespoke desktop GUI or a new standalone web app.
- Not a fourth orchestrator — the executor is the convergence of the existing ones.
- Not autonomous execution before provenance is enforced at finalization.
