# A workflow runner that owns Palette's cluster graph

- **Status:** draft; direction decided by Jeremy on 2026-10-07 (§10): Snakemake, a separate env, a login-node contact budget, intake only. Nothing is implemented, and no runner has been installed.
- **Owner:** Jeremy Delahanty. Drafted by the `cluster-runner` agent at the request of `palette-a0` (intake/merge queue).
- **Last reviewed:** 2026-10-07, against `main` at `98f458e9`.
- **Builds on:**
  - `docs/lsf_submission_framework_design.md`: the shared LSF kernel and the family planners;
  - `docs/production_dag_recording_layout_design.md`: the kernel as the production DAG engine;
  - `docs/diagnostics/pipeline_model_and_correction_loop_2026-09-02.md`, checklist N, especially N-1 and N-2;
  - `docs/workflow_provenance_prior_art.md`: the engine survey, recommendations not adopted;
  - `docs/cluster_job_dashboard_direction.md`: a dashboard that was designed and never built;
  - `docs/zarr_run_completion_contract.md`.
- **Why now:**
  - Jeremy wants to stop driving the cluster through an LLM agent invoking scripts. He wants something he can start, watch and resume himself.
  - The new v2 intake path (poller, LSF import, workstation registrar) is about to go live on cron. It already shows the failure mode a runner is for: a job that dies without writing a status file reads as "pending" forever.

## Summary

- **Recommendation:** use Snakemake (8 or later) as a *supervisor* above Palette's existing planners and evidence layer, not as a replacement for either.
  - Snakemake owns the dependency graph, scheduling, retries, keep-going, resume and the operator's view.
  - Palette keeps sole ownership of what "done" means. A job's Snakemake output is a small sentinel file, and it is written only after Palette's own completion and receipt checks pass.
  - Snakemake never declares a Zarr path as an output.
- **LSF interaction:** Snakemake runs only on ws1, and every rule is local. A cluster step is a local rule that submits one job through Palette's existing `fisheye.cluster.lsf.backend` and then waits on NFS evidence (`fisheye.flow.lsf`). This avoids depending on any LSF executor plugin. (Changed at implementation, 2026-10-08: the cluster-generic plugin would run Snakemake inside every LSF job, and the `palette-flow` env lives on ws1's local disk; see the decision log.)
- **Controller host:** the controller runs on `delahantyj-ws1`, the registry writer host. Registry writes therefore become ordinary local rules on the one host allowed to write. Cluster jobs are submitted over SSH to `login1-citrus-poller`, as today.
- **First slice:** the intake DAG (import per delivery on LSF, then register on ws1). It replaces the cron poller/registrar pair and its hand-rolled state files.
  - **Second slice:** a generic "plan executor" that runs any `LsfWorkflow` plan emitted by an existing family planner. This is how most of the remaining launchers migrate without rewriting their domain logic.
- **Weakest point:** monitoring. Snakemake has no first-class live web UI. Jeremy would get a terminal view, logs, a post-hoc HTML report, and a small Palette status page that we would build (§7).
  - If a polished live UI is the deciding criterion, Nextflow with Seqera Platform is the stronger choice, at the costs listed in §3.2.
  - This is the one decision in the doc that I think is genuinely Jeremy's (see §10, Q1).

## 1. What exists today

Palette's cluster work has three layers. They are easy to conflate, and the engine question only concerns the third.

| Layer | What it is | Engine-replaceable? |
|---|---|---|
| **Evidence** | Zarr run-completion attrs (`zarr_run_completion`), `run_provenance`, import/admission receipts, `stage_complete` gate, selectors, the registry and its single-writer gateway (`registry.shadow_publish`) | No. No engine has row-level identity, review/correction state or Palette's authority rules (prior-art doc §"what is not reinvention"). |
| **Planning** | Family planners that choose targets, models, explicit run names and resources, and emit a typed `LsfWorkflow` (`fisheye.cluster.*`, about 20 callers of `submit_lsf_workflow`), plus the analysis-workflow DAG planner (`analysis_workflows/dag.py`) | Mostly no. This is domain logic, and the engine would call it. |
| **Execution/supervision** | ~70 `scripts/submit_*_bsub.sh`, `cluster/lsf/submission.py` (submit the whole graph to LSF with `done()` dependencies, then exit), cron pollers, per-launcher state files, and an agent watching `bjobs` | **Yes.** This is what Jeremy wants replaced. |

### 1.1 The supervision gap, concretely

`fisheye.cluster.lsf` is a good typed DAG compiler. Its own design doc, though, lists as *non-responsibilities* exactly the things a runner provides (`lsf_submission_framework_design.md`, "Kernel non-responsibilities"): no retries, no resume of partial workflows, no cancellation, no live inspection. A compiled graph is handed to LSF and nothing watches it afterwards. In practice:

- A failed node leaves its `done()` dependents pending forever (LSF `PEND` with an unsatisfiable dependency) until someone notices and runs `bkill`.
- Recovery means a hand-written recovery planner per family (`clipped_inference_{import,detect_quality,keypoint}_recovery.py`) or an agent improvising one.
- Cross-launcher dependencies, such as intake to detect or masks to kinematics to analytics, are not in any graph. They are either a human running the next script or `--dependency-done JOBID` passed by hand.
- There is a known crash window between `bsub` acceptance and persisting the job ID (`submission.py:95-142`, review-wave second opinion).
- Nobody queries LSF state after submission. The only `bjobs` call in the repo is a dedup check in the intake launcher.
- Intake today: the poller writes `.claimed`/`.submitted`, and the registrar globs for a status JSON. A job killed for walltime or OOM, or one that fails before reserving its status file, stays "pending" forever. **There is a live instance of this on `main`.** The launcher passes a `--status-json` inside a `--run-dir` it has already created. `citrus_transfer_parent_workflow.py:171-175` and `:188` reject both before the status file is reserved, so every launcher-submitted import should exit 1 silently. This was found by reading the code and confirmed by `palette-a0`. The fix is PR #286, queued: it gives the workflow a fresh run dir, adds an end-to-end test of the generated job script, and makes the registrar treat "job ended without status" as failed. The cron swap stays blocked until #286 merges.

### 1.2 Launcher inventory (70 scripts)

| Class | Count | Examples | Disposition |
|---|---|---|---|
| Production stage or chain | ~41 (6 are 4-to-56-line wrappers over Python planners) | detect/crop/keypoint/mask batches, `whole_recording_keypoints`, `analysis_workflow`, `chaser_analytics`, cohort exports, `citrus_session_import` | Migrate (§8) |
| Benchmark / canary / A-B | 21 | `subject_mask_finalizer_*`, `*_sharding_benchmark`, `detect_compute_smoke` | Stay scripts; archive when the experiment closes |
| One-off migration / backfill | 5 | `instance_key_backfill`, `immutable_yolo_sharding_migration` | Stay scripts; archive once confirmed done |
| Ops / diagnostics | 3 | `registry_zarr_projection_refresh`, `recording_dish_rim_probe` | Stay scripts |

Other facts that shape the design:

- Only one shared shell helper exists (`scripts/lib/palette_lsf.sh`, 27 lines, sourced by one script). Commit pinning, clean-tree checks, SSH submit routing and run-dir layout are copy-pasted across about 35 scripts.
- Real execution-host pinning appears in only two places: the registry writer's `select[hname==…]` in the intake launcher, and an optional `-m` in one benchmark. Everything else is submit routing (`ssh login1-citrus-poller bsub …`).
- GPU stages request `gpu_l4` with `-gpu num=N`. Wide work uses LSF arrays (`name[1-N]%M`) or in-allocation bundles (`task_group.py`).

## 2. Requirements

A runner is acceptable only if it meets all of these.

1. **One definition of done, and it is Palette's.** The runner may cache Palette's verdict but must never contradict it. It must not decide a stage is done because an exit code was 0. `emit_stage_completion` currently swallows gate failures and returns `False` (`stage_complete.py:488`), so exit 0 is not yet evidence.
2. **It never deletes or moves Palette data.** Run groups live inside shared per-recording Zarrs. A runner that "cleans up outputs of failed jobs" must never be pointed at them.
3. **Immutable, explicitly named runs.** Every job's output run name is decided before submission. A retry after a failure uses a new run name, because failed runs are permanent tombstones (`zarr_run_completion_contract.md`).
4. **The single registry writer.** Registry mutation happens only on the writer host and only through `shadow_publish`.
5. **Commit pinning.** One workflow invocation runs from one commit-pinned deployment (`deploy_palette_cluster_worktree.sh`). The runner must not make it easy to point running work at a newer checkout.
6. **LSF-native resources.** Support queues, `-gpu`, memory, walltime, host select, and preferably arrays or bundles for fan-outs in the hundreds.
7. **Jeremy can operate it without an agent.** One command to start, one to see status, one to resume. Failures are visible and named.
8. **Bounded storage.** `/groups/johnson` is about 94% full. No engine-managed scratch copies of data.
9. **Python-maintainable** by a solo developer and LM agents, with fake-runner unit tests like the current kernel.

## 3. Options

### 3.1 Snakemake (8 or later)

- **Model:** file targets. A rule runs if its declared output files are missing or stale relative to its inputs. Executors are plugins (v8 or later).
- **Fit for Palette**
  - Python all the way down. Input functions can call `fisheye` directly to resolve targets from the registry, a manifest or a planner's `plan.json`. That is the decisive practical advantage for an LM-maintained Python codebase.
  - "Done = an output file exists" maps cleanly onto "done = Palette wrote a sentinel after its own checks passed" (§4). The runner then has no independent cache to disagree with.
  - `localrules` run on the controller host. With the controller on ws1, registry writes become ordinary rules that cannot run anywhere else.
  - Retries (`--retries`, with an `attempt` number available to params, which gives attempt-scoped run names), `--keep-going`, and dry run with a DAG/rulegraph print.
- **LSF executor**
  - `snakemake-executor-plugin-lsf` exists. It is community-maintained, by one author as far as I can tell, at v0.2.x. I have not verified its array support, `-gpu` rendering or how often it polls `bjobs`.
  - ~~Instead, recommend `snakemake-executor-plugin-cluster-generic` …~~ Superseded at implementation (see the decision log). That plugin's job script starts a second, single-job Snakemake inside every LSF job, so the env and Snakefile must be visible on compute nodes, and `palette-flow` is on ws1's local disk. Instead, cluster steps are local rules on ws1 that submit through `fisheye.cluster.lsf.backend` and wait on NFS. Snakemake owns only the graph.
- **Hazards that must be designed around** (none is a blocker):
  - Snakemake **removes the declared outputs of a job before rerunning it, and after a failure**. `directory()` outputs are removed wholesale. Rule: no Zarr path is ever an output. Outputs are sentinels only, enforced by a test over the Snakefile.
  - Default `--rerun-triggers` include `code`, `params` and `software-env`, so a new commit would invalidate every finished sentinel. Use `--rerun-triggers mtime` (sentinels are written once and never touched) and record the commit inside the sentinel instead.
  - **Snakemake does not re-attach to running cluster jobs after the controller dies.** On restart, those jobs are "incomplete" and get resubmitted, which could put two writers on one Zarr run. The job wrapper must take an exclusive claim keyed on `(job_key, attempt)` and refuse to start while a live LSF job holds it (§5.3). Of everything in this design, this needs the most care.
  - Dynamic fan-out, where the number of parents is known only after organize, needs `checkpoint` rules, which are workable but clunky. The slices below avoid them.
  - There is no array-job mapping: each Snakemake job is one `bsub`. Fan-outs of hundreds of tiny tasks should stay inside one Snakemake job as a bundle, or be batched with `group:`.
- **Monitoring**
  - Live: the terminal progress log, plus whatever we read from the sentinel and status files.
  - After the fact: `snakemake --report` (HTML with runtimes, rule graph and provenance) and `--summary`/`--detailed-summary`.
  - Third-party live monitors exist (e.g. Panoptes via `--wms-monitor`). I would not depend on any of them without checking that they are maintained. A Palette status page is planned instead (§7).
- **Cost:** a new dependency tree. Installing it needs Jeremy's explicit approval (AGENTS.md); recommend a separate `palette-flow` conda env rather than `palette-py311`.

### 3.2 Nextflow

- **Model:** dataflow over channels. Each task runs in its own `work/` directory with inputs staged in. The resume cache is keyed by a hash of the task script, inputs (path+size+mtime by default) and params, and stored in `.nextflow/cache`.
- **Strengths**
  - The most mature LSF executor of the options here, maintained by Seqera, with first-class arrays via the `array` directive (added in 24.04).
  - Excellent dynamic-DAG ergonomics.
  - `-with-trace/-with-report/-with-timeline` produce good post-hoc evidence.
- **The best UI:** Seqera Platform, for live runs, history, logs, resource charts and relaunch. That requires either:
  - Seqera Cloud, where the controller sends run and task metadata (names, status, resources, paths; not data) to an external service. That is a data-governance question for HHMI; or
  - a self-hosted Seqera Enterprise licence.

  The old open-source Tower is no longer maintained, as far as I know.
- **Fit problems**
  - Its native model is the opposite of Palette's. Nextflow wants outputs produced inside the task work dir and then published. Palette mutates a shared per-recording Zarr in place and records evidence there. You can make Nextflow tasks emit receipts and treat the Zarr as an external side effect, or use `storeDir` to make "a file exists in a store dir" the skip condition. Either way the `-resume` cache becomes a **second** notion of done to keep consistent with Palette's.
  - Groovy/DSL2 and a JVM controller. Domain logic stays in Python, so every planner hands Nextflow JSON. Two languages to maintain, and the place where LM agents are least reliable.
  - The `work/` directory needs a cleanup policy on a nearly full filesystem, although it would hold only small files here.
- **Verdict:** pick this if the live UI dominates and Seqera Cloud is acceptable. Otherwise the fit costs outweigh the executor advantages.

### 3.3 Extend the in-house kernel into a runner

- Add a persisted controller to `fisheye.cluster.lsf`: a reconcile loop over plan, `bjobs`, runtime-status and completion probes; submit what is ready; retry; and a TUI or web status view.
- **For:** zero new dependencies, full domain fit, and it reuses everything.
- **Against:** it means building and maintaining a workflow engine alone, and the reconcile, retry, crash-recovery and UI work is the expensive part. The 09-02 review explicitly said consolidating onto the kernel should not be the default (N-2), and the prior-art doc rated the kernel "good but redundant".
- **Verdict:** this is the fallback if the Snakemake slice fails its exit criteria (§6.4). Note that the slice keeps the kernel's backend, so nothing is lost by trying.

### 3.4 Others, briefly

| Tool | Why not first choice |
|---|---|
| **Cylc 8** | HPC-native, Python, supports LSF, has a real TUI and web UI, and is designed for recurring, cron-like workflows such as intake. But it is built around cycling graphs with its own config language and is weak at data-dependent fan-out. It is the most interesting runner-up for the intake use case specifically. |
| **Dagster / Prefect** | Best Python UIs and a software-defined-asset model that resembles Palette's "materialized run" idea. But each needs a server plus a database, has no native LSF support (`dask-jobqueue` or a custom launcher would be needed), and the asset catalog would compete with the registry as a source of truth. Too much new infrastructure for a solo maintainer. |
| **Airflow** | Heavy service, built for time-scheduled ETL; LSF support is a custom operator. No. |
| **Toil / Cromwell (CWL/WDL)** | LSF-capable, but come with a workflow language and staging model that fit worse than Nextflow's. |
| **Parsl** | Python with a good LSF provider, but it is a parallel-futures library, not a monitored DAG runner. |

### 3.5 Scorecard

Scores: ● strong, ◐ workable, ○ weak.

| Criterion | Snakemake + generic executor | Nextflow (+ Seqera) | Extend kernel |
|---|---|---|---|
| Live monitoring UI | ○ (terminal, plus a page we build) | ● with Seqera / ◐ without | ○ (we build it all) |
| Resume and caching that defer to Palette | ● (sentinel = Palette verdict) | ◐ (`storeDir`/receipts; second cache) | ● |
| Fit with Zarr directories and internal markers | ◐ (sentinels; must never declare Zarr outputs) | ○ to ◐ (work-dir model) | ● |
| Python-native fit | ● | ○ | ● |
| LSF maturity, including arrays | ◐ (our backend; no arrays) | ● | ◐ |
| Single registry writer | ● (local rules on ws1) | ◐ (local executor on ws1) | ● |
| Operator use without an agent | ◐ | ● | ○ until built |
| New infrastructure | conda env | JVM + optional service | none |

## 4. One source of truth for "done"

This is the central design decision, and it is the same whichever engine is chosen.

**Palette decides, and the runner caches the decision as a sentinel.**

```text
<flow_root>/<workflow>/<target>/<job_key>.done.json      # runner output, written once
```

A job's command is always the Palette wrapper (`fisheye.flow.run_job`, new and thin, built on `cluster.lsf.runtime.run_with_status`):

1. Claim `(job_key, attempt)` exclusively (§5.3).
2. Run the stage entrypoint with the planner-assigned run name. On attempt > 1 the name gets an `_a<attempt>` suffix, because failed runs are tombstones.
3. Run the **family completion probe**:
   - For Zarr runs: the run group's `zarr.json` has `palette_run_completion_status == "complete"` and `stage_selector_eligible is true`, read with plain `json.loads` as `analysis_workflows/availability.py:105-133` already does, plus the family validator where one exists.
   - For imports: `load_verified_recording_import_receipt`.
   - For registry rules: re-read the row through the gateway's own validation.
4. Only if the probe passes, atomically write the sentinel containing:
   - schema `palette.flow_job_done.v1`;
   - `job_key`, `attempt`, the run name(s) produced;
   - an evidence digest (receipt sha256, or the run group's `zarr.json` sha256);
   - the deployment path and full commit;
   - the LSF job ID and host.
5. Exit 0 only if the sentinel was written. Exit nonzero if the probe failed, even when the stage exited 0. This closes the N-4/G-1 gap *for runner-launched work* without waiting for the global `stage_complete` fix. That fix should still land.

Consequences:

- **No second cache.** A sentinel is only a memo of a Palette verdict, pointing at its evidence. Rebuilding sentinels from the store must always be possible. `palette-flow reconcile` re-probes the evidence and reports drift: a sentinel whose evidence no longer validates, or evidence that exists without a sentinel. It **reports; it does not delete**. Drift is an incident, not something to auto-heal.
- **Staleness belongs to Palette.** Snakemake runs with `--rerun-triggers mtime` and sentinels are written once, so the runner never invalidates work because code changed. "This should be recomputed with model B" becomes a *new target with a new run name*, decided by a planner. That matches immutable runs and the correction-loop model.
- **Run names must be predictable.** Stages that default to wall-clock names (`detect_yolo._next_run_name`, `refine_detect`, `refine_keypoints`, …) must be called with explicit `--run-name`/`--output-run`. The family planners already do this. Any entrypoint that lacks the flag gets it as a prerequisite edit, tracked per family in §8.

## 5. Architecture

```text
            ┌──────────────── delahantyj-ws1 (registry writer host) ───────────────┐
 Jeremy ──► │ palette-flow run/status/resume  →  snakemake (controller, tmux/systemd) │
            │   localrules: register_*, registry_finalize_*  ──► shadow_publish      │
            │   local rule: fisheye.flow.lsf submit+wait ──ssh──► login1-citrus-poller│
            └──────────────────────────────────────────────────────────────┬─────────┘
                                                                           │ bsub/bjobs/bkill
                                              LSF compute: scripts/py -m fisheye.flow.run_job
                                                 (pinned deployment; claim → stage → probe → sentinel)
```

### 5.1 Components (new code is small)

- **`fisheye.flow.run_job`:** the per-job wrapper from §4. It reuses `cluster.lsf.runtime` for the status envelope, signal forwarding and scratch cleanup.
- **`fisheye.flow.lsf`** (as built): submits one job per attempt through `cluster.lsf.backend` (`build_ssh_bsub_runner`, `parse_bsub_job_id`) and waits on NFS evidence:
  - while a job runs, its job script touches a heartbeat file every 60 s;
  - on exit it writes `result.json` and then `exit_code`;
  - an LSF output footer with no `exit_code` means the job died.

  Only a job with no fresh heartbeat (pending, or silent) consults the shared `bjobs` cache. If the controller restarts, it re-attaches to a submitted attempt that has no exit record.
  - **Login-node contact budget (Jeremy, 2026-10-07): no per-job polling of the login nodes; at most one check-in every 5-10 minutes.** Concretely:
    - Snakemake's frequent per-job `status` calls never leave ws1. `status` reads only NFS evidence: the runtime envelope's running/final status JSON, and as a fallback the LSF `<job_id>.out` "Resource usage summary" footer (the same "ended" signal #286 uses). "Ended with no status JSON" means **failed**, not pending.
    - Only jobs with no file evidence (still `PEND`, or killed before the envelope started) need LSF itself. These are covered by **one batched `bjobs` call for all such jobs, at most once per 5 minutes**. The result is cached on ws1 (`<flow_root>/lsf_state.json` with its timestamp), and every `status` call reads that cache.
    - The batched check is rate-limited by a ws1-local lock plus a timestamp, so concurrent controllers or `palette-flow status` cannot exceed the budget.
    - Other login-node contact is limited to `bsub` at submission and `bkill` at cancel, which are one call per job, not polling.
    - All contact goes through a single SSH ControlMaster connection. No SSH session is opened per check.
- **`fisheye.flow.probes`:** family completion probes, each a thin call into an existing validator. No new validators.
- **Snakefiles:** under `workflows/` (new top-level directory), one per recipe: `intake.smk`, `plan_executor.smk`, later `recording_analysis.smk`. Profiles live under `workflows/profiles/`.
- **`palette-flow` CLI:** a thin shell wrapper. It resolves the deployment, takes a `flock`, and runs `snakemake --profile …`. Subcommands: `run`, `status`, `resume`, `reconcile`, `report`.

### 5.2 Controller placement

- The controller runs on ws1, because the registry writer must be ws1 and because Jeremy works there.
- **Cost:** ws1 uptime becomes workflow uptime. If ws1 reboots, LSF jobs keep running and the controller resumes them on restart, protected by the claims in §5.3.
- **Alternative:** run the controller on `login1-citrus-poller` and hand registry steps to a ws1-side registrar. That keeps today's two-hop split and loses the main simplification. Only do it if Janelia forbids the SSH-polling pattern or ws1 availability proves inadequate.

### 5.3 Claims and the restart hazard

- On start, `run_job` creates `<flow_root>/claims/<job_key>.a<attempt>.claim` with `O_EXCL`, holding its LSF job ID and host.
- If the claim exists and its LSF job is still alive (judged from its runtime-status heartbeat on NFS, falling back to the cached batched `bjobs`), the new job exits with a distinct code. The executor's status check treats that code as "attached to the existing job" and polls the original job ID.
- If the claim's job is dead, the attempt counts as failed. The next attempt runs under a new run name.
- This is the same O_EXCL-claim pattern the intake poller already uses (`.claimed`), moved to the job level.
- **If a stage already owns an exclusive lock, the runner defers to it and adds no claim of its own.** Two claim mechanisms for one fact would be the duplication the intake single-writer design (PR #290) removes. Intake already holds a per-snapshot workflow lock, and `import_delivery` reports "held by another live job" with its distinct exit code. The runner maps that code to "attached" and takes no claim. Runner claims are only for stages that lack their own lock.
- **Exit codes (from PR #290):**
  - 0: published, and the probe is true; the runner writes the sentinel.
  - 65: refused. The runner does not retry; it reports an operator incident.
  - 75: held by a live holder. The runner records "attached" and does not retry within this invocation.
  - 1: failed. The runner retries up to its retry limit.
- **The intake lock is an NFSv4 `fcntl.flock`.** It is released when its holder exits, or about 90 s after a holder host dies, when the NFS lease expires. A delivery that answers 75 therefore needs no special recovery: the next cron tick, 10 minutes later, re-runs discovery and either finds the original job's evidence or takes the lock itself.

### 5.4 Deployment pinning

- `palette-flow run` requires `--palette-repo` to be a dedicated commit-pinned deployment (AGENTS.md).
- The Snakefile and every job run from that deployment's `scripts/py`. The commit is recorded in each sentinel.
- `resume` refuses if the deployment's HEAD differs from the commit recorded at `run`. Moving to a newer commit means a new invocation, and existing sentinels still count because they are Palette verdicts rather than code hashes.
- **Steps bound to the producer commit.** Some steps must run at the same commit that produced their input, not at the current pin. Intake registration is the first case: `recording_identity_authority` refuses registration unless the registering commit equals the import receipt's `producer_git_sha`, exiting 65 with `registrar_commit_mismatch` and naming the commit it needs. Pinning one commit for both sides and moving it only when no delivery sits between import and registration would work. With deliveries arriving continuously, though, that turns every deployment into a drain-and-wait. Instead:
  - `probe-import` output, and therefore the import sentinel, carries `producer_git_sha`;
  - `register_delivery` runs from the ws1 deployment at that commit (`~/.palette/deployments/ops-<sha>`), not from the current pin;
  - discovery and new imports always use the current pin;
  - moving the pin is safe at any time, except when the target adds registry migrations (2026-10-10, below). An older deployment is retired only when discovery shows no delivery whose `producer_git_sha` needs it;
  - **A move that adds registry migrations must drain first.** The first registration by new code migrates the live registry. Deliveries imported by older code would then register through code that predates the migration: the gateway's schema guard refuses non-additive migrations, and an additive one silently skips the new projections. `palette_flow.sh pin-check intake --config … --to-ops … --to-lsf …` refuses (exit 65) while any delivery imported by older code is still unregistered, counting both runner sentinels and cron-path imports from discovery. It also refuses ws1 and LSF targets that are at different commits;
  - if that deployment is missing, the rule fails as an operator incident. The runner never creates or moves deployments on its own.

## 6. First slice: the intake DAG

**Why intake first:**

- It is small: one LSF rule and one local rule.
- It is about to go live on cron.
- It has a real "pending forever" gap today.
- Its registry write naturally belongs on ws1.
- It replaces state-file bookkeeping rather than scientific code, so its contract risk is low.

**Prerequisite:** PR #286 (launcher run-dir/status-path fix and an end-to-end job-script test, §1.1) is merged and green. The runner slice builds on that tested contract and reuses its "ended without status means failed" rule in the executor's `status` command.

### 6.1 Graph

```text
discover (DAG-build time, on ws1)
  targets = sealed v2 markers under staging_dir                     (check_marker, unchanged)
          ∪ <dest_root>/.transfer_intake/<snapshot_sha>/organization_state.json
              at reserved | materialized | retiring                    (incomplete: resume)
              or complete with probe_register(sha) false               (register only)
for each snapshot_sha:
  import_delivery     [LSF short, 1 core, 4 GB, 1h]
      fresh:  run_citrus_session_import --run-dir <flow run dir> --apply   (no --register)
      resume: … --resume-transfer-plan <plan taken from state["plan"]>
      probe: state == complete ∧ every planned zarr's import receipt verifies
      → intake/<snapshot_sha>/import.done.json
  register_delivery   [localrule on ws1]
      refuse synthetic data_origin (read from state["plan"])
      load_verified_recording_import_receipt for every zarr
      ONE publish_registry_shadow mutation synchronizing all of the delivery's zarrs
      probe: registry rows present and matching receipt digests
      → intake/<snapshot_sha>/register.done.json   (1:1 with one registry publication)
```

- **The target set must include durable intake states, not only markers.** Staging, including the marker and snapshot, is deleted during finalization, and a failed retirement leaves the state at `retiring` with the marker possibly already gone. `finalize_transfer_staging` tolerates missing files once retirement has begun. A marker-only scan would orphan half-retired deliveries and imported-but-unregistered ones. (Correction from `palette-a0`, 2026-10-07.)
- **Discovery never reads runner sentinels.** "Complete but not registered" is decided by `probe_register`, which reads Palette evidence. If it read the sentinel, the runner's cache would become an authority.
- **Each attempt gets a fresh run dir:** `<flow_root>/intake/<sha>/attempt-<n>/`. The workflow requires a run dir that does not exist yet. Intake retries resume from durable state rather than minting new runs, so the attempt-suffixed run names in §4 don't apply to intake.
- **Resume from the stored plan, not from staging.** For an incomplete state, the exact plan is `state["plan"]` (also in `<run_dir>/organization_plan.json`). There may be no marker left to rebuild it from, so the import rule passes `--resume-transfer-plan`.
- **The registration mode is fixed per delivery.** On its first attempt, finalize records an `admission_contract` (`registry_path`, `require_stimulus`) and refuses a retry that changes it. Runner imports always use workstation mode (`registry_path=None`). A delivery first attempted in the old "job registers" mode cannot be resumed by the runner. Discovery reports it as `legacy_mode` and leaves it for manual handling, rather than attempting a retry that would be refused.
- **One registry publication per delivery.** Today each `shadow_synchronize_recording_import` call is its own `publish_registry_shadow`: a full backup (about 70 MB), a copy and a publish per zarr. A 4-camera delivery is therefore 4 publications and is not atomic. Slice 1 adds a small gateway function that synchronizes all of a delivery's zarrs inside one mutation: one backup, all-or-nothing. It is still an idempotent upsert keyed by zarr, so a re-run after failure is safe.
- **Measure verification cost first.** `load_verified_recording_import_receipt(verify_current_surfaces=True)` re-checks live surfaces over NFS from ws1. Time it on a real 4-camera delivery before relying on a 10-minute tick. If it is slow, the tick interval follows the measurement.
- Per-parent fan-out stays inside the one import job, as today. Splitting per-parent work into separate LSF jobs would require a `checkpoint` rule and buys nothing at current volumes.

### 6.2 Trigger

- Cron on ws1, every 10 minutes: `flock -n intake.lock palette-flow run intake`. Overlapping ticks exit immediately.
- Building the DAG (scanning `staging_dir` and `.transfer_intake/` on NFS) never touches a login node. A tick with nothing to do makes zero LSF calls.
- Each tick builds the DAG from current evidence, runs what is missing, waits for its LSF jobs, and exits.
- This replaces the pending poller + registrar cron pair. `citrus_transfer_v2_poller.check_marker` and the registrar's per-zarr logic are reused as functions, not deleted.

### 6.3 Migration of existing state

- A one-time `palette-flow reconcile intake --adopt` writes sentinels for deliveries whose `.registered` state and receipts verify. It records each delivery's `admission_contract` mode in the sentinel, so an adopted job-mode delivery is never retried in workstation mode.
- **The runner never reads v1 state.** The v1 poller's `staging/.processing_state` holds 68 legacy v1 submissions. A v2 registrar dry run against that shared directory would have marked all 68 as `import_failed`. So:
  - v2 poller state lives only in `staging/.processing_state_v2` and `.processing_logs_v2` (from #286);
  - runner state lives only under `<flow_root>`;
  - `--adopt` reads only the v2 state directory, plus Palette evidence;
  - discovery ignores v1 markers, as `check_marker` already does;
  - a test fails if any runner or v2 path resolves into the v1 state directory.
- The old `state_dir` files stay read-only as historical evidence.
- After that, the poller's `.claimed`/`.submitted` mechanism is retired. Snakemake's job claims and sentinels replace it, keyed on the same `snapshot_sha`.

### 6.4 Exit criteria (all required before slice 2)

1. The synthetic parent-intake canary (refreshed in #286) runs end to end through `palette-flow`. This includes a case seeded at `retiring` with the marker already gone, which must resume from `state["plan"]` and register.
2. One real delivery is imported and registered with no agent involved. Jeremy starts it and reads its status himself.
3. **Kill tests:**
   - `bkill` mid-import: the run shows failed within one status poll; `resume` retries, and the import resumes from `organization_state.json`.
   - Kill the controller mid-import, then restart: no duplicate import job; the restarted controller attaches through the claim.
   - Registry write failure: the register rule fails, its sentinel is absent, and the next tick retries.
4. `palette-flow reconcile intake` reports zero drift.
5. Jeremy confirms the status view (§7) answers "what is running, what failed, and why" without an agent.

If criteria 3 or 5 can't be met with reasonable effort, stop, write up why, and reconsider §3.2 or §3.3 before migrating anything else.

## 7. Monitoring, honestly

What Jeremy gets with Snakemake, in the order it would be built:

1. **The controller terminal** (tmux on ws1): live job starts, finishes and failures with log paths.
2. **`palette-flow status [workflow]`:** a table built from sentinels, runtime-status JSONs and claims, plus the cached batched `bjobs` state; `palette-flow status` itself never contacts a login node. Columns: target, job, state, attempt, LSF job ID/host, elapsed, log path, failure line. This is the `cluster_job_summary` the June dashboard doc asked for, now with a concrete data source.
3. **`palette-flow report`:** `snakemake --report` HTML for a finished invocation, stored next to the run.
4. **Later, if (2) proves useful:** a read-only status page served like the existing Datasette registry browser, or a sentinel/status table exported into it.

This is less than Seqera Platform gives out of the box. It is enough for one operator with tens to hundreds of jobs per invocation; I expect it to feel thin for multi-day campaigns with thousands of jobs. If, after slice 1, the gap is the thing Jeremy minds most, that is the signal to revisit Nextflow + Seqera. Planners are engine-neutral (§8), so the switch would cost the Snakefiles and the executor scripts, not the domain logic.

## 8. Migration path and retirement order

The unit of migration is a **family**, as in the LSF framework doc, not a script.

| Step | What moves | How | Retires |
|---|---|---|---|
| 0 | — | Merge PR #286 (intake launcher fix); approve the `palette-flow` env | — |
| 1 | Intake | §6 | `submit_citrus_session_import_bsub.sh` (kept only as a manual entry point that calls `palette-flow`), poller/registrar cron pair, `state_dir` bookkeeping |
| 2 | **Plan executor** | `plan_executor.smk` reads any `lsf_plan.json` from an existing planner. Each `LsfJob` becomes one Snakemake job: `done` dependencies become inputs, `ended` dependencies become inputs with a failure-tolerant wrapper, and a `BUNDLE` stays one job. An `ARRAY` initially becomes one job per element under `group:`, which needs a measured check of LSF load. Pilot: `whole_recording_keypoints`, which already has a 40-recording manifest | `submit_lsf_workflow`'s apply path for that family, and the per-family recovery planners as each family moves |
| 3 | Planner-backed families | Switch `whole_recording_analysis`, `whole_video_detection`, `native_detection_campaign`, `arena_geometry_campaign` and `clipped_collection_{keypoints,subject_masks}` to the plan executor, one at a time | Their thin shell wrappers become `palette-flow run plan …` |
| 4 | Shell-only chains (Tier B) | Turn each into a small planner emitting an `LsfWorkflow` (the kernel already supports this), then run it through the plan executor: detect artifact → quality → refine; crop → flat cache → registry finalizer; keypoint and refine batches; subject-mask infer → finalize | About 12 production shell launchers |
| 5 | Cross-family recipe | `recording_analysis.smk`: intake sentinel → detect → refine → crop/cache → keypoints → masks → kinematics, per recording, composed from the family fragments. **This is the first time intake triggers analysis automatically**, and it needs a separate decision about which recordings are eligible | The agent driving the chain by hand |
| 6 | Analytics and cohorts | Wrap `execute_analysis_workflow` as one job per execution, keeping its internal DAG and serial finalizer. Do not explode its nodes into Snakemake until there is a reason | `analysis_workflow`, `chaser_analytics`, `analytics_export`, `cohort_*`, `validated_behavior_cohort_export` launchers |
| 7 | `clipped_inference` (4.8k lines) | Last. It is the largest and most stable planner, and gains least | Its submit path |
| — | Never migrated | 21 benchmark/canary/A-B scripts, 5 one-offs, 3 ops scripts | Move to `scripts/experiments/` / `scripts/archive/` as each closes |

What stays a script permanently:

- operator entry points (`palette-flow` itself);
- deployment helpers (`deploy_palette_cluster_worktree.sh`, `push_and_update_groups_checkout.sh`);
- registry backup;
- benchmarks and one-offs.

What stays in Python permanently:

- `fisheye.cluster.lsf.{models,backend,runtime,task_group}`;
- every family planner.

`submission.py`'s apply loop is retired when its last caller moves.

Each step follows the AGENTS.md staging rules:

- preservation tests first (the planner's plan JSON is byte-identical before and after);
- one family per change, with CI green;
- an end-to-end canary on a commit-pinned deployment before removing a launcher.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Snakemake deletes a declared output that is real data | Outputs are sentinels only; a Snakefile lint test fails on any output outside `flow_root` or any `directory()` |
| Controller restart duplicates a running job | Job-level `O_EXCL` claims with an attach path (§5.3); tested in exit criterion 3 |
| The runner reports done while Palette disagrees | Sentinel written only by the probe; `reconcile` reports drift; no auto-heal |
| Stages exit 0 after a refused publish (G-1/N-4) | The wrapper's probe gates the sentinel; fix `stage_complete` independently |
| `bjobs` polling load or SSH churn | NFS-only status; at most one batched `bjobs` per 5 minutes, rate-limited on ws1; ControlMaster; a test asserts the budget against a fake runner |
| ws1 becomes a single point of failure | LSF work survives; claims let the controller resume; the alternative placement in §5.2 is available |
| Executor-plugin or Snakemake API churn (v8 changed the plugin API) | Pin versions in the `palette-flow` env; our executor logic lives in Palette, not in a plugin |
| Scope creep into a Palette DSL | Snakefiles stay thin and planners stay authoritative. No YAML stage language (per the LSF framework doc's decision) |
| Two orchestration paths during migration | The retirement table above; each family moves completely before the next starts |

## 10. Decisions (Jeremy, 2026-10-07)

1. **Runner:** Snakemake. A live web dashboard would be nice but is not necessary, so Nextflow/Seqera's UI advantage does not decide this. The status page in §7 step 4 stays optional.
2. **Environment:** a separate `palette-flow` conda env, rather than adding to `palette-py311`. Creating it is a separate, explicit step, and nothing has been installed yet.
3. **Login-node load:** nothing may spam the login nodes with checks. One check-in every 5-10 minutes is acceptable. Implemented as the contact budget in §5.1: NFS-only status, and at most one batched `bjobs` per 5 minutes.
4. **Scope:** slice 1 is intake only. The plan executor (§8 step 2) gets its own review after slice 1 meets its exit criteria.

Still open:
- the controller host stays ws1 (§5.2), which Jeremy did not object to;
- the receipt-verification timing that sets the tick interval (§6.1).

## Decision log

- 2026-10-10: pin moves that add registry migrations drain first, enforced by `pin-check`. This follows palette-33's finding that older code works silently against a migrated registry but misses new projections, plus the schema guard in #344/#346 that refuses non-additive migrations.
- 2026-10-09: retry cap and isolated synthetic trials (Jeremy approved both).
  - After `max_consecutive_failures` (default 3) consecutive retryable failures of a step, the runner writes a `retry_cap` hold (`refused.json`) instead of retrying every tick.
  - What counts as a failure: exit 1, a lost LSF job, a submission error, or a missing producer-commit deployment. Exit 75 ("held by another job") does not count, and success resets the count.
  - Deterministic refusals still exit 65 from `fisheye.intake`; the cap is only a backstop.
  - `allow_synthetic_isolated_registry` passes `--allow-synthetic-isolated-registry` to `register-delivery`. The config refuses it unless both the runner's registry and the registrar config's registry are the same non-canonical file. It exists only for synthetic kill tests.
- 2026-10-08: the slice-1 implementation drops the `cluster-generic` executor. Its job script runs a per-job Snakemake inside every LSF job, but the `palette-flow` env is on ws1's local disk. A copy on `/groups` would be cheap (the env measures 505 MB), so storage is not the reason. Jeremy chose the supervisor-only design for these reasons:
  - LSF jobs run only Palette code, exactly as today's launchers do;
  - there is one env to keep pinned;
  - after a controller restart the runner re-attaches to running jobs (tested), where Snakemake's executors would resubmit.

  The cost is about 300 lines of submit, wait and re-attach code that Palette owns. All rules are local on ws1, and cluster steps submit and then wait on NFS evidence through `fisheye.flow.lsf`. The login-node budget is unchanged: `bsub` once per attempt, and `bjobs` at most once per 5 minutes through a shared, flock-guarded cache, used only when a job has no fresh heartbeat.
- 2026-10-07: §5.4 adds producer-commit-bound steps. Intake registration runs at the import receipt's `producer_git_sha`, because the identity authority requires an exact commit match. This came from palette-33's `fisheye.intake` review.
- 2026-10-07: #286 merged (8e96c399). §6.3 now requires runner and v2 state to stay separate from the v1 `.processing_state`.
- 2026-10-07: recorded PR #290's exit-code table and its NFS flock lease behaviour in §5.3.
- 2026-10-07: aligned with the intake single-writer design (PR #290). Discovery uses `probe_register`, not runner sentinels. Stages that own a lock replace runner claims. Intake attempts get fresh run dirs.
- 2026-10-07: Jeremy decided §10: Snakemake, a separate env, at most one login-node check-in per 5-10 minutes, intake only. §5.1 gains an explicit login-node contact budget.
- 2026-10-07: §6 corrected after `palette-a0` review. Discovery now includes `reserved`/`materialized`/`retiring` states, resumes from `state["plan"]` and keeps the `admission_contract` mode fixed. Registration is one `publish_registry_shadow` per delivery, and receipt-verification cost is to be measured before setting the tick.
- 2026-10-07: draft opened. Recommends Snakemake as a supervisor with Palette-owned "done" sentinels, with intake as slice 1 and the plan executor as slice 2. Not accepted yet.
