# Palette: independent code-first design assessment

> Update added 2026-09-06: finding 1 is partly superseded at main `1bf9d919`.
> Six stages now have runtime verifiers, and reused inputs are dynamically
> revalidated before execution. The old metadata-only swim-bout probe is now
> refused; the other six characterization probes still reproduce their bad
> behaviors. Preserved probe assertions below describe the old snapshot, not
> desired correctness. See the [September 6 reconciliation](review_docs_rules_landing_handoff_2026-09-05.md#september-6-audit-reconciliation) for exact commits, commands, and limits.

Date: 2026-09-04. Reviewed commit: `73bee0d5194c662e3b7e535be3de92db7ef53f63`.

This is a separate assessment from [the review-wave second opinion](review_wave_second_opinion_2026-09-04.md). That report audits the handoff, historical changes, measurements, and implementation brief. This one starts from the code's public entry points, execution paths, storage abstractions, and consumers. It does not repeat the live-store survey or treat the handoff's diagnosis as a premise.

The user-requested [parallel shared-helper consolidation review](shared_helper_consolidation_review_2026-09-04.md) follows this assessment. It identifies concrete duplicated implementations, observed behavioral drift, existing helpers to adopt, and enforcement needed to preserve the strong contracts while reducing bloat.

Only this document was added to the repository during the original inspection. The user subsequently requested the rig-default and completion-state clarification in section 6 and a separate parallel consolidation review; those are documentation-only follow-ups. Source, the live registry, and stores were not edited. Synthetic checks used temporary fixtures; no commit, push, installation, deployment, or `/groups` write was performed.

## My judgment

Palette has a substantially better scientific and storage foundation than its public interfaces suggest. The explicit coordinate/frame domains, supplier identities, array contracts, physical write-ownership rules, bounded specialist readers, and immutable-generation exports are valuable engineering. They are not merely aspirational documentation: there are working implementations and focused tests.

Its main design weakness is **inconsistent guarantees when moving between components**. Some paths carry validated identity, lineage, bounded access, and publication semantics end to end. Others turn those into a run-name string, an availability boolean, an eagerly materialized dictionary, or an optimistic success status. The individual algorithms and storage helpers can be careful while the assembled workflow remains weaker than its components.

I would keep Palette as a modular, in-process platform with batch adapters. I would not begin with microservices, a new scheduler, a universal provenance envelope, or another storage abstraction. I would make its strongest existing contracts the normal application interface, and make weaker inspection/compatibility modes explicit.

This is already a sizable laboratory data platform, not a small numerical library. Its architecture needs to distinguish scientific computation, artifact access, workflow execution, and interactive presentation accordingly. Crimson integration can wait; a trustworthy bounded-access interface would benefit Palette immediately and become a useful integration boundary later.

## Inspection scope and evidence

I manually traced selected paths through `api`, `cli`, `shared`, `registry`, `analysis_workflows`, `cluster/lsf`, keypoint inference, specialist analysis readers, analytics publication, and the group viewer. I also inspected packaging and CI configuration. I did not manually read every source file, audit every scientific formula, or inspect Crimson.

A standard-library AST census parsed all 1,514 tracked Python files under `src/fisheye`, totaling 1,084,092 physical lines, without syntax errors. It found 420 modules under `utils`, 319 under `shared`, and 153 modules over 1,500 lines. These are maintenance-context measurements, not correctness scores. The approximate static import graph did **not** show one enormous cycle: its largest strongly connected components contained eight modules. The graph includes function-local/type-checking imports and omits dynamic imports, so it is not a runtime dependency proof.

Evidence below distinguishes direct code observations, synthetic reproductions, and recommendations. Source anchors refer to the reviewed commit; all shortened source paths start at `src/fisheye/`.

## What I would preserve

### Scientific identity and supplier contracts

`shared/frame_domains.py:22` names acquisition, stored-Zarr, source-video, run-frame, and crop-video domains instead of assuming every integer frame index means the same thing. Its lookup machinery includes bounded dense-versus-sparse choices (`:112`). That is the right kind of explicitness for this domain.

`analysis_workflows/provider_analysis_offers.py:240` binds provider identity to recording, run location, manifest/content information, coordinates, timing, and validity. `ProviderRequirements` (`:385`) expresses independent consumer needs; `AnalysisOffer` (`:597`) checks compatible providers and timing before declaring readiness. This supports a useful rule: consume the sufficient validated supplier, not every upstream authority again. These offers already have bindings and materializer consumers; they are not an unused design sketch.

Preserve the distinctions between canonical representation, accepted selection for a use, and human-reviewed claims. A validated body-frame supplier need not expose canonical keypoints or trigger another human-review gate. Similarly, modern mask selection should retain its existing bundle/envelope contract, not acquire a parallel generic segmentation authority.

### Storage planning is unusually deliberate

`shared/zarr/array_contracts.py:94` represents dtype, rank, symbolic dimensions, axes, units, and coordinates separately from execution. `shared/zarr/storage_planner.py:138` expresses whole-shard/whole-chunk ownership and serialized partial-update cases; `:148` plans physical layout from access intent and byte-oriented profiles. Named profiles have validation and tamper-evidence tests.

In keypoint inference, `detection/detect_keypoints_yolo.py:998` implements a two-buffer aligned shard writer, bounded queues, sequential row validation, and asynchronous flushing. It accumulates inference batches into storage-owned units rather than assuming model batch size must equal physical write size. The later publication path rechecks model bytes, validates coordinate publication, and handles eligibility/activation explicitly. This is a strong example to generalize from, not replace.

### Bounded readers and streaming exporters already exist

`shared/crop_image_source.py:835` presents several pixel backends through explicit shape and row/index reads (`:1813`, `:1824`). This is a justified abstraction: consumers should not need to know whether pixels come from dense crops, a cache, or a video-backed source.

`analysis/eye_angle_io.py:1009` supports bounded row reads. `analysis/tail_kinematics_io.py:439` offers bounded window access and checks that the selected publication has not changed during reading. `analytics_exports/eye_trace_samples.py:444` writes bounded batches into Parquet row groups with source-frame and payload checks. Similar streaming machinery exists for tail traces.

The missing piece is not the existence of streaming code. It is making this access discipline consistent at the interfaces people naturally discover first.

### Publication has a sound generation model in important paths

`analytics_exports/publication.py:135` stages and validates an exact inventory, moves an immutable generation into place, and changes the selecting manifest under a lock with a baseline-identity comparison. Expensive validation is outside the short manifest-commit critical section. The viewer's file discovery also uses manifest-selected parts, not a loose directory glob (`group_analytics_viewer/query.py:376`).

That distinction between immutable payload generations and a mutable selecting manifest is sound. The viewer cache discussed below fails to preserve it, but that is a consumer defect, not a reason to discard the publication design.

### Execution and validation have useful existing structure

The analysis DAG planner is deterministic and structurally separated from execution. Workflow profiles explicitly mark whole-recording temporal-state stages, avoiding the assumption that arbitrary row partitioning preserves temporal algorithms. LSF models represent resources, arrays/bundles, concurrency limits, and different dependency conditions explicitly (`cluster/lsf/models.py`). Whole-recording submission composes different resource classes rather than invariably reserving one GPU allocation for every downstream task.

Provenance machinery also captures substantial software/runtime detail; it would be wrong to diagnose a complete absence of environment capture. CI includes import-boundary, module-size, Zarr-open-mode, and contract-freshness gates, plus a non-editable wheel and installed-contract smoke (`.github/workflows/ci.yml:153`). These are meaningful safeguards. I did not establish the current commit's CI status.

## Ranked design concerns

Ranking reflects likely scientific/operational consequences and breadth of exposure, not a claim that every affected path has produced bad live data.

### 1. High: planning availability is still accepted as execution evidence

`analysis_workflows/runtime_verification.py` describes itself as full-strength verification of workflow-produced artifacts. Its verifier registry at `:132` contains only `subject_shape`. For other stages, `verify_persisted_stage_output` (`:137`) returns the result of metadata availability discovery without invoking a stage-specific runtime verifier.

A synthetic `swim_bouts` run containing only parent/run `zarr.json` files, complete/eligible attributes, and selectors was accepted as available by this runtime function. It contained no arrays, manifest, source binding, or scientific payload. Supplying an expected track-kinematics dependency did not make this generic path reject it.

This does **not** prove the actual swim-bout writer omits validation; individual materializers may validate strongly. It proves the common post-execution boundary does not independently deliver its stated guarantee for every stage.

Reuse has a related gap. `NodePlan` (`analysis_workflows/dag.py:14`) records a selected run/path and temporal policy, but not a concrete immutable input/recipe identity. Planning accepts `available=True` as reuse. The runner's initial-result construction (`utils/execute_analysis_workflow.py:123`) turns planned reuse into reused status without rerunning the strict stage validator there.

Keep cheap discovery for planning. At execution admission, bind reusable artifacts to the applicable stage contract and exact immutable identities, and validate that binding. A previously validated supplier may legitimately cut off upstream work; do not recreate or independently reapprove sealed ancestors. The missing distinction is *discovered candidate versus verified reusable result*, not *all ancestors must run again*.

### 2. High: training identity can change, and a single recording operation can partially commit

`registry/db.py:6918` implements `upsert_training_set` by replacing membership, query, task metadata, and `created_utc` on the same `set_id`. A temporary-registry test changed `pose_set` from `['source_a']` to `['source_b']` under one identity. The current row retained only the latter membership.

Mutable cohort definitions are useful. Historical training-set identity needs different semantics: a model referencing only that mutable ID cannot recover the original membership from the current row. A separately retained, verified training manifest can provide that evidence; this is not proof all past training provenance is lost. It is a reason not to treat `set_id` alone as immutable experimental identity.

There is also a concrete transaction boundary problem. `record_training_run` (`registry/db.py:6807`) commits its `training_runs` update before calling `record_training_model`. Injecting a failure in the latter left a committed training run and no corresponding model projection, even after caller rollback. This is a local SQLite atomicity issue, not an unavoidable cross-system transaction limitation.

Use immutable membership/split revisions for historical experiments, with names and current cohort queries remaining mutable conveniences. Either record the two local projections transactionally or explicitly model an asynchronous projection with observable pending/error state and repair. Do not present a partially completed mirror operation as an indivisible update.

### 3. High: registry query semantics disagree about which subject must satisfy a filter

`registry/query.py:251` constructs independent JSON-membership predicates for the lower and upper bounds of numeric ranges. A recording containing subjects aged 4 and 10 days matches `--dpf-min 5 --dpf-max 8`: 10 satisfies the lower bound and 4 satisfies the upper bound. No individual satisfies the interval.

The subject-lineage helper in `utils/registry_query.py:915` instead applies both bounds to the same `recording_subject_overview` row. On the same synthetic recording it correctly returns no dataset. The probe executed predicates generated by the former query builder and the actual latter helper against in-memory SQLite.

This is more important than duplicate SQL as a style concern. Query code is part of the scientific selection contract. Dataset, recording, subject, and artifact filters need explicit grain and same-entity correlation. Tests should include mixed-subject recordings, not only one-subject happy paths.

Consolidate on the existing normalized semantics where appropriate. There is no need to introduce a new query language or replace SQLite to fix this disagreement.

### 4. Medium-high: the viewer loses both bounded execution and publication identity

`group_analytics_viewer/query.py:396` caches table rows by `(export_root, export_run_id, table_name)`. It resolves manifest-selected files only on a cache miss, collects the complete table, converts it into Python dictionaries, and retains it. `load_table_rows` (`:412`) then copies those dictionaries on each call. The 32-entry limit bounds table count, not bytes.

For example, spatial-occupancy queries load the full table before applying some filters and aggregation in Python (`:1186`). Polars' streaming execution engine does not change the fact that `collect()` materializes a DataFrame, after which `to_dicts()` creates Python objects. [Polars collect documentation](https://docs.pola.rs/api/python/stable/reference/lazyframe/api/polars.LazyFrame.collect.html).

The cache also has a correctness problem. `analytics_exports/eye_trace_samples.py:731` supports overwrite into a new immutable generation under the same export run ID. `ViewerContext` (`query.py:222`) does not bind a generation or manifest digest. A synthetic cache probe changed the backing generation under a fixed key; the second call returned the first generation and performed no new scan. I found no production invalidation call for this cache.

This establishes stale-cache behavior after a supported same-ID republish, not that a particular live chart was wrong. Whole-table loading may currently be acceptable for small summary tables; I did not measure current table sizes or an out-of-memory failure.

Pin a viewer/query context to one manifest generation. Include that identity in caches and use explicit byte budgets. Apply column projection, filters, and aggregation before materialization; return bounded windows or batches for trace data. Do not replace the exact manifest inventory with directory discovery while making this change.

### 5. Medium-high: the public read API is a convenience API, not yet a dependable data-service boundary

`shared/recording.py:239` returns materialized array mappings. Detection and keypoint methods expose array-name selection but no row, time-window, or byte budget (`:267`, `:285`). Their helper, `shared/zarr_helpers.py:303`, slices every selected array in full and silently continues after access or read exceptions.

A fake array raising `OSError` was omitted without an exception or error entry, even when explicitly requested. The caller received the other array as if the mapping were an ordinary result. Missing optional arrays, unavailable data, and failed reads should not be indistinguishable at a production-facing boundary.

`Recording` also always opens direct metadata (`:170`), rather than expressing whether it is inspecting a mutable archive or consuming a published immutable generation. Its generic run-resolution path does not by itself promise every stage-specific scientific acceptance check. An explicit run name is useful for inspection, but it is not equivalent to a validated authority for an arbitrary consumer.

Make the ordinary production reader a bounded, generation-bound view using the existing stage-specific resolvers and readers. Keep exploratory full materialization and best-effort inspection available, but name them explicitly and report per-array failures. Do not silently impose a human-review gate where the consumer contract requires only a validated supplier.

### 6. Medium: scientific recipes and presentation dependencies leak into workflow orchestration

The default profile makes `bout_kinematics` depend on `track_kinematics_visualization` (`analysis_workflows/profiles/core_behavior_v1.yaml:90`). The bout command uses track kinematics, swim bouts, and eye angles; it consumes no visualization artifact (`analysis_workflows/execution.py:245`). Therefore a plotting failure can block numerical computation through a graph edge that is not a numerical input dependency.

A release may deliberately require visual evidence. If so, represent that as a reporting/publication requirement, not an undeclared scientific data dependency. Optional presentation products should generally branch from completed numerical products. The bout command also requests its own visualization artifacts; removing the upstream graph edge alone would not remove every plotting dependency in the implementation.

The command renderers also embed scientific defaults: kinematics hysteresis/smoothing, bout peak parameters, physical-activity thresholds, and pre/post windows (`execution.py:109`, `:181`, `:245`). The profile exposes temporal/export policy but not a complete typed computation recipe. The current command/commit can record what ran; however, changing the recipe through this workflow often means changing renderer code, and the structural reuse plan does not itself bind those parameters to prior results. The underlying scientific commands already expose parameters; this is not a claim that the whole library prohibits parameter overrides.

These workflows are curated recipes, which is a reasonable starting scope. Make the recipe version and effective parameters explicit inputs to computation identity. Keep storage geometry, scheduler resources, and scientific parameters distinct so resource tuning does not accidentally redefine science or silently reuse a result from another recipe.

#### User clarification and agreed design direction

The user confirmed that these are intentional defaults for recordings from their camera configuration and rig. Preserve those values and their intended applicability. The concern is where this calibrated recipe is represented and checked, not whether defaults should exist or whether these values are scientifically incorrect or unrecorded.

A named, versioned rig/camera recipe should supply the existing effective defaults through an explicit validated configuration. It should declare the recording/camera/calibration assumptions needed by those parameters, especially pixel-valued thresholds. Resolve the recipe and any permitted overrides into one effective parameter set before planning execution; record its identity and values, and compare applicable scientific inputs and parameters before reuse. Do not invent rig identifiers or compatibility criteria during this review, silently retune defaults, or require every user to specify every parameter manually. Recipe selection chooses the scientific configuration; the renderer translates it into arguments; execution configuration chooses resources. Defaults, units, override policy, and applicability belong with the recipe rather than being duplicated between those layers.

The user also endorsed distinguishing three completion claims:

| Claim | Meaning | Effect of an independent presentation failure |
|---|---|---|
| Computation complete | The declared numerical products and their scientific contracts have been validated. | Preserve valid numerical results and their reuse eligibility under the applicable contract. |
| Presentation complete | The required plots/interactive specifications for the selected numerical identities have been produced and validated. | Report the missing/failed presentation product and retry it independently where safe. |
| Deliverable complete | All products required by the selected workflow, including required presentation, are ready and mutually bound to the intended inputs/results. | Remain incomplete when a required presentation product failed; do not silently waive it. |

This is an agreed design direction, not an implemented status schema. It separates product state from attempt failure and from authority activation. A failed plot does not make valid numerical bytes invalid; equally, valid numbers do not make a required report deliverable complete. A plot is a scientific dependency only if a consumer actually requires its information or an applicable contract explicitly defines that dependency. Presentation should bind the exact numerical generation it depicts, and presentation retries should not silently change numerical identity. No new human-review requirement or exception to required CI/promotion gates is implied.

### 7. Medium: the package boundary is weaker than the shared contracts

`api.py:5` re-exports request types and verbs from `cli.palette`. The crop verb imports private planning/configuration helpers from `utils.crop_batch` (`cli/palette.py:1859`). The import census found 29 static edges into `utils` from packages other than `utils` and `diagnostics`. Production application behavior lives in a directory whose name implies miscellaneous support.

There is also a small but concrete symptom of weak result types: `_crop_result_status({})` returns `('ok', 'OK')` (`cli/palette.py:1840`). The probe does not establish that a normal crop run returns `{}`. It shows that absence of known failure fields is accepted as success at this normalization boundary.

The import-linter shared-layer contract (`pyproject.toml:111`) is useful but largely divides `shared` from peer application packages; it does not enforce a separate scientific/application/adaptor architecture. `shared` itself contains substantial domain publication logic, while `utils` contains application services. Large modules compound the difficulty of seeing which functions own side effects, contracts, or presentation.

Move responsibilities when a stable boundary is clear: CLI parsing and printing should call application services; those services should return explicit plans/results and invoke domain contracts and storage/execution adapters. Do not split files merely to satisfy a line-count aesthetic, and do not make the explicitly legacy `core/pipeline.py` the new architectural center.

## What “single source of truth” should mean here

I would not make the registry, one selector attribute, or one generic envelope authoritative for everything. These are distinct questions:

| Question | Appropriate source and existing foundation |
|---|---|
| What scientific bytes and coordinate/timing identities are these? | Immutable artifact identity and its applicable array/publication contract. |
| Which artifact is accepted for this particular use? | Stage-specific selection/activation contract, including bundle selection where applicable. |
| Where can it be found, and which recordings match a query? | Registry locators/projections and grain-correct query services. |
| What computation was requested, and what exact inputs did it bind? | Versioned recipe and a concrete execution/reuse plan. |
| What happened during one attempt, and what was published? | Attempt, validation, and publication receipts, with reconcilable registry projection. |

These are interface responsibilities, not a proposal for five new schemas. Several already have strong implementations. The work is to connect them without reducing a verified artifact to a mutable path or treating a catalog projection as scientific proof.

A useful public abstraction would be an artifact reference that delegates to the relevant existing contract, plus a bounded verified view over it. It should carry enough identity to distinguish relocation, selection changes, physical generation changes, and changes in scientific content. Consumers that need only a derived supplier should not reopen its sealed upstream pixel/landmark authorities unnecessarily.

For immutable publications, verify expensive content at the appropriate publication/admission boundary and reuse a trusted bound result where the contract permits. Rehashing every byte on every viewport request would undermine the access-aware design. Conversely, caching an unbound run-name lookup forever is not an acceptable substitute.

## Workload efficiency and distributed execution

The most valuable performance work now is to verify that the existing access-aware pieces stay connected end to end. I would choose one representative recording workflow and measure:

- Source bytes read per useful output row/window, including cold and warm metadata reads.
- Peak resident memory and live buffer count across decoding, preprocessing, GPU execution, result transfer, and shard flushing.
- Physical chunk/shard ownership, write amplification, and local staging/copy cost.
- Queue wait, allocated resources, compute time, GPU idle intervals, and publication/validation time separately.
- First-window and subsequent-window latency for the resulting analytics product, including a same-run-ID generation replacement.

These measurements should extend existing timing/provenance instrumentation, not create a competing telemetry system. I have not measured a throughput improvement or demonstrated a GPU bottleneck in this inspection.

There are three different partitioning decisions: independent scientific work, scheduler tasks, and physical storage writes. A whole-recording temporal algorithm need not become row-parallel just because storage is sharded. Independent recordings are a natural initial concurrency boundary; within a stage, preserve declared state/halo semantics and whole physical write ownership. The aligned inference writer is already a good example of separating these decisions.

Resource units should be visible in plans. `LsfResources.mem_gb` is documented as per-slot memory, so users should see both per-slot and nominal aggregate requests alongside core counts. This is a reviewability concern, not a measured claim of overreservation. CPU finalization and publication should continue to use their appropriate resource class rather than unnecessarily holding a GPU.

Failure recovery matters at least as much as DAG construction: retry must distinguish an unpublished attempt, a validated generation, a changed selection, and a projection that failed after publication. The earlier second-opinion report discusses those recovery gaps; a scheduler replacement alone would not resolve them.

Finally, explicitly state the storage failure model. The inspected analytics publisher uses atomic replacement and locking but does not explicitly flush/synchronize file and directory durability. Atomic visibility is not the same claim as survival of node/power loss on every filesystem. Python documents `fsync` separately from rename/replace. [Python OS documentation](https://docs.python.org/3/library/os.html#os.fsync). This is a contract question to resolve against the actual storage system, not a demonstrated data-loss incident or a recommendation to indiscriminately `fsync` every Zarr write.

## Where I would concentrate effort

The first priority is semantic correctness at already-exposed boundaries: runtime verification/reuse, training membership identity and transactionality, and mixed-subject query semantics. These can change which scientific result is believed or which data enter a cohort without producing an obvious execution error.

Next, make one normal application path consistently typed, bounded, and identity-preserving from input selection through execution to a reader. That includes explicit read failures, generation-aware caching, and recipe-bound reuse. Use existing supplier/array/publication contracts; remove bypasses before introducing more general infrastructure.

Then simplify orchestration boundaries: separate computation from rendering, expose effective scientific recipes, and let the CLI and batch adapters call the same application services. Apply module extraction at these boundaries rather than performing a repository-wide renaming exercise.

An appropriate first demonstration would be a bounded motion/bout analytics slice: bind exact sufficient inputs, show why reuse is valid, execute or reuse numerical stages without requiring plotting, publish an exact immutable generation, and query a small time window with explicit identity and memory bounds. Include fault injection and publication replacement, not just a happy-path smoke. This is an architectural direction for adoption into the existing authority-consolidation work queue, not a new parallel fix queue or authorization to implement it now.

I would defer Crimson integration, microservice decomposition, a registry-backend replacement, and a new workflow framework until this slice exposes a concrete need for them.

## Validation and limits

All Python commands used `scripts/py`; focused pytest execution was outside the sandbox and disabled the pytest cache and Python bytecode writes. The final run was:

```text
scripts/py -B -m pytest -q -p no:cacheprovider -rs \
  /tmp/palette-design-review-20260904-zuhe29/test_design_probes.py \
  tests/unit/fisheye/test_recording_accessor.py \
  tests/unit/fisheye/test_zarr_storage_planner.py \
  tests/unit/fisheye/test_zarr_array_contracts.py \
  tests/unit/fisheye/test_provider_analysis_offers.py \
  tests/unit/fisheye/test_analysis_workflow_dag.py \
  tests/unit/fisheye/test_analysis_workflow_execution.py \
  tests/unit/fisheye/test_frame_domains.py
```

Result: **122 passed, 1 skipped in 6.09 seconds**. Seven passing cases are characterization probes of undesirable behavior; they are not fixes or assertions that those behaviors are acceptable. The remaining 115 passes exercise existing repository tests. The skipped frame-domain integration case requires an absent historical `/nvme1` training store (`test_frame_domains.py:375`). No full CI, GPU inference, production cluster workload, or live registry acceptance run was performed in this inspection.

| Synthetic probe | Observed behavior | Scope of conclusion |
|---|---|---|
| Requested array raises `OSError` | Array omitted; other arrays returned without error. | Shared materializing reader suppresses a storage failure. |
| Complete/eligible swim-bout metadata, no payload | Common runtime verifier returns available. | Common gate is metadata-only for this stage; individual writer validation is not disproved. |
| Backing viewer generation changes under same cache key | Original rows returned; only one scan. | Cache is not generation-aware; no live chart was audited. |
| Empty crop result dictionary | Classified `ok`. | Normalizer accepts missing success evidence; real crop output was not simulated end to end. |
| Subjects aged 4 and 10; request ages 5 through 8 | One query includes the recording, normalized subject query excludes it. | Existing query paths disagree on same-subject range semantics. |
| Upsert one training-set ID with different membership | Current membership replaced. | ID alone is not an immutable historical membership reference. |
| Fail model projection after recording training run | Training run remains committed; model row absent. | The local multi-table operation is not atomic. |

The temporary path above is the executed audit artifact, not a permanent repository test. For durable reproduction, the full seven-probe source is included below. It writes only pytest temporary fixtures/in-memory data, not the live registry or stores. Save it to a temporary test file and substitute that path into the command above. The assertions intentionally characterize the reviewed commit and should be inverted or replaced when fixes are implemented.

<details>
<summary>Seven synthetic characterization probes</summary>

```python
import json
import sqlite3
import sys
from types import SimpleNamespace

import numpy as np


def test_array_accessor_silently_omits_read_failure():
    from fisheye.shared.zarr_helpers import read_zarr_array_mapping

    class BrokenArray:
        shape = (3,)

        def __getitem__(self, key):
            raise OSError("simulated storage read failure")

    result = read_zarr_array_mapping(
        {"good": np.arange(3), "broken": BrokenArray()},
        physical_prefix="example", array_names=["good", "broken"],
    )
    assert set(result) == {"good"}


def test_runtime_accepts_bout_metadata_without_payload(tmp_path):
    from fisheye.analysis_workflows.runtime_verification import verify_persisted_stage_output
    from fisheye.analysis_workflows.availability import STAGE_RUN_PARENTS

    parent = tmp_path / STAGE_RUN_PARENTS["swim_bouts"][0]
    run = parent / "missing_payload"
    run.mkdir(parents=True)
    (parent / "zarr.json").write_text(json.dumps({
        "zarr_format": 3, "node_type": "group",
        "attributes": {"palette_completion_epoch": 2,
                       "latest": "missing_payload", "latest_complete": "missing_payload"},
    }))
    (run / "zarr.json").write_text(json.dumps({
        "zarr_format": 3, "node_type": "group",
        "attributes": {"palette_run_completion_status": "complete",
                       "stage_selector_eligible": True},
    }))
    result = verify_persisted_stage_output(
        tmp_path, "swim_bouts", requested_run="missing_payload",
        dependency_runs={"track_kinematics": "expected_parent"},
    )
    assert result.available is True
    assert list(run.iterdir()) == [run / "zarr.json"]


def test_viewer_cache_does_not_revisit_changed_generation(monkeypatch):
    from fisheye.group_analytics_viewer import query

    state = {"generation": 1, "reads": 0}

    class Frame:
        def collect(self, **kwargs):
            return self

        def to_dicts(self):
            return [{"generation": state["generation"]}]

    def scan(*args, **kwargs):
        state["reads"] += 1
        return Frame()

    monkeypatch.setitem(sys.modules, "polars", SimpleNamespace(scan_parquet=scan))
    monkeypatch.setattr(query, "parquet_files", lambda *a, **k: ("fake_part",))
    query._load_table_rows.cache_clear()
    try:
        first = query._load_table_rows("/fake", "same_run", "table")
        state["generation"] = 2
        second = query._load_table_rows("/fake", "same_run", "table")
        assert first == second == ({"generation": 1},)
        assert state["reads"] == 1
    finally:
        query._load_table_rows.cache_clear()


def test_crop_result_classifier_accepts_empty_result():
    from fisheye.cli.palette import _crop_result_status
    assert _crop_result_status({}) == ("ok", "OK")


def test_registry_query_range_can_match_two_different_subjects():
    from fisheye.registry.query import _build_query, _parse_args
    from fisheye.utils.registry_query import _query_dataset_ids_by_subject_lineage

    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE datasets (dataset_id TEXT, recording_id TEXT);
        CREATE TABLE dataset_context_current (
            dataset_id TEXT, dpf_at_acquisition_effective INTEGER, dpf_values_json TEXT
        );
        CREATE TABLE recording_subject_overview (
            recording_id TEXT, cross_id TEXT, genotype TEXT, dpf_at_acquisition INTEGER
        );
        INSERT INTO datasets VALUES ('mixed', 'recording');
        INSERT INTO dataset_context_current VALUES ('mixed', NULL, '[4,10]');
        INSERT INTO recording_subject_overview VALUES ('recording', 'a', 'x', 4);
        INSERT INTO recording_subject_overview VALUES ('recording', 'b', 'y', 10);
    """)
    query, params = _build_query(_parse_args(["--dpf-min", "5", "--dpf-max", "8", "--limit", "0"]))
    condition = query.split(" WHERE 1=1", 1)[1]
    old_rows = db.execute(
        "SELECT d.dataset_id FROM datasets d JOIN dataset_context_current dcc "
        "ON d.dataset_id=dcc.dataset_id WHERE 1=1" + condition, params,
    ).fetchall()
    exact_rows = _query_dataset_ids_by_subject_lineage(
        SimpleNamespace(conn=db), cross_id=None, genotype=None,
        dpf=None, dpf_min=5, dpf_max=8,
    )
    assert [r["dataset_id"] for r in old_rows] == ["mixed"]
    assert exact_rows == set()
    db.close()


def test_training_set_membership_can_change_under_same_id(tmp_path):
    from fisheye.registry.db import Registry

    registry = Registry(tmp_path / "test_only.sqlite")
    try:
        registry.upsert_training_set(set_id="pose_set", name="test", task_type="pose",
            query_filter={}, dataset_ids=["source_a"])
        registry.upsert_training_set(set_id="pose_set", name="test", task_type="pose",
            query_filter={}, dataset_ids=["source_b"])
        row = registry.conn.execute(
            "SELECT dataset_ids_json FROM training_sets WHERE set_id='pose_set'"
        ).fetchone()
        assert json.loads(row["dataset_ids_json"]) == ["source_b"]
    finally:
        registry.close()


def test_training_run_commits_before_model_projection(tmp_path, monkeypatch):
    import pytest
    from fisheye.registry.db import Registry

    registry = Registry(tmp_path / "test_only.sqlite")
    try:
        def fail_projection(**kwargs):
            raise RuntimeError("simulated projection failure")

        monkeypatch.setattr(registry, "record_training_model", fail_projection)
        with pytest.raises(RuntimeError, match="projection"):
            registry.record_training_run(
                run_id="test_pose", set_id=None, task_type="pose", config_path=None,
                manifest_path=None, model_path=None, metrics_path=None, status="success",
            )
        registry.conn.rollback()
        assert registry.conn.execute(
            "SELECT count(*) FROM training_runs WHERE run_id='test_pose'"
        ).fetchone()[0] == 1
        assert registry.conn.execute(
            "SELECT count(*) FROM training_models WHERE run_id='test_pose'"
        ).fetchone()[0] == 0
    finally:
        registry.close()
```

</details>
