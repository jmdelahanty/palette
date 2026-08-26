# Protocol Semantic Step Identity Contract

<!-- contract-meta
status: active
implementation: partial
version: 2
last_updated: 2026-08-24
last_verified: 2026-08-25
-->

## Purpose

Palette must distinguish a standalone producer-authored `SOLID_BLACK` step
from the pre, training, and post windows nested inside a `CHASER` step. A
recording with the recipe `CHASER` has no standalone black baseline. That is a
valid protocol variant, not missing metadata.

Recording names, stimulus-run names, authored step labels, and legacy epoch
roles are locators or display text. They are never protocol-step identity.

## Producer authority

Historical Citrus snapshot-v1 H5 files provide one indivisible contract:

- `/protocol_snapshot/protocol_semantic_hash`;
- `/protocol_snapshot/protocol_semantic_json`;
- `/protocol_snapshot/protocol_trial_index_json`.

The semantic hash is `sha256:<64 lowercase hex>` over the exact UTF-8 semantic
JSON bytes. The trial index must bind that hash and agree with the semantic
recipe in ordered step index, mode ID, and duration. Partial presence,
malformed JSON, digest mismatch, unknown modes, conflicting color evidence,
or recipe disagreement is corruption and fails closed.

Citrus snapshot v1 does not independently checksum the trial-index bytes.
Palette therefore stores a clearly labeled
`palette_computed_trial_index_sha256` with integrity status
`palette_computed_not_producer_asserted`. It must not be described as a
producer checksum.

Citrus snapshot v2 adds a producer-authored exact-byte trial-index SHA-256 and
an exact `/protocol_execution` document bound to the same trial index and
semantic recipe. Its step and chaser-phase intervals are canonical half-open
intervals on `stimulus_frame_num`. Any camera frame IDs in that document are
explicitly correspondence evidence, not interval authority and not Orange
acquisition-row identity.

## Palette materialization

New stimulus imports store the exact source documents as bounded `uint8`
arrays under:

```text
analysis/stimulus_runs/<run>/protocol_semantic_snapshot/
  protocol_semantic_json_utf8
  protocol_trial_index_json_utf8
analysis/stimulus_runs/<run>/protocol_execution/
  protocol_execution_json_utf8
  frame_correspondence_proxy/
    stimulus_frame_num
    camera_frame_id_correspondence
    protocol_step_index
    chaser_phase_id
    in_realized_protocol
```

The run and every `steps/step_<i>` group bind the full semantic hash, recipe
index, family, mode, trial-index status, display context, and optional resolved
color. The importer reads the stored arrays back, re-reads the still-open H5
before completion, and verifies the full consolidated publication.

For v2, the correspondence-proxy arrays are sealed with exact dtype, shape,
content, manifest, and execution hashes. They support reproducible
visualization and exploratory alignment even when camera IDs repeat, skip, or
are unavailable. `sealed` here means that the stored derivation can be
revalidated; it does not mean that a camera ID has been proved equal to an
acquisition row. The proxy is permanently labeled `selector_eligible=False`
and `scientific_use_class=visualization_and_exploratory_alignment_only`.

An exact acquisition-row mapping requires the live, frame-bound identity chain
from Citrus `stimulus_frame_num` to Shaman/Orange `recording_frame_id`, followed
by Orange's finalized `recording_frame_id - 1` row convention and an exact
recording/camera binding. Timestamps can quantify or improve a proxy but cannot
substitute for that identity chain.

The state model is:

- absent status and absent semantic storage: not yet inspected/unknown;
- `legacy_missing`: the checked source H5 lacked the entire modern contract;
- `verified`: exact snapshot arrays and every materialized step binding passed;
- producer-declared v2 `unsupported`: a valid non-groupable producer state,
  not corruption and not legacy; Palette currently fails closed until it has a
  versioned materialized non-groupable run state;
- interrupted v2 execution: a valid realized recipe prefix, not a complete
  recipe; Palette currently fails closed until it has a versioned incomplete
  execution state and successor policy;
- any partial, contradictory, or mismatched state: error, never legacy fallback.

Completed, selector-visible, or selector-eligible stimulus runs are immutable.
Backfill reports `requires_immutable_successor` rather than modifying them.
Only unpublished runs may receive an idempotent in-place metadata backfill.

## Scientific classification

A standalone baseline exists only when the exact producer step validates as:

```text
stimulus_mode_id = 4
stimulus_mode = SOLID_BLACK
stimulus_family = solid_color
resolved sRGB color = [0, 0, 0, 255]
display_context = solid_black
```

A `CHASER` step has `display_context=chaser`. Its future nested windows are
named `chaser_pre`, `chaser_training`, and `chaser_post`. Existing immutable v1
artifacts using `black_before`, `chaser`, and `black_after` remain compatibility
artifacts; those caller roles do not prove standalone-black identity.

## Registry projection

Registry migration 72 adds nullable semantic identity to
`recording_stimulus_runs` and `recording_stimulus_steps`. Extraction reloads
and validates the stored snapshot arrays and every step binding. Existing rows
remain `NULL` until their Zarr authority is inspected; migration does not guess
`legacy_missing`.

The two known GoodBatBadBat cohorts can be queried by the full producer hash or
bounded recipe:

```sql
SELECT protocol_semantic_hash, protocol_recipe_label, COUNT(*) AS recordings
FROM recording_stimulus_runs
WHERE is_latest = 1 AND protocol_semantic_status = 'verified'
GROUP BY protocol_semantic_hash, protocol_recipe_label;
```

The existing `protocol_hash` remains the Palette-derived authored-protocol
definition hash. It is not replaced by, and must not be confused with, the
producer semantic hash.

## Selector-ineligible successor consumer and publication

`palette.protocol_semantic_chaser_selection_adapter.v2` is the versioned pure
successor candidate. It represents:

```text
optional standalone_solid_black step
exact CHASER step
  chaser_pre
  chaser_training
  chaser_post
```

Every nested interval must bind the exact semantic hash and CHASER step index
and be wholly contained in that step. The roles must bind the exact versioned
source-window labels `pre_event`, `training_event`, and `post_event` in
non-overlapping chronological order; swapping arbitrary caller windows is not
accepted. The evidence also binds the exact source stimulus run path and the
immutable stimulus fingerprint already sealed into the epoch selection. Legacy
`black_before`, `chaser`, `black_after`, and `all_black` keys are rejected
rather than reinterpreted.

One-step `CHASER` recipes report standalone-baseline comparisons as
`not_applicable`. A two-step `SOLID_BLACK -> CHASER` recipe distinguishes a
baseline that is selected from one that exists but is not selected. It never
substitutes `chaser_pre` or a legacy `black_before` role.

The historical v1 result projects three independent capability assessments
into the strict profile planner:

- semantic CHASER windows are ready for selector-ineligible exploratory
  candidates;
- a standalone baseline is ready, missing, or not applicable according to the
  exact recipe and explicit selection; and
- production publication remains `review_required` while producer contracts
  are pending.

The hierarchy can be atomically published below
`analysis/protocol_semantic_chaser_selection_runs/<run>`. Its strict reader
revalidates direct/consolidated metadata, the current source epoch selection,
the complete materialized semantic evidence and source fingerprint, role-to-
step containment, loader-minted current timeline evidence, exact hierarchy
recompilation, the fixed role set/order, manifest digests, and typed role arrays.
The run is immutable, selector-ineligible, has no production authority, and
does not update the registry.

The provider-aware position suite may consume that loader-minted handle. It
uses exactly `chaser_pre`, `chaser_training`, and `chaser_post`, rejects caller
role aliases on the semantic path, and stores the semantic run, manifest,
selection, protocol, boundary-policy, and source-epoch identities in its own
immutable provenance. Its `baseline_role` remains a chaser behavior-role
contrast within each epoch (for example, aggressive versus inert); it is not a
standalone protocol-black comparison. The binding states that scope explicitly,
includes no standalone baseline interval, and carries the locally computed
trial-index digest/status plus each role's producer step reference. Legacy
caller-bound and semantic-v2 position publications have explicit, different
epoch-binding modes.

Historical snapshot-v1 sources retain the unresolved raw
`STEP_END.camera_frame_id` convention. Their selector-ineligible adapter uses
only the common safe interior, `[STEP_START, STEP_END)`, records a one-frame
terminal exclusion when needed, and fails on wider crossings. Snapshot v2 has
no such endpoint ambiguity: it supplies canonical half-open
`stimulus_frame_num` intervals. Palette nevertheless does not convert those
intervals into acquisition-row selections until the exact frame-bound identity
chain is present. No production selector or maintained profile consumes either
the historical conservative candidate or the v2 sealed correspondence proxy.
