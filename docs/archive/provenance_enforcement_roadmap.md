<!-- ARCHIVED 2026-07-17: implementation roadmap superseded by active completion and provenance contracts. -->

# Provenance enforcement roadmap

<!-- contract-meta
status: active
created: 2026-07-02
owner: jeremy
related: docs/diagnostics/codebase_review_2026-07-01.md,
         docs/palette_cli_narrow_waist_design.md,
         docs/identity_lineage_staleness_review.md
-->

## The framing: capture improved, enforcement did not

As of 2026-07-02 the pipeline *captures* provenance in many more places than it did
at the 07-01 review — the `palette` run-verbs stamp `cli_provenance`
(git_sha/config_hash/params) into crop/detect/keypoint runs, authoritative-run
pointers stamp approved_by/approved_at/git_sha, and the silent-wrong-data slice stamps
decode contracts. But provenance is still **stamped-if-you-use-the-waist, not
required-if-absent**, and the production path bypasses the waist.

**The structural hole:** `cli_provenance` defaults to `None`
(`run_detect_with_registry_model.py:362`), and the bsub cluster scripts call the
runners directly rather than through `palette`. So **production cluster runs carry no
code provenance.** Stamping happens at the CLI convenience layer; the production path
skips it.

**The fix principle:** move enforcement *down* from the waist to the **finalization
layer**, where every path converges. Both `palette detect` and a bsub job ultimately
call the same run-completion marker. That marker — not the CLI — is where "cannot be
marked complete without X" belongs. Enforce there and the bsub hole closes for free.

## The five enforcement levers (current state)

Status pointer: see docs/diagnostics/codebase_review_2026-07-01.md ("Remediation
delta") for the live, sourced count — the table below is corrected to match it as of
2026-07-04 but will drift again, so treat the scoreboard as authoritative.

| Lever | State (2026-07-04) | File |
|---|---|---|
| Stage-array validation | **Slice 1 partially landed** — enforcement (raise, not just warn) is live for 7 stages: `arena_assignment`, `crop`, `detect`, `detect_quality`, `keypoints`, `refined_keypoints`, `tracking` (verified via `_ENFORCE_STAGE_ARRAY_VALIDATION_FOR` frozenset). "Toward 100%" continues — not yet every non-deprecated specced stage. | `registry/stage_complete.py:64` |
| Required provenance at finalization | None — `cli_provenance` optional, defaults None, bsub skips it | `run_*_with_registry_model.py` |
| `legacy_default` strict mode | Still `True` — uninstrumented runs pass as complete | `shared/zarr_run_completion.py:178` |
| Content hashes vs mtime | Still mtime (`zarr_mtime_ns`) | `registry/status_ledger.py` |
| `subject_mask_data_profile` | Does not exist | — |

## Sequenced slices

### Slice 1 (first — safest high-leverage): stage-array validation → toward 100%

**Partially landed** — the promotion allowlist has grown from 1 stage
(`detect_quality`) to 7 (see the lever table above). The method below continues to
apply to the remaining non-deprecated specced stages.

Validation already runs for every stage with a `StageSpec`; results land in
`details_json` as warnings. Only the allowlist
`_ENFORCE_STAGE_ARRAY_VALIDATION_FOR` gates the *raise*. The code comment already
prescribes the promotion path: shadow-telemetry-confirm the writers emit required
arrays, then add the stage.

Method (evidence-driven, not a blind dump): for each non-deprecated specced stage that
goes through finalization, gather real-run evidence (the existing warning telemetry over
actual stores) that writers always emit the required arrays. Classify:
- **complies** → promote into the enforce set;
- **does not comply** → a latent bug (writer omits a required array, or the spec is
  wrong) — report, fix the writer or the spec, do not promote until fixed;
- exclude deprecated stages (eye_masks/refined_eye_masks) — being deleted.

Each promoted stage flips malformed-run detection from warning to hard refusal; each
failure surfaced is a real data-correctness bug. Finite, incremental, low-risk.

### Slice 2 (bigger): required provenance at finalization

A run cannot be marked complete without `{git_sha, code_version, config_hash}`, enforced
at the completion marker so **both** the waist and bsub paths are covered — closing the
production hole. Requires a `legacy_default → strict` epoch keyed on a store schema
version so existing stores still read as complete (otherwise every historical run
retroactively invalidates). That epoch mechanism is also what finally lets `legacy_default`
flip to strict (lever 3). Also thread provenance into the bsub runner-invocation path so
cluster runs populate it going forward.

### Slice 3 (lower urgency): content hashes

Status: landed in `agent/provenance-content-hashes`.

Move artifact identity from `stat_v1` (path+size+mtime) toward content hashes, or
explicitly mark weak fingerprints as weak. The run-provenance payload now carries
non-gating `input_artifacts` entries for model/checkpoint inputs (`content_v1`
file hashes and `manifest_v1` runtime manifests where exact checkpoint files are
not exposed), and deployment artifacts have a read-only verification helper for
comparing registry hashes to on-disk files. This remains a recorded audit field;
the epoch-2 finalization gate still blocks only on `git_sha` and `config_hash`.

### Slice 4 (feature, not enforcement): `subject_mask_data_profile`

Give the strategic subject-mask stage the data-profile path the deprecated eye-mask
stage had. Best folded into the registry reconcile/profile work.

## Sequencing rationale

Slice 1 first: it's the designed mechanism, surfaces bugs while it enforces, and is the
lowest-risk. Slice 2 next: it needs the epoch mechanism and benefits from Slice 1 having
proven the finalization-gate pattern. 3 and 4 after. Every slice converts something this
session merely *stamped* into something the pipeline *requires* — which is what makes the
waist, the authority pointers, and the decode census trustworthy by construction rather
than by convention.
