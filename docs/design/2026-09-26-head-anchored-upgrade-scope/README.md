# Mask Apply selects each row's tail method: legacy first, head-anchored fallback

- **Status:** accepted
- **Owner:** labeling/Apply work (session palette-12, for the user); last reviewed 2026-09-26
- **Builds on:** `docs/training_head_anchored_tail_geometry.md` (#192, merged at c7026aeb)
- **Classification:** scientific/derivation-selection change, versioned as tail successor format v3. It applies to training review successors only; production subject-shape defaults are unchanged.

## Problem

#192 added the head-anchored tail method (`head_anchored_skeleton_path_v1`, recipe `…tail11_v4`), but only as a manual, targeted refresh. It covered only `snout_extension_too_long` rows, and only on sources not already upgraded.

An ordinary mask Apply re-derived each row with the row's recorded method, which is legacy for almost every row. A row the legacy method cannot join therefore failed on every Apply, however well its mask was fixed.

Example: 2026-01-28T20-51-00Z arena 3, ROI 0, a tightly curled fish. Legacy fails with `snout_extension_no_mask_path`, while head-anchored gives a valid tail (overlay reviewed by the user on 2026-09-26).

## Design

1. **Successor format v3**, policy `new_mask_seed_and_review_version_reference_crop_sharded_head_anchored_fallback_v3`, becomes the default for mask Apply. It keeps v2's layout: a crop reference and whole-run shards.
2. **Per-row method selection** from the current applied masks and saved head points (`legacy_first_head_anchored_fallback_v1`):
   - Derive with legacy.
   - If legacy fails with a snout-join reason (`snout_extension_too_long` or `snout_extension_no_mask_path`), derive with head-anchored.
   - Use head-anchored (code `1`) only if it yields a valid tail. Otherwise keep legacy (code `0`) and its failure reason.
   - Other failures, such as a fragmented body mask or the body touching the crop border, are never retried: they need a mask fix or an acceptance.
3. **Selection is stateless.** Every Apply recomputes it from the masks, so a row goes back to legacy if legacy later succeeds. Crop-border acceptances apply to both methods.
4. **Recorded provenance:**
   - The seed and edit runs store `tail_derivation_method_code` per row, with the v4 recipe's method mapping.
   - The proof records `tail_method_selection`: the policy, the fallback reasons, the digest of the row method codes, and the head-anchored rows.
   - `source_row_method_codes_sha256` still binds the source's codes when the source was upgraded.
5. **Compatibility:**
   - v1 and v2 successors stay readable and valid.
   - An Apply whose earlier partial publication used v1 or v2 resumes in that format, so the same source keeps the same version.
   - The explicit #192 upgrade function still works but is no longer needed for these rows.
6. **Only the 19-landmark schema has the head-anchored method.** Legacy-schema reviews stay legacy under v3.

## Preservation evidence

- **Frozen legacy golden:** `redscare_head_anchored_legacy_golden_v1.npz` is unchanged.
- **All successor behaviours hold under v3** (the format matrix now runs v3, v2 and v1): manual points and clears, preserved labels, refusal on source change, training eligibility, crop-border acceptance, and supersede and carry-forward.
- **v3 against v2 from the same source:** rows the legacy method derives stay legacy, and every per-row array is byte-identical. The reason labels are compared decoded, since that column is fixed-width and sized to the longest label.
- **Rule tests on both fixtures:** RedScare long-snout rows, and the arena 3 no-mask-path row (`tests/fixtures/arena3_snout_no_mask_path_row_v1.npz`). A controlled test checks that the fallback is taken only when head-anchored succeeds, and that non-snout-join failures are never retried. Both mutants (ignoring head-anchored validity; switching every row) are caught.
- **Real-storage replay** on private copies of the live archives, v2 versus v3 from each current source:

| Recording | Method changes | Still failing | All other rows | Apply time |
|---|---|---|---|---|
| 20-51-00 arena 3 | ROI 0 to head-anchored, now valid | none | identical content (the reason column width changed 70 to 64 bytes because ROI 0's failure label went away) | 23.3 s to 25.5 s |
| 21-18-51 arena 1 | none | ROIs 47 and 48 (`fragmented_subject_body_mask`, one stray pixel each; fixed with the #222 editor tools) | byte-identical | 25.6 s to 27.7 s |
| 20-51-00 arena 2 | none (ROIs 8, 93 and 96 were already head-anchored via #192; v3 recomputes the same choice) | none | byte-identical | 22.7 s to 25.0 s |

## Rollout

- Code merges through the normal queue with full CI. The deploy is Tier 2.
- New mask Applies then write v3 successors. No existing version is rewritten, and no production analytics are rerun.

## Decision log

- 2026-09-26: accepted by the user: "old first, head-anchored fallback" on every mask Apply, not head-anchored for every row, and not a wider manual upgrade.
- 2026-09-26: opened as a proposal to widen the manual #192 upgrade (reason list plus re-upgrade). Superseded by the decision above.
