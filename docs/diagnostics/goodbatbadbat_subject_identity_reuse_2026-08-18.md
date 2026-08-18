# GoodBatBadBat subject identity reuse incident

Status: diagnostic and incident record; no production correction applied.

Incident date: 2026-08-18

Operator decision timestamp: 2026-08-18T12:31:57Z

## Incident summary

The cohort `goodbatbadbat_keypoint_triad_talk_20260818_v2` contains 84
recordings. The operator has confirmed that all 84 recordings represent
distinct animals. The acquisition-time subject UUID assignment nevertheless
reused eight UUIDs across pairs of consecutive recordings in the same arena.
The reuse affects 16 recordings.

The current cohort manifest therefore describes 76 UUID-distinguished subjects
even though the biological cohort contains 84 distinct animals. This is an
acquisition provenance error, not evidence that any animal was recorded twice.

## Exact collisions

Each UUID below was assigned to both recordings in its pair:

| Capture-time subject UUID | First recording | Second recording |
|---|---|---|
| `70544de4-ea39-4be8-8e79-018ba1d1bb5a` | `2026-08-11T19-44-55Z_arena_4_goodbatbadbat` | `2026-08-11T20-20-15Z_arena_4_goodbatbadbat` |
| `7b52d822-2bae-46f0-9578-0d7ecaaba3b7` | `2026-08-10T18-56-00Z_arena_4_goodbatbadbat` | `2026-08-10T19-32-48Z_arena_4_goodbatbadbat` |
| `7dae89ef-8915-45f1-8341-b322fd2b33e7` | `2026-08-10T18-56-00Z_arena_1_goodbatbadbat` | `2026-08-10T19-32-48Z_arena_1_goodbatbadbat` |
| `7ee86b08-2390-4c63-925f-80c7067323e7` | `2026-08-11T19-44-55Z_arena_2_goodbatbadbat` | `2026-08-11T20-20-15Z_arena_2_goodbatbadbat` |
| `96684728-f284-4c9e-9da8-1141c687aa4a` | `2026-08-10T18-56-00Z_arena_3_goodbatbadbat` | `2026-08-10T19-32-48Z_arena_3_goodbatbadbat` |
| `9ef56cb5-9feb-4d23-ad3e-9c7ede3718a6` | `2026-08-11T19-44-54Z_arena_1_goodbatbadbat` | `2026-08-11T20-20-15Z_arena_1_goodbatbadbat` |
| `c90fb69f-9f6e-4cf7-ad4a-ea1d8688b334` | `2026-08-11T19-44-55Z_arena_3_goodbatbadbat` | `2026-08-11T20-20-15Z_arena_3_goodbatbadbat` |
| `dfbbe1f4-f2bc-4883-b5da-1f4943a9ebf0` | `2026-08-10T18-56-00Z_arena_2_goodbatbadbat` | `2026-08-10T19-32-48Z_arena_2_goodbatbadbat` |

## Impact

The prior 76-subject grouped figures are invalid for subject-balanced
interpretation because eight distinct animal pairs were collapsed into one
subject identity. They must not be presented as the canonical 76-animal
cohort result.

The following remain usable as recording-scoped outputs:

- the 84 per-recording analysis units; and
- the 91,770 merged per-bout rows.

The per-bout rows and recording-level measurements were not changed by this
incident. However, any analysis that groups, averages, or models by the
capture-time subject UUID must treat the affected identities as unresolved
until the correction is completed.

## Urgent analysis disposition

For time-sensitive analysis, the approved workaround is to use
`recording_id` as the recording-by-animal unit and set the analysis sample size
to `n=84`. This is a temporary analytical unit, not a biological subject UUID
and not a canonical identity correction.

This workaround is authorized by the operator decision recorded at the
timestamp above. Any exported figure or table using it must state that it is
recording-scoped and bind the cohort manifest, analysis run, and this incident
decision. It must not claim that the capture-time UUIDs have been corrected or
that the UUID-based subject registry is authoritative.

## Canonical correction boundary

The correction must follow the subject identity contracts in
[`docs/subject_metadata_identity_corrections.md`](../subject_metadata_identity_corrections.md)
and the subject-correction section of
[`docs/experiment_setup_contract.md`](../experiment_setup_contract.md).

Do not:

- invent replacement UUIDs in Palette;
- rewrite source H5 files or capture-time metadata;
- edit completed subject-metadata or experiment-setup runs;
- directly mutate `subjects` or `recording_subjects` registry identity rows;
- overwrite the existing cohort manifest or old exports; or
- reinterpret `recording_id` as a biological subject identifier.

MetaZebrobot must mint or verify the corrected biological identities. The
canonical repair then requires a Palette immutable subject-metadata successor
and a paired experiment-setup successor, each with explicit parent lineage,
digests, correction reason, operator identity, review timestamp, and the
MetaZebrobot registration record and verification assertion.

If acquisition or MetaZebrobot evidence establishes that the first recording
kept its UUID and the second recording received the reused UUID, the likely
repair is eight replacement UUIDs for the second members of these pairs. That
is only a conditional repair hypothesis; it is not an approved mapping. If
ownership of either UUID cannot be established, both members of the pair must
be explicitly reviewed. No replacement UUIDs are specified in this document.

The current Palette repository documents the successor boundary, but the
production successor publisher and pair activation protocol are not
implemented. Consequently, no canonical correction or registry activation can
be performed yet.

Existing subject/setup runs, detection and analytics runs, registry rows, and
old exports remain immutable evidence. A future corrected export must retain
both the source-bound digests of the derived run and the active correction
authority so that acquisition provenance and corrected biological identity
remain distinguishable.

## Prevention for future acquisition

The acquisition workflow should:

1. mint and register a UUID for each new biological animal before its recording
   begins;
2. bind that registration evidence to the recording snapshot;
3. fail closed when a UUID is reused in a new recording, unless the operator
   explicitly declares a permitted repeat-session relationship; and
4. persist the registration response, UUID, recording identity, arena, and UTC
   timestamp as immutable acquisition evidence.

Automatic UUID generation removes the need for an operator to manually invoke
UUID creation, but it must still be registration-aware and collision-checked.
Generation alone is not proof of biological identity.

## Prompt for the acquisition/MetaZebrobot agent

Please investigate and prepare the evidence for the
`goodbatbadbat_keypoint_triad_talk_20260818_v2` subject-identity reuse
incident. The operator confirms that all 84 recordings are distinct animals;
eight capture-time UUIDs were each reused across the two consecutive
same-arena recordings listed in the Palette incident document
`docs/diagnostics/goodbatbadbat_subject_identity_reuse_2026-08-18.md`.

For every affected pair, inspect the acquisition registration evidence and
MetaZebrobot records and determine whether the first UUID is valid for the
first recording, whether the second recording needs a newly minted UUID, and
whether either ownership is uncertain. Do not invent UUIDs, edit source H5s,
rewrite completed runs, or mutate registry identity rows. Return a
digest-bound correction manifest containing the recording IDs, original UUIDs,
verified MetaZebrobot record IDs, any newly minted replacement UUIDs, evidence
digests, reviewer, reason, and UTC review timestamp. If a pair cannot be
resolved, mark both members for explicit review.

Also update the acquisition path so a UUID is automatically minted and
registered before each new animal recording, reuse fails closed unless an
explicit repeat-session declaration is present, and the registration response
is snapshotted with the recording. Palette can consume the resulting verified
manifest only after its immutable subject/setup successor publisher and paired
activation contract are available.

## Data-preservation statement

This document records the incident and the temporary analysis disposition
only. No production data, source H5, completed subject/setup run, registry row,
or existing export was modified by this documentation change.
