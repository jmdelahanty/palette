# Batman whole-recording keypoint canary v007

Date: 2026-08-05

Status: **pipeline/contract pass; scientific-yield failure; not eligible for
activation**

## Purpose

This selector-ineligible canary exercised the whole-recording keypoint
producer after adding the digest-bound pose-model input contract. It used the
newest selected five-keypoint model and the existing verified 348 x 348 flat
ROI cache for one Batman recording. The runtime transformed each native crop
by centered zero padding to the model's 512 x 512 training-source canvas and
then invoked the bound Ultralytics 256 x 256 network-input adapter.

The canary did not authorize selector or registry changes.

## Immutable inputs

- Palette commit: `e60d9c473cee485a7ac7fc73c81e1f0f8a35b3be`
- Recording: `2026-07-21T19-38-32Z_arena_2_Batman`
- Crop run: `crop_geometry_v2_348_20260805`
- ROI cache shape: `[126214, 348, 348]`, `uint8`
- Model set: `pose_all_registry_reviewed_v2_keypoints_20260520_v001`
- Model run: `pose_all_registry_reviewed_v2_kpt5_warm_v2_20260520_retry2`
- Weights SHA-256:
  `cce63d534a8f1491db1e2c71cb9236768c445722013dc39faeaf62a9d0a9a377`
- Model-input contract SHA-256:
  `ce7159a95afd16c35054e08fd5a7978e147148419b2b502dc457869958d8f0c4`
- Model-input contract payload digest:
  `e0eac62db3ede00668103a3f12b496da1b56c85dcf772d11558a76da97d31a78`
- Training Ultralytics version: `8.3.214`
- Runtime Ultralytics version: `8.3.169`
- Runtime preprocessing-probe SHA-256:
  `d141f8e12a791d6b4b0c99ae3dfc24c6d6c11b63f9739df755d1d7bbe4b1d35a`

The runtime version was accepted only after it reproduced the frozen
preprocessing probe exactly. The failed v006 attempt rejected the version
difference before cache staging; v007 used the strengthened version-plus-probe
compatibility contract.

## Jobs and outcome

| Stage | LSF job | Outcome |
|---|---:|---|
| prediction | `153283733` | done; terminal artifact complete |
| strict v2 refinement/finalization | `153283734` | done; four selector-ineligible runs imported atomically |
| candidate validation | `153283735` | done; one candidate structurally validated |

Prediction processed all 126,214 ROI rows in 274.0 seconds. The progress rate
was approximately 460.7 ROI rows/s. The terminal scientific result was:

- successful poses: `0`;
- failed poses: `126214`;
- `no_pose_detection_above_threshold`: `126214`;
- all other frozen failure codes: `0`.

This rules out missing keypoint payloads, empty payloads, and skeleton
keypoint-count mismatches. The preprocessing equivalence probe proves
implementation equivalence for the frozen synthetic input, not scientific
equivalence to the model's training-image distribution.

A subsequent identity-fixed, evenly spaced 128-row diagnostic identified the
synthetic border as the immediate incompatibility:

| Submitted profile | detections at 0.25 | at 0.05 | at 0.01 | at 0.001 |
|---|---:|---:|---:|---:|
| native 348 | 10 | 22 | 36 | 58 |
| 348 center-zero-padded to 512 | 0 | 0 | 0 | 0 |
| 348 center-padded to 512 with value 114 | 0 | 0 | 0 | 4 |

The model therefore still responds to native Batman crops, although native
yield is too low to promote without review. Lowering the threshold cannot
rescue the zero-padded profile. The correct next comparison is real 512x512
source-camera context around the same instance, not another synthetic fill
value.

## Publication safety result

The terminal artifact is complete because it represents every crop row with
an exact terminal failure code. The strict finalizer successfully created the
raw-keypoint, quality, refined-keypoint, and body-frame candidate runs so the
full storage/publication boundary could be validated. These are structural
candidate artifacts, not usable keypoint authorities.

The final candidate receipt records:

- `apply=false`;
- `activation_performed=false`;
- `registry_integrity_before == registry_integrity_after`;
- `registry_status=unchanged_candidate_validated`;
- `selector_status=unchanged_candidate_ineligible`.

Direct parent metadata checks found no `latest`, `latest_complete`, or
`pending` selector for any of the four run families. The only parent mutation
was the expected completion epoch associated with atomically importing a
validated, selector-ineligible child.

## Evidence locations

Run root:

`/groups/johnson/johnsonlab/jeremy/logs/whole_recording_keypoints/batman_kpt5_v2_canary_20260805_v007`

Key receipts:

- `terminal/2026-07-21T19-38-32Z_arena_2_Batman.zarr/terminal_receipt.json`
  (payload digest
  `bea18dbaa1b2477731d35d283ece6d0dc3e44e671b69101cda01c3814bf7f528`)
- `finalization/2026-07-21T19-38-32Z_arena_2_Batman.json`
- `registry/registry_finalizer.153283735.json`

## Required next gate

Do not activate these runs. Before another full-duration inference, run a
small, representative image-backed comparison that inspects the actual crop
content and model detections under at least:

1. native 512 x 512 crops reconstructed from source frames around the same
   detection centers;
2. the successful v005 native/stride-aligned profile; and
3. any legacy model-native crop generation used by successful training or
   prior inference.

The comparison must keep instance identity fixed, report per-profile success
rate and confidence distributions, and visually review representative edge,
center, and failed crops. A full recording should be resubmitted only after a
profile produces scientifically plausible keypoints on that bounded gate.
