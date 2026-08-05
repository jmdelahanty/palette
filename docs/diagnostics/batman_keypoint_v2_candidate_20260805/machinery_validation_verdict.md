# Batman keypoint-v2 machinery validation verdict

Date: 2026-08-05

Verdict: **PASS for selector-ineligible pipeline machinery; FAIL for current-model
scientific promotion**

## Decision boundary

The whole-recording keypoint machinery is sufficiently validated to support
continued development, bounded canaries, and successor-model evaluation while
the pose model is retrained. This verdict does not approve a keypoint authority,
production selector, registry activation, or broad Batman processing.

The two decisions are intentionally separate:

- machinery correctness asks whether exact inputs reach complete, validated,
  immutable outputs without identity loss or accidental activation;
- scientific acceptance asks whether the selected model produces sufficiently
  accurate and complete poses for the intended recording domain.

The first passed. The second failed for the current contract-bound model/input
pairing.

## Accepted machinery gates

| Gate | Evidence | Verdict |
|---|---|---|
| Commit-pinned cluster execution | Detached Palette deployments at `9598f402`, `e60d9c47`, and `575b272d`; shared `/groups` checkout unchanged | pass |
| Exact model selection | Model set, run, weights path, and SHA-256 bound in plan and terminal receipts | pass |
| Model-input contract | Training source shape, network shape, stride, runtime adapter, package artifacts, and payload digest validated before inference | pass |
| Runtime fail-closed behavior | v006 rejected unreviewed Ultralytics version before cache staging | pass |
| Runtime compatibility proof | Reviewed `8.3.169` runtime reproduced preprocessing probe SHA-256 `d141f8e12a791d6b4b0c99ae3dfc24c6d6c11b63f9739df755d1d7bbe4b1d35a` | pass |
| Crop/cache identity | Exact crop-v2 run, 126,214 rows, instance keys, cache manifest, payload size, and payload SHA-256 validated | pass |
| Node-local execution | 15 GB flat cache copied and verified on node-local scratch before inference; temporary job data cleaned | pass |
| Successful-row path | v005 produced 89,527 successful pose rows and 36,687 explicit failures | pass, legacy input profile |
| All-failure path | v007 represented all 126,214 rows with exact `no_pose_detection_above_threshold` codes | pass |
| Terminal completeness | Every requested crop row reached one terminal success/failure state; receipt and array digests sealed | pass |
| Strict four-surface finalization | Raw keypoint, quality, refined keypoint, and body-frame runs built and independently validated | pass |
| Atomic publication | Candidate children imported by hidden same-parent staging and atomic rename | pass |
| Storage contract | Exact schemas, dtypes, manifests, storage plans, and direct/consolidated declarations validated | pass |
| Activation safety | Candidate runs remained selector-ineligible; no `latest`, `latest_complete`, or `pending` selector moved | pass |
| Registry safety | Final validator used `apply=false`; registry integrity before and after was identical | pass |
| Consumer interoperability | Crimson opened exact typed keypoint-v2 surfaces, retained offsets once, traversed and rendered v005, and reported zero stale frames | pass |
| Real-context diagnostic | Read-only v002 materialized digest-bound 512x512 source-camera context for exact identities without archive, selector, or registry mutation | pass |

## Scientific-model failure

The current historical model is not approved for the intended production input
policy.

- v007's contract-bound 512-to-256 path produced zero successful poses over
  126,214 rows.
- The failure histogram contained 126,214
  `no_pose_detection_above_threshold` rows and zero payload/schema failures.
- The bounded real-context gate selected 128 rows that were all successful in
  v005 and proved exact crop-row and instance-key equality.
- Native 348, synthetic 512, and real-source 512 profiles all produced zero
  detections at `imgsz=256`, even at `conf=0.001`, on those identities.
- The same identities succeeded through v005's persisted 348-to-352 tensor
  profile.

This localizes the current blocker to model/domain/effective-scale suitability,
not storage, orchestration, identity, publication, or Crimson decoding.

## Immutable evidence

### Successful-path canary

- Palette commit: `9598f402e27c18b5ff2dfc390811cc0472a5eaec`
- Jobs: `153273676`, `153273677`, `153273678`
- Result: 89,527/126,214 successful poses
- Crimson evidence commit reported by Crimson:
  `f4edbff7b5c3e6d341395f35092b2a8997d5c3d5`

### Current contract/failure-path canary

- Palette commit: `e60d9c473cee485a7ac7fc73c81e1f0f8a35b3be`
- Jobs: `153283733`, `153283734`, `153283735`
- Run root:
  `/groups/johnson/johnsonlab/jeremy/logs/whole_recording_keypoints/batman_kpt5_v2_canary_20260805_v007`
- Terminal receipt payload digest:
  `bea18dbaa1b2477731d35d283ece6d0dc3e44e671b69101cda01c3814bf7f528`

### Real source-context benchmark

- Palette implementation commit:
  `575b272d5a5d1d1a725f5c40cb72f5d291226a71`
- Job: `153283778`
- Result:
  `/groups/johnson/johnsonlab/jeremy/logs/whole_recording_keypoints/batman_pose_source_context_20260805_v002/result.json`
- Result SHA-256:
  `ef28414260e700447bf07a578780204d1287267bdf99f716f73ccdc114e98de2`

## Authorized work while retraining

The machinery verdict supports:

- producing a successor model package with a directly emitted model-input
  contract;
- running small selector-ineligible successor-model gates;
- comparing successor outputs against reviewed Batman holdouts;
- using the existing atomic terminal/finalization/publication path unchanged;
- asking Crimson to validate one successful successor canary before activation.

It does not support:

- activating v005 or v007;
- treating structural completeness as pose usability;
- processing the remaining 35 Batman recordings with the current model;
- lowering confidence thresholds to manufacture apparent yield; or
- changing the historical model's declared training contract after the fact.

## Successor-model activation gate

Once retraining finishes, only the model-dependent portion needs repeating:

1. publish the model package and directly emitted input contract;
2. run a bounded Batman holdout gate with reviewed success/accuracy thresholds;
3. run one selector-ineligible full-recording canary through the already
   validated machinery;
4. have Crimson validate exact typed reads, alignment, traversal, cancellation,
   and a source-matched visual sample;
5. review scientific yield and quality independently of machinery health; and
6. activate the four selectors only through the separate atomic activation
   operation after all gates pass.

