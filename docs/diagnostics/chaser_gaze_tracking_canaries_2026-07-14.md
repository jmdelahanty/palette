# Chaser gaze-tracking canaries, 2026-07-14

## Scope

This diagnostic established convention and implementation gates. It is not a
cohort result and does not support a biological conclusion about gaze tracking.

One long non-chaser recording and two GoodCopBadCop recordings were used. The
two chaser canaries deliberately came from different recording dates and
arenas. Eye-angle computation ran through the analysis DAG on Citrus; gaze
components were built read-only while the implementation was still uncommitted.

## Eye convention validation

All required numeric identities passed on 3,072 bounded sample rows per
recording:

| Recording | Eye run | Total eye rows | Valid sampled rows | Marginal sampled rows |
| --- | --- | ---: | ---: | ---: |
| `sleepyfish_2026_05_05_17_45_30_cam2010095` | `eye_angles_sleepyfish_core_canary_20260713_01` | 1,169,010 | 3,067 | 0 |
| `2026-06-14T21-12-08Z_arena_1_GoodCopBadCop` | `eye_angles_gaze_canary_20260714_a1` | 120,221 | 3,023 | 0 |
| `2026-06-21T18-18-32Z_arena_4_GoodCopBadCop` | `eye_angles_gaze_canary_20260714_a4` | 127,122 | 3,025 | 0 |

The bounded overlay grids were visually reviewed and the mask ellipses, body
axes, and directed gaze axes were plausible. The ellipse forward-half-plane
direction assumption remains explicit and is not considered numerically
validated.

The GoodCopBadCop DAG jobs were LSF jobs `153097202` and `153097203`; both
completed successfully.

## Read-only gaze builds

The arena-1 canary contained 140,035 chaser-distance frame rows. Its empirical
gaze ranges were approximately 54.0 to 96.7 degrees for the left eye and -98.2
to -56.3 degrees for the right eye. The independent eye-body versus track
heading gate used 115,712 jointly valid rows: median absolute disagreement was
0.51 degrees, the 95th percentile was 5.70 degrees, and circular resultant
length was 0.9978. The build found 197 sustained candidate lock-on events.

The arena-4 canary contained 139,877 frame rows. Its empirical ranges were
approximately 50.8 to 99.6 degrees left and -100.6 to -52.6 degrees right. The
heading gate used 124,769 rows: median absolute disagreement was 0.59 degrees,
the 95th percentile was 6.75 degrees, and resultant length was 0.9950. The build
found 194 sustained candidate lock-on events.

Both recordings resolved two protocol roles, `aggressive` and `inert`. Most
pre/post static gains were near zero. Some training cells had little accessible
support; one inert/right-eye cell had only three accessible frames and correctly
returned no regression estimate. Raw lock fractions could be large, whereas
real-minus-rotated-virtual excesses were generally much smaller. This is the
expected warning that wall geometry and body orientation can create compelling
raw alignment without object-specific eye tracking.

## Rollout gates

Before interpreting a cohort result:

1. Commit and publish the exact implementation used by cluster tasks.
2. Persist and inspect the two canary components using that clean commit.
3. Snapshot the production cohort from the registry by exact protocol, source
   origin, and prerequisite status; do not select by a filename substring.
4. Run remaining eye-angle stages, convention checks, and gaze components.
5. Treat sparse eye/range/epoch cells as missing rather than zero.
6. Aggregate at recording/fish level and retain rotated-reference counts and
   heading-gate diagnostics in the cohort table.

## Initial cohort execution diagnostic

LSF array `153097370` snapshotted exactly 32 source GoodCopBadCop recordings
with the required modern mask, keypoint, and track-kinematics stages. Twenty-seven
tasks completed the eye and gaze component on the first pass. Five exited
without publishing a partial gaze component:

- two were the established canaries, where the DAG correctly refused a second
  differently named eye run;
- two had a dense `frame_angles` projection ending shortly before the final
  chaser-distance frame; and
- one older egocentric component lacked persisted role labels.

The newly generated eye runs and all convention reports from this array remain
valid. The gaze consumer was hardened to bound the dense frame projection and
mark uncovered recording-tail frames unavailable. Persisted normalized role
labels remain preferred; older components fall back to the authoritative source
stimulus protocol and still refuse unresolved/unknown roles. Real-data
read-only reruns passed both compatibility cases before recovery submission.
