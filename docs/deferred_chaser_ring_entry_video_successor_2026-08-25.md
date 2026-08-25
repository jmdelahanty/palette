# Deferred chaser ring-entry video successor — 2026-08-25

## Status

Deferred optimization/visualization work. This document records scope and
acceptance requirements; it does not authorize a selector, registry, production,
or deployment change.

The current exploratory renderer is
`src/fisheye/visualization/chaser_ring_traversal.py`. It can draw per-entry
trajectories and bout segments and can emit per-chaser MP4/GIF files, but its
public collectors intentionally fail closed pending sealed inputs. Its unsealed
inspection paths, `latest` discovery, nominal-FPS timing, and historical distance
thresholds are not publication evidence.

## Successor objective

Create a receipt-bound visualization successor over explicitly named immutable
chaser products. It should render the fish crossing distance rings around each
chaser, preserve every contributing acquisition-frame identity, and make the
static and moving-chaser coordinate frames scientifically explicit.

The renderer must support every sealed fish-position provider as a first-class
authority. `detection_bbox_centroid.v1`,
`keypoint_anatomical_triad_mean.v1`, and future providers are peers; no provider
may be labeled legacy or silently preferred by the renderer.

## Required event definitions

The video must not conflate these distinct scientific views:

1. **Near-field visits:** enter when distance is strictly below 5 mm; exit when
   distance is strictly above 6 mm. Dwell and entry rates use exact adjacent
   acquisition-frame session-timestamp intervals. Invalid timestamps, invalid
   distance rows, or nonadjacent frame IDs break continuity and censor an active
   visit. No interpolation or nominal-FPS substitution is allowed.

2. **Response-shell traversals:** show the 8–16 mm bout-response shell and the
   exact distance-bin policy of the bound generalized bout-response product.
   The existing 15 mm entry / 20 mm exit traversal definition may be retained
   only as a separately named, versioned event policy. It must never be reported
   as the 5/6 mm near-field visit definition.

Both definitions may be rendered in one package, but their labels, counts,
thresholds, and receipts must remain distinct.

## Exact inputs

Require explicit non-selector bindings to:

- one immutable chaser-relative-frame run and manifest digest;
- one immutable protocol-semantic selection and manifest digest;
- one reviewed arena geometry selection and physical-scale digest;
- one fish-position provider authority and digest, inherited without discovery
  from the relative-frame source;
- the generalized bout-response successor when bout segments are drawn;
- the escape/freeze successor when bouts are classified or styled as escape or
  freeze responses;
- an exact source-video identity only when camera pixels are decoded. A
  trajectory-only animation must say that no video pixels were consumed.

Every source must belong to the same recording. Coordinate-frame and scale
authorities must agree exactly. Session timestamps are acquisition-time evidence;
the receipt must keep `physical_presentation_verified=false` unless a later
authority establishes camera-exposure/presentation synchronization.

## Epoch and coordinate-frame behavior

- Render `chaser_pre`, `chaser_training`, and `chaser_post` only from their exact
  half-open semantic intervals.
- Static pre/post views may rotate into the existing object-at-origin frame, but
  must draw the reviewed wall geometry and label the transform.
- Training/random-wander views must use a per-frame moving chaser-centric frame;
  centering rings on a median or static chaser position is prohibited.
- Frames outside the exact semantic intervals, including an independent black
  baseline, require their own explicit semantic selection and cannot be silently
  folded into the chaser session.

## Outputs and receipt

Write only external, regenerable operation artifacts by default:

- per-chaser MP4 (and optionally GIF) for each requested epoch/event policy;
- a static contact sheet of retained entries;
- a typed entry inventory containing event ID, chaser identity and role, semantic
  epoch, start/end acquisition-frame IDs, exact timestamps, censor status, closest
  approach, bout IDs, and rendered-frame mapping;
- one strict JSON receipt binding every source run/digest, provider authority,
  geometry/scale authority, thresholds, event policy, renderer commit, output file
  SHA-256, and entry-inventory SHA-256.

The receipt and every output must state selector-ineligible,
non-authoritative, and registry-unchanged. Rendering must use a temporary path
followed by atomic rename and must never mutate an analysis Zarr or selector.

## Acceptance evidence

- deterministic in-memory tests for 5/6 mm hysteresis, 8–16 mm shell membership,
  invalid/nonadjacent gap censoring, exact timestamp duration, and moving-frame
  transforms;
- provider-parity tests showing detection/bbox centroid and keypoint-triad inputs
  pass the same contract without special cases;
- direct/deep source audit before rendering and content-hash verification after
  rendering;
- a controlled current-recording canary with manual visual inspection of at least
  one static and one moving-chaser entry for each available chaser role;
- fail-closed behavior when no valid entries exist. Empty panels or videos must not
  be fabricated.

## Current prerequisite discovered on 2026-08-25

The first keypoint-triad radial/near-field plot exposed unexpectedly sparse valid
distance coverage: only one training epoch/chaser series was populated. Determine
whether this is genuine source-provider validity/occurrence evidence or an overly
restrictive projection before using this recording as the video-successor canary.
