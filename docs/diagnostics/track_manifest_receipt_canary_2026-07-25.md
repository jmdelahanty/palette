# Track manifest receipt canary, 2026-07-25

## Decision

The receipt-backed full-motion manifest shortcut tested by Sleepyfish canary
`track_kinematics_sleepyfish_cam2010095_manifest_receipt_canary_20260725_v001`
is rejected. It must not be used for production publication or public-reader
authority.

The independent exhaustive reader (LSF job `153173486`) rejected the run:

```text
ValueError: Track full-motion manifest differs from the exact live payload,
domains, aliases, derivations, or authorities.
```

The candidate is selector-ineligible, and selectors were restored to the
previous independently accepted run
`track_kinematics_sleepyfish_cam2010095_scientific_receipt_canary_20260724_v003`.

## What passed

- Materialization completed on host `h07u31` as LSF job `153173472`.
- The 1,169,010-row decoded payload root remained exactly
  `0d246d1df9424314bd7c9c2cd9246fb64206fb3f38961522e64447bd87bab6e3`.
- Scientific totals matched the accepted comparison run.
- Four allocated slots were sufficient. The workflow submission default remains
  four slots, while explicit `--ncores` overrides remain available.

## Performance result

| Phase | Accepted baseline | Rejected receipt canary |
|---|---:|---:|
| Entire materializer | 320.34 s | 324.40 s |
| Authoritative publish | 123.69 s | 118.07 s |
| Post-rename binding | 33.83 s | 34.82 s |
| Completion and pointer publication | 73.99 s | 67.29 s |
| Shard materialization and decoded validation | 3.79 s (8 workers) | 7.22 s (4 workers) |

The modest publication reduction did not offset other runtime variation, and
the public-reader failure makes the shortcut unusable regardless of speed.

## Safety boundary

An exact decoded output-payload receipt is not sufficient authority for a
full-motion publication manifest. The public contract also binds live row
identity, temporal lineage, source authority, domains, aliases, derivations,
and coordinate records. Reusing output hashes without an exact receipt for
those relationships can mint a self-consistent manifest that differs from a
fresh public read.

Until a stronger binding design is specified and independently tested, full
motion manifest construction must hash and validate every live published array.
The existing external exhaustive-reader canary remains a required acceptance
check for future publication optimizations.
