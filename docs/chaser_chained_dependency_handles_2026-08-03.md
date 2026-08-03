# Chaser chained dependency handles — 2026-08-03

This checkpoint closes two maintained chained-input boundaries without changing
production selectors or storage profiles.

## Exact maintained inputs

- `chaser_gaze_tracking` accepts
  `--egocentric-dependency-handle-json`. The handle must be a canonical,
  self-digested `palette.chaser_component_dependency_handle` for an
  `egocentric_bearing` component with the exact v1 semantic schema.
- `chaser_near_field_occupancy` accepts
  `--quadrant-occupancy-dependency-handle-json`. The handle must bind an exact
  `chaser_quadrant_occupancy` v1 component.

Both readers validate the handle against the selected chaser-distance base,
open its exact component path, and revalidate the persisted component manifest
and payload. A wrong family, path, manifest digest, handle digest, or base
publication seal is terminal. An invalid explicit handle never falls back to a
selector, child scan, or historical name.

The resulting gaze and near-field publications record the exact upstream base
run/path and component manifest SHA-256 in their scientific `source_refs` and
run-lineage payloads.

## Historical compatibility boundary

Historical component-name and `latest` discovery is available only through an
explicit opt-in:

- `--legacy-egocentric-component-compatibility`
- `--legacy-quadrant-occupancy-component-compatibility`

The legacy path records no component manifest digest. It is not the maintained
candidate-publication path and does not grant selector eligibility.

The batch near-field runner accepts the same handle JSON. For multi-recording
work it can also mint one exact handle per archive from an explicitly named,
sealed quadrant component; `latest` is rejected unless the legacy flag is
present. The historical `cra_primary_endpoint_component` wrapper is itself an
explicit compatibility API: using that alias sets the legacy policy, while
combining it with an exact handle is rejected.

## Non-goals

This checkpoint does not activate a selector, change a production profile,
modify a registry, or make a selector-ineligible candidate authoritative.
