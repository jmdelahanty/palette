# Frame clock v2: read Orange's declared camera timescale

Status: **accepted and implemented** (2026-10-12). Jeremy confirmed the scope
and both open questions directly to Palette: Q1 yes (a contradiction refuses
publication), Q2 yes (`pmc_exit_status` must be 0). The scope is agent-contracts
`orange-palette/camera_exposure_host_clock_v1_2026-10-12.md` (main `bde1d4f`,
status approved). Owner: Palette intake (Palette H5 import readiness).
Classification: a scientific/identity semantics change to the single
frame-clock authority, so it gets a new record version.

## 1. What changes, and what does not

**In scope (contract gap 1):** for the camera timestamp, Palette reads
Orange's declared classification, timescale and TAI−UTC offset. It stops
inferring them when the declaration is direct.

**Out of scope:** the §4 gates (PTP offset 3 µs flag / 10 µs refusal,
receive floor, frame spacing) and publishing the host-clock exposure time.
Palette keeps its current looser PTP check unchanged.

**Unchanged:**
- The stored arrays, their digests and dtypes.
- `timestamp_sys` semantics: host `CLOCK_REALTIME`, POSIX UTC.
- The 256 published v1 archives. They keep their v1 records byte for byte
  and keep verifying.

## 2. Why a new version

Registration and every live re-verification rebuild the clock record from
the source (`acquisition_frame_clock_source_sha256`). They then require it
to equal the published record. Changing how v1 derives `clock_surfaces`
would make every existing archive fail that comparison. So:

- New publications write `palette.acquisition_frame_clock.v2`.
- Readers accept v1 and v2.
- Re-verification rebuilds a record with the **same version as the
  published one**. v1 derivation stays frozen.

## 3. Source fields read (Orange `recording_session.json`)

`timestamp_clock_contract.clocks["camera_<serial>"]`:

| Field | Use |
| --- | --- |
| `classification` | `"ieee1588_tai"` required for a declaration |
| `timescale`, `origin` | `"TAI"`, `"1970-01-01T00:00:00 TAI"` |
| `semantic_authority` | recorded verbatim |
| `inference.rule_id` | recorded (`orange_camera_clock_classification_v3`) |
| `inference.result` | declaration only if `"direct_ptp_time_properties_pass"` |
| `inference.expected_tai_minus_utc_seconds` | the declared offset |
| `inference.checks` | every value must be `true` |
| `evidence_snapshot.host_ptp_management_evidence.time_properties` | `currentUtcOffset`, `currentUtcOffsetValid`, `ptpTimescale` |
| `evidence_snapshot.host_ptp_management_evidence.captured_at_utc`, `pmc_exit_status` | recorded |

Orange checked these paths against a real recording made after the fix: the
2026-10-11 gate recording
`fourcam_fused_recorder_realfish_int8_192_native_ownerpush_ring_od8ff_s16_integration_20261011_215128`,
cameras 2010093–2010096. Every path matches. Observed values:

- `origin` is `"1970-01-01T00:00:00 TAI"`, with a space. Palette's v1
  vocabulary spells it with an underscore, so the record keeps the declared
  string verbatim in `orange_declaration` and the v1 spelling on the surface.
- `inference.checks` has 12 keys, all booleans.
- `currentUtcOffset` is an integer, 37. `currentUtcOffsetValid` and
  `ptpTimescale` are booleans, not 0/1.
- `pmc_exit_status` is 0.
- The recording start comes from top-level `recording.started_at_utc`, not
  `created_at_utc`. In that recording the evidence was captured 10 s before
  the start, because the headless runner refreshes it per run.

Also present but not read: `host_ptp_management_evidence.available`,
`schema_id` `orange.host_ptp.management_evidence` v1, `source_path`,
`capture_method`, `inference.thresholds_ns` and
`semantic_authority_by_property`. Palette has not yet read this recording
itself; the first such delivery to reach intake is Palette's own check.

## 4. Rule

**Declared** when all of the following hold:
- `classification == "ieee1588_tai"`;
- `inference.result == "direct_ptp_time_properties_pass"`;
- every `inference.checks` value is `true`;
- `time_properties.currentUtcOffsetValid` is `true`;
- `time_properties.ptpTimescale` is `true` (boolean);
- `time_properties.currentUtcOffset == inference.expected_tai_minus_utc_seconds`.

**Inferred** (today's v1 rule, unchanged) when the declaration is absent,
when `inference.result` is anything else (`fallback_inference_pass`,
`device_defined`, ...), or for any recording before the ptp4l change.

**Cross-check, declared case:** Palette still measures the median
camera−system delta. It must lie within tolerance (1 s, as v1) of the
declared offset; the ~9 ms receive delay is well inside that. A contradiction **refuses**
publication (Q1, decided).

**Freshness (Orange's correction):** `captured_at_utc` can be hours or days
older than the recording. A ptp4l restart in between is not visible. Palette
records `captured_at_utc` and the recording start next to the declaration
and **does not gate on them**.

## 5. Record grammar v2

`schema_id: palette.acquisition_frame_clock.v2`, `schema_version: 2`. Every
v1 field is kept with the same meaning. Only the camera surface's semantics
and the evidence change.

`clock_surfaces.camera_timestamp_ns` gains:

```json
"semantic_authority": "orange_declared" | "palette_inferred",
"tai_minus_utc_seconds": 37 | null
```

In the declared case: `time_reference_kind: "absolute_epoch"`,
`timescale: "IEEE-1588_PTP_TAI"`, `origin: "1970-01-01T00:00:00_TAI"`
(the v1 spellings, so consumers need no new vocabulary), and
`semantic_status: "producer_declared_by_orange_ptp_time_properties"`.

In the inferred case, the surface is exactly v1's, plus
`semantic_authority: "palette_inferred"` and `tai_minus_utc_seconds`: 37 when
v1 inferred TAI, null otherwise.

`clock_semantic_evidence` gains:

```json
"orange_declaration": {
  "present": true,
  "used": true,
  "reason": null,
  "classification": "ieee1588_tai",
  "timescale": "TAI",
  "origin": "1970-01-01T00:00:00 TAI",
  "semantic_authority": "host_ptp_management_dataset_plus_camera_lock_evidence",
  "rule_id": "orange_camera_clock_classification_v3",
  "result": "direct_ptp_time_properties_pass",
  "expected_tai_minus_utc_seconds": 37,
  "checks": {"...": true},
  "time_properties": {"currentUtcOffset": 37, "currentUtcOffsetValid": true, "ptpTimescale": true},
  "host_evidence_captured_at_utc": "...",
  "pmc_exit_status": 0,
  "recording_started_at_utc": "<recording.started_at_utc>",
  "manifest_sha256": "sha256 of recording_session.json"
},
"palette_cross_check": {"measured_median_delta_ns": 0, "agrees": true}
```

Without a usable declaration: `present`/`used`, plus `reason`
(`"absent"`, `"result:<value>"`, `"check_false:<name>"`, ...).

## 6. Compatibility

- Readers of `clock_surfaces` that only use `unit`, `time_reference_kind`,
  `timescale` and `origin` see the same vocabulary in v2.
- A consumer that needs to know who decided the timescale reads
  `semantic_authority`.
- The 256 published v1 archives are never republished by this change.
  Republishing them as v2 would need its own decision.

## 7. Decisions (Jeremy, 2026-10-12)

- **Q1: refuse.** If Orange declares TAI but Palette's measured
  camera−system delta does not lie within 1 s of the declared offset,
  `AcquisitionFrameClockError` refuses publication. A missing measurement
  (no shared valid timestamps) also refuses.
- **Q2: required.** A declaration counts only with `pmc_exit_status == 0`
  and `host_ptp_management_evidence.available is true`.

## 7a. Implementation notes

- `AcquisitionFrameClockSource.orange_clock_declaration` is read at source
  load and is never used by v1 rules.
- `_build_record(source, schema_version)` chooses the rules.
- `acquisition_frame_clock_source_sha256(source, schema_version=...)` is
  called by the identity authority with the published record's version, so
  v1 archives are rebuilt with v1 rules.
- Publication stays idempotent: if the selected clock is an older version
  and this source still rebuilds it exactly, it remains selected and no v2
  run is written on replay.
- Consumers accept either version: `provider_track_motion_source_handle`
  uses `is_supported_acquisition_frame_clock_schema`, and
  `provider_recording_timing_authority` stamps the record's own `schema_id`.
- The `validated_behavior` export projection contract keeps its v1
  `source_schema_id` label for now. v2 keeps every v1 evidence key, and the
  PTP clock domain the export reads.

## 8. Tests (before code)

- v1 records rebuild byte for byte, from v1 fixtures and from one live
  archive.
- v2 declared, v2 fallback (`fallback_inference_pass`) and v2 absent.
- Each failing `checks` entry.
- Offset mismatch between `currentUtcOffset` and
  `expected_tai_minus_utc_seconds`.
- A cross-check contradiction (Q1).
- Stale `captured_at_utc` is recorded and not gated.
- A reader accepts v1 and v2; an unknown version is refused.
