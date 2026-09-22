# Validated behavior frame-clock export

The installed profile
`validated_core_behavior_bout_kinematics_frame_clock_v1` extends
`validated_core_behavior_bout_kinematics_v1` without changing the older
profile. It contains all eleven existing behavior and bout tables plus two
clock tables.

## Tables

`recording_clock_metadata` contains one row per admitted recording. It records:

- the Orange `session_id`, session start declaration, and camera ID;
- the raw recording, `recording_frame_index.parquet`, recording manifest, and
  PTP summary paths where available;
- SHA-256 digests for those raw metadata sources and the bound acquisition
  authority;
- the camera and host clock domains, origins, timescales, and semantic status;
- explicit within-session and cross-session alignment status; and
- `equal_frame_rate_alignment_valid=false`.

`acquisition_frame_clock_samples` contains one row for every source camera
frame. It records the recording-local frame key, Orange recording frame ID,
camera timestamp, host timestamp, and a validity flag for each timestamp.
Timestamp values are meaningful only when their corresponding validity flag is
true.

## Joins

Join any framewise behavior table to `acquisition_frame_clock_samples` with:

```text
(recording_id, source_acquisition_frame_index)
```

For two cameras in the same synchronized recording session, first require equal
`session_id` values and compatible camera clock metadata, then match valid
`camera_timestamp_ns` values with a tolerance appropriate to the acquisition
rate. Do not align cameras by `time_seconds`, frame number, or equal frame rate.

Recordings from different sessions remain separate. Cross-session alignment
requires a separately validated traceable absolute-clock claim or an explicit
external anchor mapping. The profile carries the data needed to make a future
claim, but does not manufacture that claim from matching frame rates.

`system_timestamp_ns` is the host `CLOCK_REALTIME` observation captured after
frame retrieval. It is useful for diagnostics and wall-clock context. The
camera/PTP surface is the primary within-session camera-alignment coordinate
when `within_session_alignment_status` says that synchronization evidence is
available.

## Source integrity

The bundle builder reopens the complete raw clock source and refuses it when:

- its row count or complete zero-based parent-frame domain differs from the
  admitted source video;
- its camera or recording identity differs from the acquisition authority;
- the raw frame-index SHA-256 differs from the digest sealed by
  `source_video_metadata`; or
- its recording manifest session identity conflicts with the analysis Zarr.

The export remains selector-ineligible under the existing validated-behavior
publication contract. Its manifest, shard receipts, Parquet file hashes, source
bindings, and software commit provide the immutable handoff identity.
