# In-memory clock audit probes — 2026-09-06

Companion to [the ingestion audit](../pipeline_evidence_ingestion_audit_2026-09-06.md),
findings F8/F9. Both the parallel reviewer and root reproduced these findings at
`0a9531f9914f30a487704388a12a088a8d3365fe`. The command below is root's exact
independent probe. It is diagnostic evidence, not a passing rejection test or a
new runtime validator. No installation or source/data modification is involved.

Run from that Palette worktree with `scripts/py`. `AGENTS.md` is used only as an
existing file for a stat/relative-path stand-in; it is not parsed as clock evidence.
Summary content and Parquet reads are replaced in memory. The source loader and
semantic/array validators themselves are unpatched. This does not read production
arrays, require real Zarr I/O, or publish a clock.

```bash
scripts/py - <<'PY'
from pathlib import Path
from unittest.mock import patch
import json
import numpy as np
import pyarrow as pa
from fisheye.shared import acquisition_frame_clock as clock

root = Path.cwd()
source_path = root / 'AGENTS.md'
base = 1_700_000_000_000_000_000
system = np.asarray([base, base + 10, base + 20], dtype=np.int64)
camera = system + 37_000_000_000
valid = np.ones(3, dtype=np.bool_)
for status in ('slave', 'unlocked', 'unsynchronized', 'not synchronized', 'faulty', 'listening'):
    summary = {'sync': {'mode': 'ptp', 'camera_sync_enabled': True}, 'cameras': {'2010095': {'sync_camera_enabled': True, 'ptp_register_reads': 3, 'ptp_offset_ns': {'samples': 3, 'minimum': 0, 'maximum': 0}, 'latch_minus_frame_ns': {'samples': 3, 'minimum': 0, 'maximum': 0}, 'ptp_status': status}}}
    with patch.object(clock, '_clock_summary_path', return_value=source_path), patch.object(clock, '_safe_json_object', return_value=summary):
        surfaces, evidence = clock._clock_semantics(root, camera_id='2010095', camera=camera, camera_valid=valid, system=system, system_valid=valid)
    print(json.dumps({'probe': 'ptp_status', 'status': status, 'time_reference_kind': surfaces['camera_timestamp_ns']['time_reference_kind'], 'noncontradictory': evidence['explicit_ptp_status_does_not_contradict_synchronization']}))

for name, ids, parents, timestamps in (
    ('valid', [1, 2, 3], [0, 1, 2], [100, 110, 120]),
    ('fractional', [1.9, 2.9, 3.9], [0.9, 1.9, 2.9], [100.9, 110.9, 120.9]),
    ('negative_source_ids', [-3, -2, -1], [0, 1, 2], [100, 110, 120]),
    ('shifted_source_ids', [400, 401, 402], [0, 1, 2], [100, 110, 120]),
    ('overflow_reversal', [1, 2, 3], [0, 1, 2], [np.iinfo(np.int64).max, np.iinfo(np.int64).min, np.iinfo(np.int64).min + 1]),
):
    table = pa.table({'camera_serial': ['2010095'] * 3, 'recording_frame_id': ids, 'parent_frame_index': parents, 'timestamp': timestamps, 'timestamp_sys': [100, 110, 120]})
    with patch.object(clock.pq, 'read_table', return_value=table):
        loaded = clock._load_parquet_source(source_path, recording_dir=root, camera_id='2010095')
    clock._validate_source(loaded, expected_frame_count=3)
    print(json.dumps({'probe': 'source_admission', 'case': name, 'accepted': True, 'recording_frame_id': loaded.recording_frame_id.tolist(), 'parent_frame_index': loaded.parent_frame_index.tolist(), 'camera_timestamp_ns': loaded.camera_timestamp_ns.tolist()}))
PY
```

Observed at the audited commit:

- `slave`, `unlocked`, `unsynchronized`, and `not synchronized` all report
  `absolute_epoch` with `noncontradictory=true`.
- `faulty` and `listening` report `device_defined_unknown_epoch` with
  `noncontradictory=false`.
- All five source cases are accepted. Fractional values become integers; negative
  and shifted recording IDs remain as supplied; the overflow reversal remains in
  the accepted camera timestamp vector.

The 37-second camera/system difference is fixture input satisfying the classifier's
current hard-coded criterion, not independent evidence of an actual TAI/UTC offset
for any recording. Qualifying summary fields in this probe are synthetic.

For regression coverage, convert these diagnostics into explicit expected-refusal
tests only as part of an authorized correction, preserving valid controls and
documenting the supported frame-ID origin and PTP-state grammar. Do not alter
historical receipts or reclassify recording clocks through this diagnostic.
