# Native unified experimental H5 candidates

The public stimulus importer supports the explicit
`unified_experimental_h5_v1` input profile. This is a native, immutable,
selector-ineligible candidate import, **not** v5/v6 coordinate normalization,
production analysis adoption, physical registration, or scientific acceptance.
The ordinary v5/v6 importer remains unchanged.

```bash
scripts/py -m fisheye.analysis.import_stimulus_to_zarr \
  /path/to/finalized-synthetic-recording.h5 /path/to/candidate.zarr \
  --source-profile unified_experimental_h5_v1 \
  --finalization-receipt /path/to/finalized-receipt.json \
  --run-name unified_native_candidate
```

Both the explicit H5 and its external receipt are required. A recognized native
H5 without explicit profile selection is rejected before destination mutation;
there is no metadata-only fallback, inferred companion, or gap repair. Existing
run names, including failed tombstones, cannot be overwritten. A retry uses a
fresh name. The importer does not change `latest`, `latest_complete`,
`authoritative_run`, registry status, or any production authority.

## What is validated

Admission checks both independent integrity domains: the exact open H5's
post-close file bytes against its external receipt, and its closed internal
dependency manifest against native logical types, shapes, references and payload
digests. It also verifies component accounting, exact uint64 row/frame identities,
recording/camera correspondence, authored and executed protocol identity, static
geometry/presentation consistency, source namespace and submitted identity
agreement, and optional appearance applicability/replay.

Renderer-only frames may remain unmapped. Selected correspondence components
must cover all of their canonical rows exactly once. A Chaser table is not
invented for a frame-only, grid, or grating component. Current acquisition and
held-target acquisition identities remain distinct. Appearance ownership is
step-local and uses half-open stimulus-frame intervals; a profile may be reused
in a later step by a different Chaser. Nominal renderer code luminance is not
measured projector luminance.

Source evidence is preserved, not upgraded: for example, recorded geometry
readiness and acceptance claims remain source claims, not independent Palette
approval. There is no new manual-review requirement or scientific acceptance
receipt. Technical completion, contract validity, use-scoped acceptance, and
activation remain separate, consistent with the
[acceptance checklist](diagnostics/authority_acceptance_implementation_checklist_2026-08-27.md).

## Native storage and reading

`analysis/stimulus_runs/<run>/native_h5` preserves the original H5 paths. Each
source dataset is represented by a group containing a one-dimensional `uint8`
`payload`. Fixed-size values use exact packed little-endian bytes; variable
strings use unsigned-64-bit little-endian lengths followed by their exact bytes.
The versioned `palette.unified_h5_native_storage` manifest binds every path,
type, shape, payload hash, typed attribute value, original finalization receipt,
admission summary, run identity, owner, and Palette producing-code provenance.
This storage digest is **not** the H5 container-byte digest or an old Palette
coordinate-product digest. Writer input provenance uses the normalized source
path already verified against the exact open H5 handle.

The existing Palette physical planner and array factory choose regular local
chunks (`scratch_compute_v1`, 1 MiB target). One importer owns all physical
writes; no Dask workers share chunks. Source admission retains the producer's
64 MiB dataset, 2,000,000-element and 8 MiB text budgets. Native inventory is
additionally limited to 4 MiB and its final manifest to 8 MiB; variable-attribute
conversion buffers and per-node attribute values are bounded to 8 MiB.

```python
import zarr
from fisheye.shared.unified_h5.storage import load_unified_stimulus_candidate

root = zarr.open_group("/path/to/candidate.zarr", mode="r", use_consolidated=True)
candidate = load_unified_stimulus_candidate(root, run_name="unified_native_candidate")
frames = candidate.read_table("/frames/stimulus", start=0, stop=4)
execution = candidate.read_json("/protocol/executed/execution_index_json")
attributes = candidate.typed_attributes("/metadata/session")
```

The unpatched reader verifies the copied payloads, manifest, producing-code
provenance, completion/ineligibility, and direct-versus-consolidated metadata
before returning the candidate. Table reads retain the original structured
dtype, including wide strings, nested fields, signed timestamps and full-width
uint64 keys. Other native arrays use `read_dataset`; `typed_attributes` exposes
exact attribute type/shape/value bytes without coercing floats or integer IDs.
Payload reads recheck their hash and candidate generation. Validation is bounded
but deliberately rehashes the complete requested dataset; it is not a receipt
cache or a production-read throughput claim.

Candidates must remain immutable. The native manifest attests to the validated
source-to-copy operation; reading a copy does not reconstruct the original H5
container or independently authenticate an external issuer. A reader rejects
missing/stale consolidation, changed copied payloads or metadata, changed owner,
incomplete runs, and conflicting provenance. Failures during import retain only
an owned ineligible tombstone; loss of ownership stops writes and cleanup.

Physical-coordinate normalization, downstream analysis consumers, parent-transfer
dispatch, real-data canaries, activation, and old-companion retirement are later,
separately scoped work. See the [implementation handoff](diagnostics/unified_h5_import_handoff_2026-09-12.md)
for exact commits, tests, CI, and integration status.
