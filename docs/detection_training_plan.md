# Detection Training Data Pipeline Plan

## Goals
- Preserve capture-time provenance without blocking acquisition.
- Enable reproducible dataset curation and model training.
- Keep audit trails for dish/cross lineage, camera setup, and protocol context.

## Capture-Time Provenance (H5)
- The rig writes a zebrobot snapshot into the H5 at acquisition time.
- Never block acquisition: write a partial snapshot if API calls fail.
- Avoid PII fields; keep only non-identifying metadata.
 - Write session context fields (rig/arena/canvas/camera) into H5 root attrs.
   - See `docs/session_context.md` for required/recommended fields.

Snapshot contents (no PII):
- Dish: dish_id, cross_id, genotype, dof, fish_count, species, sex
- Cross: cross_id, line_strain, parents (identifier + sex)

H5 layout:
- Root attrs: dish_id, cross_id, zebrobot_snapshot_utc, zebrobot_schema_version=1
- /zebrobot_snapshot/snapshot_json (UTF-8 JSON)

Partial snapshot example:
- status="partial"
- missing=["cross"] or ["dish"]
- cross=null if unavailable

## Zarr Metadata Mirror
- On H5 import or post-processing, mirror the snapshot into:
  - analysis_metadata.zebrobot_snapshot
 - Mirror session context into:
   - analysis_metadata.session_context
- H5 remains source of truth; Zarr mirror is for downstream tools.

## Dataset Registry (SQLite)
Purpose: catalog datasets, provenance, and training runs.

Minimal tables:
- datasets: dataset_id, zarr_path, created_utc, last_seen_utc, status, hash
- provenance: dataset_id, dish_id, cross_id, line_strain, genotype, parents, species, sex
  - plus: rig_id, arena_id, camera_id, canvas_name
- detection_sources: dataset_id, refined_run, source_type, counts, created_utc
- training_sets: set_id, name, query_filter, dataset_ids, created_utc
- training_runs: run_id, set_id, config_path, manifest_path, model_path, metrics_path, created_utc

Registry location:
- Default: runs/registry/palette_registry.sqlite
- Override via PALETTE_REGISTRY_PATH or --registry on commands
- Optional config: configs/fisheye/registry.yaml (registry_path)

Setting the env var (examples):
- One-off command:
  - PALETTE_REGISTRY_PATH=/nvme1/palette_registry.sqlite python -m fisheye.registry.status
- Current shell session:
  - export PALETTE_REGISTRY_PATH=/nvme1/palette_registry.sqlite
- Persist for bash:
  - echo 'export PALETTE_REGISTRY_PATH=/nvme1/palette_registry.sqlite' >> ~/.bashrc
  - source ~/.bashrc

CLI:
- Scan datasets: python -m fisheye.registry.scan /path/to/zarr_root --recursive
- Coverage report: python -m fisheye.registry.status --list-issues

Registry behavior:
- Missing/moved Zarrs are marked status=missing, not deleted.
- Periodic scan refreshes hashes and last_seen_utc.

## Data Curation + Manifest Generation
- Preflight tool validates each Zarr:
  - images_ds exists
  - refined detections present
  - bbox sanity checks
  - zebrobot snapshot presence (strict or warn)
- Output:
  - training manifest (dataset list + provenance fields)
  - YOLO config (train.yaml) pointing to curated datasets
- Dry-run prints what will be generated without writing files.

Registry integration:
- prepare_detect_training: pass --register to upsert datasets and provenance.

## Model Training
- Training reads the generated config + manifest.
- Run outputs (weights, metrics, config, manifest) are registered in SQLite.
- Training reproducibility: config + manifest are the source of truth.
 - Optional exports:
   - `--export-onnx` writes `<run>/exports/onnx/<run_id>.onnx`
   - `--export-trt` writes `<run>/exports/tensorrt/<run_id>_<precision>.engine`
   - TensorRT engine manifest: `<run>/exports/tensorrt/<run_id>_<precision>.manifest.json`

Registry integration:
- train_detection: pass --log-registry [--manifest <manifest.json>].

Standalone export (existing run):
- python -m fisheye.training.export_detection /path/to/run --export-trt --trtexec /usr/local/TensorRT-10.0.1.6/bin/trtexec
- Reuse an existing ONNX: add --onnx-path /path/to/model.onnx (skips ONNX export)

## Maintenance and Auditing
- Rehydrate partial snapshots when zebrobot is available.
- Track provenance drift if metadata fields change.
- Keep audit notes in registry and training manifests.

## Plan of Record
1. Rig-side zebrobot snapshot writer (H5).
2. Zarr mirror of snapshot during import.
3. Dataset registry schema + scanner.
4. Curation/preflight tool + manifest generator.
5. Training integration with registry logging.
6. Audit utilities (coverage, provenance completeness).

## Open Questions
- API base URL and access pattern on rigs.
- Where the registry DB should live (per-rig vs shared).
- How to name datasets (dataset_id scheme).
