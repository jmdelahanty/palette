"""Stage one authenticated flat ROI cache, then run one mask shard inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from uuid import uuid4

import zarr

from fisheye.segmentation import infer_unet_subject_masks
from fisheye.shared.flat_roi_cache import (
    cleanup_staged_flat_roi_cache,
    stage_flat_roi_cache_manifest,
)
from fisheye.shared.subject_mask_attempt import (
    validate_subject_mask_attempt,
    validate_subject_mask_scientific_identity,
)
from fisheye.shared.subject_mask_worker_receipt import (
    RAW_SUBJECT_MASK_WORKER_OUTPUT_PATHS,
    validate_subject_mask_worker_semantic_receipt,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)

WORKER_RECEIPT_SCHEMA_ID = "palette.subject_mask_inference.worker_receipt"
WORKER_RECEIPT_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_receipt(path: Path, payload: dict[str, object]) -> None:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--roi-cache-manifest", required=True, type=Path)
    parser.add_argument("--roi-cache-staging-dir", required=True, type=Path)
    parser.add_argument("--worker-receipt-json", required=True, type=Path)
    return parser


def _completed_run_evidence(arguments: Sequence[str]) -> dict[str, object]:
    parsed = infer_unet_subject_masks._build_arg_parser().parse_args(list(arguments))
    if parsed.run_name is None:
        raise ValueError(
            "Staged subject-mask inference requires an explicit --run-name."
        )
    if (
        parsed.output_parent
        != infer_unet_subject_masks.SUBJECT_MASK_SHARD_OUTPUT_PARENT
    ):
        raise ValueError(
            "Staged cache inference must write an immutable "
            "subject_mask_shard_runs child."
        )
    archive = Path(parsed.zarr_path).expanduser().resolve()
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    run_path = f"{parsed.output_parent}/{parsed.run_name}"
    run = root[run_path]
    if run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        raise RuntimeError("Subject-mask inference returned without a complete run.")
    if run.attrs.get("stage_selector_eligible") is not False:
        raise RuntimeError(
            "Subject-mask inference shards must remain selector-ineligible."
        )
    science = run.attrs.get(
        infer_unet_subject_masks.SUBJECT_MASK_SCIENTIFIC_IDENTITY_ATTR
    )
    attempt = run.attrs.get(infer_unet_subject_masks.SUBJECT_MASK_ATTEMPT_ATTR)
    if not isinstance(science, dict) or validate_subject_mask_scientific_identity(
        science
    ):
        raise RuntimeError(
            "Completed subject-mask shard has invalid scientific identity."
        )
    if not isinstance(attempt, dict) or validate_subject_mask_attempt(attempt):
        raise RuntimeError("Completed subject-mask shard has invalid attempt metadata.")
    if attempt["payload"]["run_path"] != run_path:
        raise RuntimeError("Completed subject-mask attempt names a different run path.")
    if attempt["payload"]["scientific_identity_digest"] != science["digest"]:
        raise RuntimeError("Completed subject-mask attempt/science binding differs.")
    binding = run.attrs.get(
        infer_unet_subject_masks.SUBJECT_MASK_WORKER_SEMANTIC_RECEIPT_ATTR
    )
    if not isinstance(binding, dict) or set(binding) != {
        "schema_id",
        "schema_version",
        "payload_digest",
        "relative_path",
        "document_sha256",
        "storage",
    }:
        raise RuntimeError(
            "Completed subject-mask shard lacks its semantic receipt binding."
        )
    relative_path = str(binding.get("relative_path") or "")
    expected_prefix = f"{run_path}/"
    if (
        binding.get("storage") != "strict_json_sidecar_v1"
        or not relative_path.startswith(expected_prefix)
        or Path(relative_path).is_absolute()
        or ".." in Path(relative_path).parts
    ):
        raise RuntimeError("Subject-mask semantic receipt path is unsafe or stale.")
    receipt_bytes = (archive / relative_path).read_bytes()
    if hashlib.sha256(receipt_bytes).hexdigest() != binding.get("document_sha256"):
        raise RuntimeError("Subject-mask semantic receipt document digest differs.")
    try:
        semantic_receipt = json.loads(
            receipt_bytes.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON token {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError("Subject-mask semantic receipt is not strict JSON.") from exc
    required_paths = list(RAW_SUBJECT_MASK_WORKER_OUTPUT_PATHS)
    if bool(parsed.write_masks_roi):
        required_paths.insert(1, "masks_roi")
    validated_receipt = validate_subject_mask_worker_semantic_receipt(
        semantic_receipt,
        scientific_identity=science,
        attempt=attempt,
        required_paths=required_paths,
    )
    if (
        validated_receipt["payload_digest"] != binding.get("payload_digest")
        or validated_receipt["payload"]["run_path"] != run_path
    ):
        raise RuntimeError("Subject-mask semantic receipt binding differs.")
    return {
        "archive_path": str(archive),
        "run_path": run_path,
        "completion_status": RUN_STATUS_COMPLETE,
        "stage_selector_eligible": False,
        "attempt_id": attempt["payload"]["attempt_id"],
        "attempt_payload_digest": attempt["payload_digest"],
        "scientific_identity_digest": science["digest"],
        "source_roi_pixels_sha256": run.attrs.get("source_roi_pixels_sha256"),
        "model_artifact_sha256": science["payload"]["model"].get("artifact_sha256"),
        "semantic_receipt_payload_digest": validated_receipt["payload_digest"],
        "semantic_receipt_document_sha256": binding["document_sha256"],
        "semantic_receipt_relative_path": relative_path,
    }


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(argv) if argv is not None else None
    args, forwarded = build_parser().parse_known_args(arguments)
    inference_args = infer_unet_subject_masks._build_arg_parser().parse_args(forwarded)
    if inference_args.run_name is None:
        raise ValueError(
            "Staged subject-mask inference requires an explicit --run-name."
        )
    if (
        inference_args.output_parent
        != infer_unet_subject_masks.SUBJECT_MASK_SHARD_OUTPUT_PARENT
    ):
        raise ValueError(
            "Staged cache inference requires --output-parent "
            "subject_mask_shard_runs."
        )
    attempt_id = inference_args.attempt_id or str(uuid4())
    effective_forwarded = list(forwarded)
    if inference_args.attempt_id is None:
        effective_forwarded.extend(["--attempt-id", attempt_id])
    started_at = _utc_now()
    staged_manifest: Path | None = None
    receipt: dict[str, object] = {
        "schema_id": WORKER_RECEIPT_SCHEMA_ID,
        "schema_version": WORKER_RECEIPT_SCHEMA_VERSION,
        "status": "running",
        "started_at_utc": started_at,
        "source_roi_cache_manifest": str(
            args.roi_cache_manifest.expanduser().resolve()
        ),
        "staging_dir": str(args.roi_cache_staging_dir.expanduser().resolve()),
        "forwarded_arguments": list(effective_forwarded),
        "attempt_id": attempt_id,
        "lsb_jobid": os.environ.get("LSB_JOBID"),
        "lsb_jobindex": os.environ.get("LSB_JOBINDEX"),
    }
    _write_receipt(args.worker_receipt_json, receipt)
    try:
        staged_manifest, staging = stage_flat_roi_cache_manifest(
            args.roi_cache_manifest,
            staging_dir=args.roi_cache_staging_dir,
        )
        receipt["roi_cache_staging"] = staging
        infer_unet_subject_masks.main(
            [*effective_forwarded, "--roi-cache-manifest", str(staged_manifest)]
        )
        run_evidence = _completed_run_evidence(
            [*effective_forwarded, "--roi-cache-manifest", str(staged_manifest)]
        )
        if run_evidence["attempt_id"] != attempt_id:
            raise RuntimeError(
                "Persisted subject-mask attempt differs from worker attempt."
            )
        if run_evidence["source_roi_pixels_sha256"] != staging["copy"]["source_sha256"]:
            raise RuntimeError(
                "Persisted subject-mask pixel identity differs from staged cache."
            )
        receipt.update(
            {
                "status": "complete",
                "finished_at_utc": _utc_now(),
                "run": run_evidence,
            }
        )
        _write_receipt(args.worker_receipt_json, receipt)
    except BaseException as exc:
        receipt.update(
            {
                "status": "failed",
                "finished_at_utc": _utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        _write_receipt(args.worker_receipt_json, receipt)
        raise
    finally:
        if staged_manifest is not None:
            cleanup_staged_flat_roi_cache(staged_manifest)


if __name__ == "__main__":
    main()


__all__ = [
    "WORKER_RECEIPT_SCHEMA_ID",
    "WORKER_RECEIPT_SCHEMA_VERSION",
    "build_parser",
    "main",
]
