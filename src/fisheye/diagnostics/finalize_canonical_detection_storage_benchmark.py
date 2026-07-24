#!/usr/bin/env python3
"""Validate and aggregate completed canonical detection benchmark blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from fisheye.cluster.lsf.bundle import write_json_snapshot
from fisheye.shared.zarr.benchmark_matrix import (
    require_storage_benchmark_matrix_manifest,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def finalize_benchmark(
    *,
    matrix_path: Path,
    workflow_root: Path,
    output: Path,
) -> dict[str, object]:
    matrix = _read_json(matrix_path.expanduser().resolve())
    require_storage_benchmark_matrix_manifest(matrix)
    root = workflow_root.expanduser().resolve()
    output_path = output.expanduser().resolve()
    if not output_path.is_relative_to(root):
        raise ValueError("Aggregate output must be below the workflow root.")
    if output_path.exists():
        raise FileExistsError(f"Aggregate output already exists: {output_path}")

    candidates = {
        str(candidate["candidate_id"]): candidate
        for candidate in matrix.get("candidates", [])
    }
    block_summaries: list[dict[str, object]] = []
    all_published: list[str] = []
    for repetition in matrix.get("repetitions", []):
        scale_id = str(repetition["scale_id"])
        repetition_index = int(repetition["repetition_index"])
        block_path = (
            root
            / "reports"
            / "blocks"
            / f"{scale_id}_repetition_{repetition_index:03d}.json"
        )
        block = _read_json(block_path)
        if block.get("status") != "complete" or block.get("fixture_unchanged") is not True:
            raise RuntimeError(f"Benchmark block is not complete and exact: {block_path}")
        expected_ids = [str(trial["candidate_id"]) for trial in repetition["trials"]]
        actual_records = block.get("candidates")
        if not isinstance(actual_records, list):
            raise ValueError(f"Benchmark block lacks candidate records: {block_path}")
        actual_ids = [str(record["candidate_id"]) for record in actual_records]
        if actual_ids != expected_ids:
            raise RuntimeError(f"Benchmark block candidate order mismatch: {block_path}")
        for record in actual_records:
            candidate_id = str(record["candidate_id"])
            if candidate_id not in candidates:
                raise RuntimeError(f"Unknown candidate in block: {candidate_id}")
            if record.get("physical_fingerprint") != candidates[candidate_id].get(
                "physical_fingerprint"
            ):
                raise RuntimeError(f"Candidate fingerprint mismatch: {candidate_id}")
            published = Path(str(record["published_candidate"])).resolve()
            if not published.is_relative_to(root) or not (published / "zarr.json").is_file():
                raise RuntimeError(f"Published candidate is missing or unsafe: {published}")
            for field in (
                "local_write_report",
                "publication_report",
                "prfs_read_report",
            ):
                evidence = Path(str(record[field])).resolve()
                if not evidence.is_relative_to(root) or not evidence.is_file():
                    raise RuntimeError(f"Candidate evidence is missing: {evidence}")
            if record.get("prfs_reads", {}).get("all_exact") is not True:
                raise RuntimeError(f"PRFS read validation failed: {candidate_id}")
            all_published.append(str(published))
        block_summaries.append(
            {
                "scale_id": scale_id,
                "repetition_index": repetition_index,
                "block_report": str(block_path),
                "candidate_count": len(actual_records),
                "total_seconds": block.get("total_seconds"),
            }
        )

    aggregate = {
        "schema_id": "palette.canonical_detection_storage_benchmark_aggregate",
        "schema_version": 1,
        "status": "complete",
        "matrix": str(matrix_path.expanduser().resolve()),
        "matrix_fingerprint": matrix.get("matrix_fingerprint"),
        "workflow_root": str(root),
        "blocks": block_summaries,
        "published_candidates": all_published,
        "summary": {
            "block_count": len(block_summaries),
            "published_candidate_count": len(all_published),
            "registry_updates": 0,
            "selector_updates": 0,
            "training_artifacts": 0,
            "profile_promoted": False,
        },
    }
    write_json_snapshot(output_path, aggregate)
    return aggregate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--workflow-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = finalize_benchmark(
        matrix_path=args.matrix,
        workflow_root=args.workflow_root,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
