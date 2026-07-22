from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from fisheye.cohorts.registry import compute_manifest_sha256
from fisheye.cohorts.spec import canonical_sha256


def test_cohort_report_submitter_uses_frozen_members_and_dependency(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[3]
    registry = tmp_path / "registry.sqlite"
    registry.touch()
    cohort_query = {
        "schema_id": "palette.cohort_query",
        "schema_version": 1,
        "cohort_id": "report_cohort",
    }
    cohort = {
        "schema_id": "palette.frozen_cohort_manifest",
        "schema_version": 1,
        "manifest_canonicalization": "json_sorted_keys_no_manifest_sha256_v1",
        "created_utc": "2026-07-18T00:00:00Z",
        "cohort_id": "report_cohort",
        "cohort_name": "Report cohort",
        "cohort_query": cohort_query,
        "cohort_query_sha256": canonical_sha256(cohort_query),
        "registry": {
            "query_snapshot_sha256": "a" * 64,
            "access_mode": "read_only",
        },
        "selection_policy": {"include_every_match": True, "limit": None},
        "member_count": 2,
        "members": [
            {
                "dataset_id": "dataset_a",
                "recording_id": "recording_a",
                "zarr_path": "/recording_a.zarr",
                "zarr_origin": "source",
                "zarr_use": "analysis",
                "dataset_status": "active",
            },
            {
                "dataset_id": "dataset_b",
                "recording_id": "recording_b",
                "zarr_path": "/recording_b.zarr",
                "zarr_origin": "source",
                "zarr_use": "analysis",
                "dataset_status": "active",
            },
        ],
        "selection_summary": {"included_count": 2, "blocked_count": 0},
    }
    cohort["manifest_sha256"] = compute_manifest_sha256(cohort)
    cohort_path = tmp_path / "frozen_cohort.json"
    cohort_path.write_text(json.dumps(cohort), encoding="utf-8")
    output_root = tmp_path / "analytics"
    log_dir = tmp_path / "logs"

    result = subprocess.run(
        [
            "bash",
            str(repo / "scripts" / "submit_cohort_report_bsub.sh"),
            "--cohort-manifest",
            str(cohort_path),
            "--export-run-id",
            "report_export_v1",
            "--report-id",
            "recording_montages_v1",
            "--registry",
            str(registry),
            "--output-root",
            str(output_root),
            "--palette-repo",
            str(repo),
            "--log-dir",
            str(log_dir),
            "--dependency-done",
            "123456",
        ],
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "PALETTE_PYTHON": sys.executable},
    )

    bsub_line = next(
        line for line in result.stdout.splitlines() if line.startswith("bsub_command=")
    )
    assert "-w done\\(123456\\)" in bsub_line
    run_dir = log_dir / "cohort_report_report_export_v1_recording_montages_v1"
    job = (run_dir / "run_cohort_report.sh").read_text(encoding="utf-8")
    assert "Frozen dataset binding changed" in job
    assert 'recording_args+=(--recording-id "${recording_id}")' in job
    assert "fisheye.reporting montage" in job
    assert "fisheye.reporting publish-montage-report" in job
    assert "fisheye.reporting check-report" in job
    assert "--index-registry" in job
