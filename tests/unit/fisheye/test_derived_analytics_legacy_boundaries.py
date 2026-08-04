from __future__ import annotations

import json
from pathlib import Path

import pytest

from fisheye.baseline_strategy.contracts import (
    BASELINE_STRATEGY_FEATURES_TABLE,
    LEGACY_SCHEMA_VERSION as BASELINE_LEGACY_VERSION,
    SCHEMA_ID as BASELINE_SCHEMA_ID,
)
from fisheye.baseline_strategy.query import strategy_table_parts
from fisheye.training_response.contracts import (
    LEGACY_SCHEMA_VERSION as TRAINING_LEGACY_VERSION,
    SCHEMA_ID as TRAINING_SCHEMA_ID,
    TRAINING_RESPONSE_FEATURES_TABLE,
)
from fisheye.training_response.query import training_response_table_parts


@pytest.mark.parametrize(
    ("schema_id", "schema_version", "table_name", "resolver"),
    [
        (
            BASELINE_SCHEMA_ID,
            BASELINE_LEGACY_VERSION,
            BASELINE_STRATEGY_FEATURES_TABLE,
            strategy_table_parts,
        ),
        (
            TRAINING_SCHEMA_ID,
            TRAINING_LEGACY_VERSION,
            TRAINING_RESPONSE_FEATURES_TABLE,
            training_response_table_parts,
        ),
    ],
)
def test_v1_layout_requires_explicit_compatibility_policy(
    tmp_path: Path,
    schema_id: str,
    schema_version: int,
    table_name: str,
    resolver,
) -> None:
    run_id = "legacy-run"
    part = (
        tmp_path
        / "v1"
        / table_name
        / f"analysis_run_id={run_id}"
        / "part-00000.parquet"
    )
    part.parent.mkdir(parents=True)
    part.write_bytes(b"legacy fixture bytes")
    manifest = tmp_path / "v1" / "manifests" / f"analysis_run_id={run_id}.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_id": schema_id,
                "schema_version": schema_version,
                "analysis_run_id": run_id,
                "part_files_by_table": {table_name: [str(part)]},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        resolver(tmp_path, run_id, table_name)

    assert resolver(
        tmp_path,
        run_id,
        table_name,
        allow_legacy_layout=True,
    ) == (part.resolve(),)
