from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from fisheye.group_analytics_viewer.artifacts import (
    discover_published_image_artifacts,
    has_semantic_montage_artifacts,
    load_published_image_bytes,
)
from fisheye.registry.db import Registry
from fisheye.reporting import (
    check_report_manifest,
    index_report_manifest,
    publish_semantic_montage_report,
    query_indexed_reports,
    report_output_dir,
    resolve_analytics_export_binding,
    verify_report_manifest_sha256,
)


def test_semantic_montage_is_published_indexed_and_discoverable(
    tmp_path: Path,
) -> None:
    analytics_root = tmp_path / "analytics"
    export_manifest = analytics_root / "v1" / "manifests" / "export_run_id=export_001.json"
    export_manifest.parent.mkdir(parents=True)
    export_manifest.write_text(
        json.dumps({"export_run_id": "export_001"}) + "\n",
        encoding="utf-8",
    )
    registry_path = tmp_path / "registry.sqlite"
    registry = Registry(registry_path)
    try:
        registry.upsert_analytics_export(
            export_run_id="export_001",
            export_manifest_path=export_manifest,
            output_root=analytics_root,
            row_counts_by_table={"recording_summary": 2},
            part_files_by_table={"recording_summary": []},
            collection_id="collection_001",
        )
    finally:
        registry.close()

    source_dir = tmp_path / "semantic_montages"
    source_dir.mkdir()
    image_path = source_dir / "stimulus-chaser-distance_trace_montage.png"
    Image.new("RGB", (320, 180), (40, 80, 120)).save(image_path, format="PNG")
    semantic_manifest = source_dir / "semantic_montage_manifest.json"
    semantic_manifest.write_text(
        json.dumps(
            {
                "schema_id": "palette.semantic_visualization_montages.v1",
                "schema_version": 1,
                "source_report_plan_sha256": "plan-sha",
                "visualization_ids": ["stimulus.chaser.distance_trace"],
                "nonready_count": 0,
                "nonready": [],
                "outputs": [
                    {
                        "visualization_id": "stimulus.chaser.distance_trace",
                        "visualization_contract_id": (
                            "palette.stimulus.chaser.distance_trace.v1"
                        ),
                        "path": str(image_path),
                        "tile_count": 2,
                        "width_px": 320,
                        "height_px": 180,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    binding = resolve_analytics_export_binding(registry_path, "export_001")
    output_dir = report_output_dir(binding, "semantic-montages-v1")
    published = publish_semantic_montage_report(
        semantic_manifest_path=semantic_manifest,
        output_dir=output_dir,
        report_id="semantic-montages-v1",
        analytics_export=binding.to_dict(),
    )
    report_manifest = Path(published["manifest_path"])
    report_payload = json.loads(report_manifest.read_text(encoding="utf-8"))
    assert verify_report_manifest_sha256(report_payload)
    assert check_report_manifest(report_manifest, check_files=True)["ok"] is True
    assert report_payload["artifact_count"] == 1
    assert report_payload["artifacts"][0]["artifact_role"] == "cohort_montage"

    registry = Registry(registry_path)
    try:
        assert index_report_manifest(registry, report_manifest) == (
            "export_001",
            "semantic-montages-v1",
        )
    finally:
        registry.close()
    indexed = query_indexed_reports(registry_path, export_run_id="export_001")
    assert [row["report_id"] for row in indexed] == ["semantic-montages-v1"]

    catalog = discover_published_image_artifacts(analytics_root, "export_001")
    assert catalog.diagnostics == ()
    assert has_semantic_montage_artifacts(catalog) is True
    assert len(catalog.artifacts) == 1
    artifact = catalog.artifacts[0]
    assert artifact.visualization_id == "stimulus.chaser.distance_trace"
    assert artifact.width_px == 320
    assert load_published_image_bytes(artifact) == image_path.read_bytes()


def test_published_artifact_discovery_rejects_a_tampered_manifest(
    tmp_path: Path,
) -> None:
    report_dir = (
        tmp_path
        / "v1"
        / "reports"
        / "export_run_id=export_001"
        / "report_id=report_001"
    )
    report_dir.mkdir(parents=True)
    (report_dir / "report_manifest.json").write_text(
        json.dumps(
            {
                "report_id": "report_001",
                "analytics_export": {"export_run_id": "export_001"},
                "manifest_sha256": "not-valid",
                "artifacts": [],
            }
        ),
        encoding="utf-8",
    )

    catalog = discover_published_image_artifacts(tmp_path, "export_001")

    assert catalog.artifacts == ()
    assert [item.code for item in catalog.diagnostics] == [
        "manifest_sha256_mismatch"
    ]
