from __future__ import annotations

import json
from pathlib import Path

from fisheye.diagnostics.publish_coordinate_catalog_canary import (
    publish_coordinate_catalog_canary,
)
from fisheye.shared.zarr.canonical_detection_manifest import (
    CANONICAL_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
)
from fisheye.shared.zarr.crop_manifest import (
    CROP_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
)
from fisheye.shared.zarr.refined_detection_manifest import (
    REFINED_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
)
from tests.unit.fisheye.test_detection_snapshot_publication import (
    RECORDING_IDENTITY,
    _build_sources,
)


def test_canary_publishes_three_coordinate_aware_selector_ineligible_artifacts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    _build_sources(source)
    video = tmp_path / "camera.mp4"
    video.write_bytes(b"deterministic-test-video")
    recording_manifest = tmp_path / "recording_manifest.json"
    recording_manifest.write_text(
        """{
          "camera_id":"test_camera",
          "session_uuid":"snapshot_publication_recording",
          "files":{"cams":["camera.mp4"]}
        }""",
        encoding="utf-8",
    )
    destination = tmp_path / ".palette_benchmarks" / "coordinate_canary_v1"

    result = publish_coordinate_catalog_canary(
        source_analysis_zarr=source,
        recording_identity=RECORDING_IDENTITY,
        camera_identity="test_camera",
        video_reference=video,
        recording_manifest=recording_manifest,
        legacy_detect_run="detect_source",
        legacy_refined_run="refined_source",
        destination=destination,
        work_root=tmp_path,
        canonical_run_id="detect_coordinate_v3",
        refined_run_id="refined_coordinate_v2",
        crop_run_id="crop_coordinate_v2",
        crop_size=32,
        crimson_commit="a" * 40,
        crimson_review_path="docs/review.md",
        crimson_review_sha256="b" * 64,
    )

    assert result["status"] == "complete"
    assert destination.is_dir()
    handoff = json.loads(
        (destination / "handoff_manifest.json").read_text(encoding="utf-8")
    )
    payload = handoff["payload"]
    assert payload["selector_eligible"] is False
    assert payload["production_state_changes"] == []
    assert payload["artifacts"]["canonical"]["manifest_schema_version"] == (
        CANONICAL_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
    )
    assert payload["artifacts"]["refined"]["manifest_schema_version"] == (
        REFINED_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
    )
    assert payload["artifacts"]["crop"]["manifest_schema_version"] == (
        CROP_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
    )
    assert (
        payload["coordinate_samples"]["normalized_detection"]["maximum_absolute_error"]
        == 0.0
    )
    assert (
        payload["coordinate_samples"]["rowwise_roi_to_source"]["maximum_absolute_error"]
        == 0.0
    )
    assert payload["validation"]["copied_artifact_tree_equality"] is True
    assert payload["validation"]["source_metadata_unchanged"] is True
    assert payload["validation"]["source_video_stat_unchanged"] is True
    assert list(tmp_path.glob("coordinate-catalog-canary-*")) == []
