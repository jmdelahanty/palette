from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_ENTRYPOINTS = (
    "src/fisheye/utils/run_detect_with_registry_model.py",
    "src/fisheye/utils/run_detections_batch.py",
    "src/fisheye/utils/run_recording_analysis_pipeline.py",
    "src/fisheye/inference/predict_detections.py",
    "src/fisheye/utils/run_detection_local_publish.py",
    "src/fisheye/utils/run_detection_artifact.py",
)


def _imports_low_level_detector(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        if module.endswith("detection.detect_yolo") and any(
            alias.name == "detect_yolo" for alias in node.names
        ):
            return True
    return False


def test_supported_detection_entrypoints_do_not_import_low_level_writer() -> None:
    offenders = [
        relative
        for relative in CANONICAL_ENTRYPOINTS
        if _imports_low_level_detector(REPO_ROOT / relative)
    ]
    assert offenders == []


def test_only_shared_candidate_boundary_imports_low_level_writer() -> None:
    candidate = REPO_ROOT / "src/fisheye/shared/detection_candidate.py"
    assert _imports_low_level_detector(candidate)


def test_atomic_publisher_does_not_depend_on_registry_command_module() -> None:
    publisher = (
        REPO_ROOT / "src/fisheye/utils/run_detection_local_publish.py"
    ).read_text(encoding="utf-8")
    assert "run_detect_with_registry_model import" not in publisher


def test_latest_based_direct_submitter_is_retired() -> None:
    assert not (REPO_ROOT / "scripts/submit_detect_quality_refine_bsub.sh").exists()
