from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from fisheye.utils import materialize_detection_subject_position as module


COMMIT = "a" * 40
MANIFEST = "b" * 64
RUN_NAME = "position_detection_v1"
RUN_PATH = f"analysis/subject_position_runs/observation/{RUN_NAME}"


def _plan() -> SimpleNamespace:
    return SimpleNamespace(
        final_manifest_sha256=MANIFEST,
        run_path=RUN_PATH,
        publication_attempt_uuid="00000000-0000-0000-0000-000000000001",
        as_dict=lambda: {"run_path": RUN_PATH},
    )


def _execute(tmp_path: Path, **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "source_run_path": "detect_runs/canonical_v3",
        "output_run_name": RUN_NAME,
        "scratch_root": tmp_path / "scratch",
        "publication_attempt_uuid": "00000000-0000-0000-0000-000000000001",
        "palette_commit": COMMIT,
        "workflow_id": "cohort_v1",
        "expected_manifest_sha256": MANIFEST,
        "apply": False,
    }
    values.update(overrides)
    return module.execute(tmp_path / "archive.zarr", **values)  # type: ignore[arg-type]


def test_plan_is_read_only_and_binds_expected_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "build_plan", lambda *args, **kwargs: _plan())

    result = _execute(tmp_path)

    assert result["status"] == "planned_no_writes"
    assert result["manifest_sha256"] == MANIFEST
    assert result["selector_eligible"] is False
    assert result["writes"] is False


def test_plan_rejects_manifest_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "build_plan", lambda *args, **kwargs: _plan())

    with pytest.raises(ValueError, match="differs from expectation"):
        _execute(tmp_path, expected_manifest_sha256="c" * 64)


def test_apply_publishes_and_reopens_exact_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "build_plan", lambda *args, **kwargs: _plan())
    monkeypatch.setattr(
        module,
        "publish_subject_position_run",
        lambda plan, keep_scratch: {"status": "published"},
    )
    monkeypatch.setattr(
        module,
        "load_subject_position_source_handle",
        lambda *args, **kwargs: SimpleNamespace(manifest_sha256=MANIFEST),
    )

    result = _execute(tmp_path, apply=True)

    assert result["status"] == "published_selector_ineligible"
    assert result["manifest_sha256"] == MANIFEST
    assert result["writes"] is True


def test_existing_output_requires_and_validates_exact_manifest(tmp_path, monkeypatch):
    target = tmp_path / "archive.zarr" / RUN_PATH
    target.mkdir(parents=True)
    calls: list[dict[str, object]] = []

    def load(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            manifest_sha256=MANIFEST,
            estimator_record={"estimator_id": "detection_bbox_centroid.v1"},
        )

    monkeypatch.setattr(module, "load_subject_position_source_handle", load)

    result = _execute(tmp_path, apply=True)

    assert result["status"] == "reused_exact"
    assert calls[0]["expected_manifest_sha256"] == MANIFEST
    with pytest.raises(ValueError, match="only with --expected-manifest"):
        _execute(tmp_path, apply=True, expected_manifest_sha256=None)
