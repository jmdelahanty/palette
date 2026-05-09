from __future__ import annotations

from pathlib import Path
from typing import Any

from fisheye.utils import audit_analysis_staleness as audit


class FakeGroup:
    def __init__(
        self,
        attrs: dict[str, Any] | None = None,
        children: dict[str, "FakeGroup"] | None = None,
    ) -> None:
        self.attrs = attrs or {}
        self.children = children or {}

    def __contains__(self, key: str) -> bool:
        return key in self.children

    def __getitem__(self, key: str) -> "FakeGroup":
        return self.children[key]

    def group_keys(self) -> list[str]:
        return list(self.children)


def _root_with_eye_angle_source(
    *,
    source_fingerprint: str = "abc",
    expected_fingerprint: str | None = "abc",
    latest_shape: str = "shape_1",
    source_stale: bool = False,
) -> FakeGroup:
    shape_attrs: dict[str, Any] = {
        "lineage_hash": source_fingerprint,
        "fingerprint_status": "complete",
    }
    if source_stale:
        shape_attrs["source_subject_mask_stale"] = {"state": "stale", "reason": "mask_edit"}
    source_ref: dict[str, Any] = {"path": "analysis/subject_shape_runs/shape_1"}
    if expected_fingerprint is not None:
        source_ref["fingerprint"] = expected_fingerprint
    return FakeGroup(
        children={
            "analysis": FakeGroup(
                children={
                    "subject_shape_runs": FakeGroup(
                        attrs={"latest": latest_shape},
                        children={
                            "shape_1": FakeGroup(shape_attrs),
                            "shape_2": FakeGroup({"lineage_hash": "newer"}),
                        },
                    ),
                    "eye_angle_runs": FakeGroup(
                        children={
                            "eye_1": FakeGroup(
                                {
                                    "schema_id": "analysis.eye_angle_runs",
                                    "source_refs": {"subject_shape": source_ref},
                                }
                            )
                        }
                    ),
                }
            )
        }
    )


def _run_single_eye_audit(root: FakeGroup, monkeypatch, tmp_path: Path, **kwargs: Any) -> audit.RunAudit:
    monkeypatch.setattr(audit, "open_zarr_root", lambda path, mode="r": root)
    results = audit.audit_zarr_analysis_staleness(
        tmp_path / "archive.zarr",
        run_families={"eye_angle_run"},
        **kwargs,
    )
    assert len(results) == 1
    return results[0]


def test_audit_marks_matching_source_fingerprint_fresh(monkeypatch, tmp_path: Path) -> None:
    result = _run_single_eye_audit(_root_with_eye_angle_source(), monkeypatch, tmp_path)

    assert result.status == "fresh"
    assert result.sources[0].status == "fresh"


def test_audit_marks_fingerprint_mismatch_stale(monkeypatch, tmp_path: Path) -> None:
    root = _root_with_eye_angle_source(source_fingerprint="actual", expected_fingerprint="expected")

    result = _run_single_eye_audit(root, monkeypatch, tmp_path)

    assert result.status == "stale"
    assert result.sources[0].status == "stale"
    assert result.sources[0].expected_fingerprint == "expected"
    assert result.sources[0].actual_fingerprint == "actual"


def test_audit_warns_when_source_is_not_latest_by_default(monkeypatch, tmp_path: Path) -> None:
    root = _root_with_eye_angle_source(latest_shape="shape_2")

    result = _run_single_eye_audit(root, monkeypatch, tmp_path)

    assert result.status == "warning"
    assert result.sources[0].status == "source_not_latest"
    assert result.sources[0].latest_run_id == "shape_2"
    assert result.sources[0].referenced_run_id == "shape_1"


def test_audit_can_require_latest_sources(monkeypatch, tmp_path: Path) -> None:
    root = _root_with_eye_angle_source(latest_shape="shape_2")

    result = _run_single_eye_audit(
        root,
        monkeypatch,
        tmp_path,
        require_latest_sources=True,
    )

    assert result.status == "stale"
    assert result.sources[0].status == "stale"
    assert "latest" in result.sources[0].message


def test_audit_warns_when_expected_fingerprint_missing(monkeypatch, tmp_path: Path) -> None:
    root = _root_with_eye_angle_source(expected_fingerprint=None)

    result = _run_single_eye_audit(root, monkeypatch, tmp_path)

    assert result.status == "warning"
    assert result.sources[0].status == "unverifiable_missing_expected_fingerprint"


def test_audit_marks_source_explicit_stale(monkeypatch, tmp_path: Path) -> None:
    root = _root_with_eye_angle_source(source_stale=True)

    result = _run_single_eye_audit(root, monkeypatch, tmp_path)

    assert result.status == "stale"
    assert result.sources[0].status == "source_explicit_stale"


def test_audit_infers_common_source_run_attr_paths(monkeypatch, tmp_path: Path) -> None:
    root = FakeGroup(
        children={
            "analysis": FakeGroup(
                children={
                    "track_kinematics_runs": FakeGroup(
                        children={
                            "offline": FakeGroup(
                                children={
                                    "tk_1": FakeGroup({"lineage_hash": "tkhash"}),
                                }
                            )
                        }
                    ),
                    "swim_bout_runs": FakeGroup(
                        children={
                            "bouts_1": FakeGroup(
                                {
                                    "source_track_kinematics_run": "tk_1",
                                    "source_fingerprints": {
                                        "source_track_kinematics_run": "tkhash",
                                    },
                                }
                            )
                        }
                    ),
                }
            )
        }
    )

    monkeypatch.setattr(audit, "open_zarr_root", lambda path, mode="r": root)
    results = audit.audit_zarr_analysis_staleness(
        tmp_path / "archive.zarr",
        run_families={"swim_bout_run"},
    )

    assert len(results) == 1
    assert results[0].status == "fresh"
    assert results[0].sources[0].path == "analysis/track_kinematics_runs/offline/tk_1"
