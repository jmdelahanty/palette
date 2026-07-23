from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from fisheye.analysis import tail_kinematics_io as module
from fisheye.analysis.tail_kinematics_io import (
    TailKinematicsIOError,
    catalog_tail_kinematics_run,
    discover_tail_kinematics_run_options,
    load_tail_kinematics_window,
    resolve_tail_kinematics_run,
)


def _array(group: zarr.Group, name: str, values: np.ndarray) -> None:
    group.create_array(name, data=values, chunks=values.shape, overwrite=True)


def _fixture(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fps: float | None = 3.0,
    canonical_source: bool = True,
) -> zarr.Group:
    root = zarr.open_group(str(tmp_path / "tail.zarr"), mode="w")
    if fps is not None:
        root.attrs["fps"] = fps

    tail_parent = root.require_group("analysis/tail_kinematics_runs")
    tail_parent.attrs["latest"] = "tail_run"
    tail_parent.attrs["latest_complete"] = "tail_run"
    tail = tail_parent.require_group("tail_run")
    tail.attrs.update(
        {
            "palette_run_completion_status": "complete",
            "schema_version": 2,
            "method": "body_frame_spline_tangent",
            "source_subject_shape_run": "analysis/subject_shape_runs/shape_run",
            "source_subject_shape_publication_manifest_sha256": "a" * 64,
        }
    )
    frames = np.arange(0, 30, 3, dtype=np.int64)
    angles = np.arange(frames.size * 10, dtype=np.float32).reshape(frames.size, 10)
    _array(tail, "source_acquisition_frame_index", frames)
    _array(tail, "valid", np.asarray([True, True, True, False, True] * 2))
    _array(tail, "tail_angle_deg", angles)
    _array(tail, "tail_angle_sample_s", np.linspace(0.05, 0.95, 10, dtype=np.float32))
    _array(tail, "tail_tip_angle_deg", angles[:, -1])
    _array(
        tail,
        "tail_tip_lateral_deflection_px",
        np.linspace(-2, 2, frames.size, dtype=np.float32),
    )
    _array(tail, "tail_angle_rms_deg", np.sqrt(np.mean(angles**2, axis=1)))

    shape = root.require_group("analysis/subject_shape_runs/shape_run")
    shape.attrs["palette_run_completion_status"] = "complete"
    _array(shape, "source_acquisition_frame_index", frames.copy())
    body = shape.require_group("components/subject_body")
    curvature = (
        np.arange(frames.size * 32, dtype=np.float32).reshape(frames.size, 32) / 100.0
    )
    _array(body, "tail_curvature_px_inv", curvature)
    _array(body, "tail_sample_s", np.linspace(0, 1, 32, dtype=np.float32))
    _array(body, "tail_sample_valid", np.asarray([True] * frames.size))
    if canonical_source:
        binding = SimpleNamespace(
            array_node=body["tail_curvature_px_inv"],
            validity_node=body["tail_sample_valid"],
        )
        proof = SimpleNamespace(
            manifest=SimpleNamespace(record_sha256="a" * 64),
            require_scalar_surface=lambda *_args, **_kwargs: binding,
        )
        monkeypatch.setattr(
            module,
            "load_persisted_subject_shape_coordinate_publication",
            lambda *_args, **_kwargs: proof,
        )
    monkeypatch.setattr(
        module,
        "load_tail_kinematics_coordinate_publication",
        lambda *_args, **_kwargs: SimpleNamespace(
            _run=tail,
            manifest=SimpleNamespace(record_sha256="b" * 64),
            measurements={
                "tail_tip_angle_deg": SimpleNamespace(),
                "tail_tip_lateral_deflection_px": SimpleNamespace(),
                "tail_angle_rms_deg": SimpleNamespace(),
            },
            source=SimpleNamespace(
                run_path="analysis/subject_shape_runs/shape_run",
            ),
        ),
    )
    return root


def test_discovers_catalogs_and_bounds_tail_run(tmp_path, monkeypatch) -> None:
    root = _fixture(tmp_path, monkeypatch)

    options = discover_tail_kinematics_run_options(root)
    assert [option.run_name for option in options] == ["tail_run"]
    assert options[0].is_latest
    assert options[0].sample_count == 10

    catalog = catalog_tail_kinematics_run(root)
    assert catalog.fps == 3.0
    assert catalog.fps_source == "root.attrs.fps"
    assert catalog.time_start_s == 0.0
    assert catalog.time_stop_s == 9.0
    assert catalog.source_shape_run_name == "shape_run"
    assert catalog.source_shape_run_path == "analysis/subject_shape_runs/shape_run"
    assert catalog.source_curvature_sample_count == 32
    assert catalog.tail_publication_manifest_sha256 == "b" * 64

    window = load_tail_kinematics_window(
        root,
        start_s=2.0,
        stop_s=5.0,
        scalar_series=("tail_tip_angle_deg", "tail_tip_lateral_deflection_px"),
    )
    np.testing.assert_array_equal(window.frame_indices, [6, 9, 12, 15])
    np.testing.assert_allclose(window.time_seconds, [2, 3, 4, 5])
    assert window.angle_deg.shape == (4, 10)
    assert window.dense_curvature_px_inv.shape == (4, 32)
    assert np.isnan(window.angle_deg[1]).all()
    assert np.isnan(window.scalar_series["tail_tip_angle_deg"][1])
    assert window.source_paths["dense_curvature"].endswith("tail_curvature_px_inv")


@pytest.mark.parametrize(
    "run_spec",
    (
        "tail_run",
        "analysis/tail_kinematics_runs/tail_run",
    ),
)
def test_explicit_tail_run_accepts_only_controlled_name_forms(
    tmp_path,
    monkeypatch,
    run_spec,
) -> None:
    root = _fixture(tmp_path, monkeypatch)

    _run, run_name, run_path = resolve_tail_kinematics_run(root, run_spec)

    assert run_name == "tail_run"
    assert run_path == "analysis/tail_kinematics_runs/tail_run"


@pytest.mark.parametrize(
    "run_spec",
    (
        "analysis/subject_shape_runs/tail_run",
        "analysis/tail_posture_view_runs/tail_run",
        "nested/tail_run",
        "analysis/tail_kinematics_runs/nested/tail_run",
        "analysis/tail_kinematics_runs",
        "/analysis/tail_kinematics_runs/tail_run",
        "analysis/tail_kinematics_runs/tail_run/",
        "analysis/tail_kinematics_runs//tail_run",
    ),
)
def test_explicit_tail_run_rejects_wrong_family_or_nested_paths(
    tmp_path,
    monkeypatch,
    run_spec,
) -> None:
    root = _fixture(tmp_path, monkeypatch)

    with pytest.raises(TailKinematicsIOError, match="bare child name or the exact path"):
        resolve_tail_kinematics_run(root, run_spec)


def test_implicit_tail_run_fails_closed_during_selector_handoff(
    tmp_path,
    monkeypatch,
) -> None:
    root = _fixture(tmp_path, monkeypatch)
    parent = root["analysis/tail_kinematics_runs"]
    candidate = parent.create_group("candidate")
    candidate.attrs.update(
        {
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
        }
    )
    parent.attrs["latest"] = "candidate"
    parent.attrs["latest_complete"] = "candidate"

    with pytest.raises(TailKinematicsIOError, match="activation may be in progress"):
        resolve_tail_kinematics_run(root)

    _run, run_name, _run_path = resolve_tail_kinematics_run(root, "tail_run")
    assert run_name == "tail_run"


@pytest.mark.parametrize(
    ("latest", "latest_complete"),
    (("tail_run", "candidate"), ("candidate", "tail_run")),
)
def test_implicit_tail_run_rejects_each_intermediate_selector_pair(
    tmp_path,
    monkeypatch,
    latest: str,
    latest_complete: str,
) -> None:
    root = _fixture(tmp_path, monkeypatch)
    parent = root["analysis/tail_kinematics_runs"]
    candidate = parent.create_group("candidate")
    candidate.attrs.update(
        {
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
        }
    )
    parent.attrs["latest"] = latest
    parent.attrs["latest_complete"] = latest_complete

    with pytest.raises(TailKinematicsIOError, match="activation may be in progress"):
        resolve_tail_kinematics_run(root)

    _run, run_name, _run_path = resolve_tail_kinematics_run(root, "tail_run")
    assert run_name == "tail_run"


def test_catalog_uses_sealed_source_path_not_legacy_run_alias(
    tmp_path,
    monkeypatch,
) -> None:
    root = _fixture(tmp_path, monkeypatch)
    root["analysis/tail_kinematics_runs/tail_run"].attrs[
        "source_subject_shape_run"
    ] = "wrong_shape_alias"

    catalog = catalog_tail_kinematics_run(root)

    assert catalog.source_shape_run_path == "analysis/subject_shape_runs/shape_run"
    assert catalog.source_shape_run_name == "shape_run"


def test_refuses_oversized_projection(tmp_path, monkeypatch) -> None:
    root = _fixture(tmp_path, monkeypatch)
    with pytest.raises(TailKinematicsIOError, match="viewer limit"):
        load_tail_kinematics_window(root, start_s=0, stop_s=9, max_rows=4)


def test_fails_closed_when_subject_shape_rows_are_misaligned(tmp_path, monkeypatch) -> None:
    root = _fixture(tmp_path, monkeypatch)
    root["analysis/subject_shape_runs/shape_run/source_acquisition_frame_index"][3] = 10
    with pytest.raises(TailKinematicsIOError, match="lineage does not align"):
        load_tail_kinematics_window(root, start_s=2, stop_s=5)


def test_reader_rejects_frame_index_alias_without_canonical_array(
    tmp_path,
    monkeypatch,
) -> None:
    root = _fixture(tmp_path, monkeypatch)
    tail = root["analysis/tail_kinematics_runs/tail_run"]
    values = np.asarray(tail["source_acquisition_frame_index"][:])
    del tail["source_acquisition_frame_index"]
    _array(tail, "frame_indices", values)

    with pytest.raises(
        TailKinematicsIOError,
        match="source_acquisition_frame_index.*missing",
    ):
        catalog_tail_kinematics_run(root)


def test_requires_fps_for_time_projection(tmp_path, monkeypatch) -> None:
    root = _fixture(tmp_path, monkeypatch, fps=None)
    with pytest.raises(TailKinematicsIOError, match="requires positive recording fps"):
        load_tail_kinematics_window(root, start_s=0, stop_s=1)


def test_rejects_legacy_subject_shape_source_without_canonical_publication(
    tmp_path,
    monkeypatch,
) -> None:
    root = _fixture(
        tmp_path,
        monkeypatch,
        canonical_source=False,
    )

    with pytest.raises(TailKinematicsIOError, match="not an exact canonical"):
        catalog_tail_kinematics_run(root)


@pytest.mark.parametrize(
    ("manifest_sha256", "message"),
    (
        (None, "must persist a lowercase SHA-256"),
        ("b" * 64, "publication digest does not match"),
    ),
)
def test_rejects_missing_or_mismatched_source_publication_digest(
    tmp_path,
    monkeypatch,
    manifest_sha256,
    message,
) -> None:
    root = _fixture(tmp_path, monkeypatch)
    tail = root["analysis/tail_kinematics_runs/tail_run"]
    if manifest_sha256 is None:
        del tail.attrs["source_subject_shape_publication_manifest_sha256"]
    else:
        tail.attrs["source_subject_shape_publication_manifest_sha256"] = manifest_sha256

    with pytest.raises(TailKinematicsIOError, match=message):
        catalog_tail_kinematics_run(root)


def test_scalar_catalog_requires_typed_measurement_publication(
    tmp_path,
    monkeypatch,
) -> None:
    root = _fixture(tmp_path, monkeypatch)
    tail = root["analysis/tail_kinematics_runs/tail_run"]
    monkeypatch.setattr(
        module,
        "load_tail_kinematics_coordinate_publication",
        lambda *_args, **_kwargs: SimpleNamespace(
            _run=tail,
            manifest=SimpleNamespace(record_sha256="b" * 64),
            measurements={"tail_tip_angle_deg": SimpleNamespace()},
            source=SimpleNamespace(
                run_path="analysis/subject_shape_runs/shape_run",
            ),
        ),
    )

    catalog = catalog_tail_kinematics_run(root)

    assert catalog.scalar_series == ("tail_tip_angle_deg",)


def test_window_discards_copy_when_tail_publication_changes(
    tmp_path,
    monkeypatch,
) -> None:
    root = _fixture(tmp_path, monkeypatch)
    tail = root["analysis/tail_kinematics_runs/tail_run"]
    calls = 0

    def _tail_proof(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            _run=tail,
            manifest=SimpleNamespace(
                record_sha256=("b" if calls == 1 else "c") * 64
            ),
            measurements={
                "tail_tip_angle_deg": SimpleNamespace(),
                "tail_tip_lateral_deflection_px": SimpleNamespace(),
                "tail_angle_rms_deg": SimpleNamespace(),
            },
            source=SimpleNamespace(
                run_path="analysis/subject_shape_runs/shape_run",
            ),
        )

    monkeypatch.setattr(
        module,
        "load_tail_kinematics_coordinate_publication",
        _tail_proof,
    )

    with pytest.raises(TailKinematicsIOError, match="changed while.*copied"):
        load_tail_kinematics_window(root, start_s=2.0, stop_s=5.0)


def test_window_discards_copy_when_source_publication_changes(
    tmp_path,
    monkeypatch,
) -> None:
    root = _fixture(tmp_path, monkeypatch)
    body = root["analysis/subject_shape_runs/shape_run/components/subject_body"]
    binding = SimpleNamespace(
        array_node=body["tail_curvature_px_inv"],
        validity_node=body["tail_sample_valid"],
    )
    calls = 0

    def _shape_proof(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            manifest=SimpleNamespace(
                record_sha256=("a" if calls < 3 else "c") * 64
            ),
            require_scalar_surface=lambda *_args, **_kwargs: binding,
        )

    monkeypatch.setattr(
        module,
        "load_persisted_subject_shape_coordinate_publication",
        _shape_proof,
    )

    with pytest.raises(TailKinematicsIOError, match="changed while.*copied"):
        load_tail_kinematics_window(root, start_s=2.0, stop_s=5.0)
