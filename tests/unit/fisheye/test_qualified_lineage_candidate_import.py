from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from fisheye.shared.zarr import qualified_lineage_candidate_import as module


class _Attrs(dict):
    def put(self, values: dict[str, object]) -> None:
        self.clear()
        self.update(values)


class _Array:
    def __init__(self) -> None:
        self.attrs = _Attrs({"shadow_only": True, "benchmark_only": True})


class _Group:
    def __init__(
        self,
        *,
        attrs: dict[str, object] | None = None,
        arrays: tuple[tuple[str, _Array], ...] = (),
        children: dict[str, "_Group"] | None = None,
    ) -> None:
        self.attrs = _Attrs(attrs or {})
        self._arrays = arrays
        self._children = children or {}

    def arrays(self) -> tuple[tuple[str, _Array], ...]:
        return self._arrays

    def __contains__(self, name: str) -> bool:
        return name in self._children

    def __getitem__(self, name: str) -> "_Group":
        current = self
        for part in name.split("/"):
            current = current._children[part]
        return current

    def get(self, name: str) -> "_Group | None":
        return self._children.get(name)


def test_mark_production_candidate_removes_benchmark_state() -> None:
    root_array = _Array()
    nested_array = _Array()
    run = _Group(
        attrs={
            "status": "complete",
            "stage_selector_eligible": False,
            "shadow_only": True,
            "benchmark_only": True,
        },
        arrays=(("root", root_array),),
        children={"instances": _Group(arrays=(("values", nested_array),))},
    )

    module._mark_production_candidate(run, nested_tables=("instances",))

    assert run.attrs["production_candidate"] is True
    assert run.attrs["immutable_snapshot"] is True
    assert run.attrs["stage_selector_eligible"] is False
    assert "shadow_only" not in run.attrs
    assert "benchmark_only" not in run.attrs
    for array in (root_array, nested_array):
        assert array.attrs == {"selector_eligible": False}


def test_clip_evidence_requires_one_directory_per_member(
    tmp_path: Path,
) -> None:
    manifest = {
        "payload": {
            "source_detection": {
                "members": [
                    {
                        "clip_index": 0,
                        "source_refined_run_id": "refined_clip_0",
                    }
                ]
            }
        }
    }

    with pytest.raises(ValueError, match="Expected one evidence directory"):
        module._clip_evidence(tmp_path, recording_manifest=manifest)

    (tmp_path / "clip_000000_a").mkdir()
    (tmp_path / "clip_000000_b").mkdir()
    with pytest.raises(ValueError, match="found 2"):
        module._clip_evidence(tmp_path, recording_manifest=manifest)


def test_atomic_import_rejects_invalid_existing_candidate(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "analysis.zarr"
    archive.mkdir()
    target = archive / "crop_runs" / "crop_v2"
    target.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="not reusable"):
        module._atomic_import(
            archive=archive,
            local_run=tmp_path / "unused",
            family="crop_runs",
            run_id="crop_v2",
            role="crop",
            selectors={},
            validator=lambda _path: {"valid": False, "errors": ["digest mismatch"]},
            copy_backend="python",
        )


def test_atomic_import_reuses_exact_candidate_without_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "analysis.zarr"
    archive.mkdir()
    target = archive / "crop_runs" / "crop_v2"
    target.mkdir(parents=True)
    run = _Group(attrs={"stage_selector_eligible": False})
    family = _Group(children={"crop_v2": run})
    root = _Group(children={"crop_runs": family})
    monkeypatch.setattr(module, "open_zarr_root", lambda *_args, **_kwargs: root)

    receipt = module._atomic_import(
        archive=archive,
        local_run=tmp_path / "unused",
        family="crop_runs",
        run_id="crop_v2",
        role="crop",
        selectors=module._selector_snapshot(root, "crop_runs"),
        validator=lambda _path: {"valid": True, "errors": []},
        copy_backend="python",
    )

    assert receipt["status"] == "reused_exact_complete_candidate"


def test_publish_fails_before_staging_on_recording_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "analysis.zarr"
    archive.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    fake = SimpleNamespace(recording_identity="recording_expected")
    monkeypatch.setattr(
        module,
        "inspect_qualified_lineage_source",
        lambda **_kwargs: fake,
    )
    monkeypatch.setattr(
        module,
        "open_zarr_root",
        lambda *_args, **_kwargs: _Group(attrs={"recording_id": "recording_other"}),
    )

    with pytest.raises(ValueError, match="recording_id differs"):
        module.publish_qualified_lineage_candidate(
            analysis_zarr=archive,
            refined_archive=tmp_path / "refined.zarr",
            refined_run_id="refined_v2",
            refined_clip_evidence_root=tmp_path / "evidence",
            crop_archive=tmp_path / "crop.zarr",
            crop_run_id="crop_v2",
            scratch_root=scratch,
        )

    assert tuple(scratch.iterdir()) == ()
