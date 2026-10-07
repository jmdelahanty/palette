"""Review-run producers finalize editable subject masks at the level mask Apply QC requires.

Browser mask Apply refuses runs whose declared component metric level differs
from ``EDITABLE_REVIEW_METRIC_LEVEL``, so producers of runs made for labeling
pass that constant, and a run finalized at it passes Apply QC's contract check.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import zarr

from fisheye.labeling import web_subject_mask_apply_qc as apply_qc
from fisheye.refinement import finalize_subject_masks as finalizer
from fisheye.refinement.finalize_subject_masks import EDITABLE_REVIEW_METRIC_LEVEL
from fisheye.training import training_review_artifact_publication
from fisheye.utils import bootstrap_training_review_surfaces
from tests.unit.fisheye.test_finalize_subject_masks import _build_probability_root, _patch_refined_subject_provenance


@pytest.mark.parametrize("module", [training_review_artifact_publication, bootstrap_training_review_surfaces])
def test_review_producers_finalize_at_the_apply_qc_metric_level(module):
    calls = [
        node for node in ast.walk(ast.parse(inspect.getsource(module)))
        if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == "finalize_subject_masks"
    ]
    assert calls, "producer no longer calls finalize_subject_masks"
    for call in calls:
        level = {k.arg: k.value for k in call.keywords}.get("metric_level")
        assert isinstance(level, ast.Name) and level.id == "EDITABLE_REVIEW_METRIC_LEVEL", ast.unparse(call)[:200]


def _finalized_run(tmp_path: Path, monkeypatch, level: str) -> zarr.Group:
    _patch_refined_subject_provenance(monkeypatch)
    zarr_path = tmp_path / f"analysis_{level}.zarr"
    _build_probability_root(zarr_path)
    finalizer.finalize_subject_masks(
        zarr_path, subject_run="subject_probs_001", refined_run="refined_review", chunk_size=1,
        metric_level=level, defer_registry_status=True,
    )
    return zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)["refined_subject_masks_runs/refined_review"]


def test_a_run_finalized_at_the_constant_passes_apply_qc_and_cheap_does_not(tmp_path, monkeypatch):
    run = _finalized_run(tmp_path, monkeypatch, EDITABLE_REVIEW_METRIC_LEVEL)
    assert run.attrs["component_metric_level"] == EDITABLE_REVIEW_METRIC_LEVEL
    apply_qc._require_compatible_contract(run)

    cheap = _finalized_run(tmp_path, monkeypatch, "cheap")
    with pytest.raises(RuntimeError, match="requires full component metrics"):
        apply_qc._require_compatible_contract(cheap)
