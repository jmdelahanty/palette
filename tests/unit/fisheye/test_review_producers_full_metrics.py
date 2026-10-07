"""Review-run producers finalize editable subject masks with full component metrics.

Browser mask Apply refuses runs that declare cheap metrics, so a review run
made for labeling must start full.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from fisheye.training import training_review_artifact_publication
from fisheye.utils import bootstrap_training_review_surfaces


@pytest.mark.parametrize("module", [training_review_artifact_publication, bootstrap_training_review_surfaces])
def test_review_producers_finalize_masks_with_full_metrics(module):
    calls = [
        node for node in ast.walk(ast.parse(inspect.getsource(module)))
        if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == "finalize_subject_masks"
    ]
    assert calls, "producer no longer calls finalize_subject_masks"
    for call in calls:
        level = {k.arg: k.value for k in call.keywords}.get("metric_level")
        assert isinstance(level, ast.Constant) and level.value == "full", ast.unparse(call)[:200]
