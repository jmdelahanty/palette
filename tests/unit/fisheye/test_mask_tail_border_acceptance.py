"""A reviewer exception is exact per row and cannot be forged by loose metadata."""

from types import SimpleNamespace
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from fisheye.training.mask_tail_border_acceptance import (
    ATTR,
    SCHEMA,
    acceptance_action,
    active_acceptances,
    body_digest,
    validate_acceptance_record,
)


def _record():
    body = np.zeros((8, 8), dtype=np.uint8)
    body[3:, 4] = 1
    identity = {
        "roi_idx": 0,
        "source_crop_row_ids": 5,
        "frame_indices": 19,
        "source_frame_idx": 19,
    }
    record = {
        "schema": SCHEMA,
        "roi_idx": 0,
        "row_identity": identity,
        "source_crop_run": "crop-one",
        "body_mask_sha256": body_digest(body),
        "accepted_at_mask_revision": 2,
        "accepted_by": "reviewer",
        "accepted_at_utc": "2026-09-23T12:00:00+00:00",
        "reason": "Slightly clipped visible tail tip",
        "tip_semantics": "visible_centerline_endpoint_at_crop_border_no_extrapolation",
    }
    return body, identity, record


def test_acceptance_evidence_refuses_wrong_source_and_malformed_fields():
    body, identity, record = _record()
    check = lambda value, **kwargs: validate_acceptance_record(
        "0",
        value,
        body=kwargs.get("body", body),
        source_crop_run=kwargs.get("crop", "crop-one"),
        row_identity=kwargs.get("identity", identity),
    )
    assert check(record)
    assert not check(record, crop="crop-two")
    assert not check(record, identity={**identity, "frame_indices": 20})
    altered = body.copy()
    altered[4, 3] = 1
    assert not check(record, body=altered)
    for update in (
        {"accepted_by": 11},
        {"accepted_at_utc": "yesterday"},
        {"accepted_at_utc": "2026-09-23T12:00:00"},
        {"reason": ""},
        {"accepted_at_mask_revision": -1},
        {"body_mask_sha256": "bad"},
        {"source_crop_run": ""},
        {"row_identity": {"roi_idx": 0, "frame_indices": 19}},
    ):
        with pytest.raises(ValueError):
            check({**record, **update})
    with pytest.raises(ValueError):
        validate_acceptance_record(
            "00", record, body=body, source_crop_run="crop-one", row_identity=identity
        )


def test_malformed_falsey_acceptance_map_is_not_treated_as_empty():
    group = SimpleNamespace(attrs={ATTR: []})
    with pytest.raises(ValueError):
        active_acceptances(group)


def test_accept_action_requires_text_reason():
    with pytest.raises(ValueError, match="reason must be text"):
        acceptance_action({"action": "accept", "reason": {"text": "clipped"}})


def test_subject_mask_browser_tail_status_preserves_local_work():
    node = shutil.which("node")
    assert node is not None, "Node.js is required for the browser script regression"
    repo = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [node, str(repo / "tests/unit/fisheye/test_subject_mask_tail_border_ui.cjs")],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
