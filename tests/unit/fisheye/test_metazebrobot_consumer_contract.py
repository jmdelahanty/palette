"""Palette's MetaZebrobot reliance still holds at the pinned API (offline)."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from fisheye.shared.zebrobot_subject_reference import MZB_PIN

VENDORED = Path(__file__).resolve().parents[2] / "contracts" / "metazebrobot_consumers"
DIGESTS = {
    "consumers.json": "8d2b38780ad08c7111cb6f5bd81723169e5ce96cad44f489e6cf442ee071bda3",
    "verify_consumers.py": "0e5ec76c15aef37120eee3fb160a7f3b79a1ad7e37760313996414c1f091befc",
    "consumer_openapi.json": "f5280e430d4b5f10c3643cb89a6187eacc45fdac7af55b2e81f5e315b7b754dc",
}


@pytest.mark.parametrize("name", sorted(DIGESTS))
def test_vendored_files_are_the_pinned_bytes(name):
    assert hashlib.sha256((VENDORED / name).read_bytes()).hexdigest() == DIGESTS[name]


def test_intake_pin_names_the_vendored_contract():
    assert MZB_PIN["consumer_openapi_sha256"] == DIGESTS["consumer_openapi.json"]
    assert MZB_PIN["consumers_json_sha256"] == DIGESTS["consumers.json"]


def test_palette_reliance_is_served_by_the_pinned_api():
    result = subprocess.run(
        [
            sys.executable, "verify_consumers.py",
            "--spec", "consumers.json",
            "--openapi-file", "consumer_openapi.json",
            "--consumer", "palette",
        ],
        cwd=VENDORED, capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout
