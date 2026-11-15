"""Palette's MetaZebrobot reliance still holds at the pinned API (offline)."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from fisheye.shared.zebrobot_subject_reference import MZB_PIN, consumer_schema_matches_pin

VENDORED = Path(__file__).resolve().parents[2] / "contracts" / "metazebrobot_consumers"
DIGESTS = {
    "consumers.json": "6d2baa4f8776eba8f9c65a0d72c2b986f316c2334e249949687e3963390c7e05",
    "verify_consumers.py": "0e5ec76c15aef37120eee3fb160a7f3b79a1ad7e37760313996414c1f091befc",
    "consumer_openapi.json": "d80e79a68c65a04c5ec3b519648b9cf82985bde987abdce378e7a7d07e92710f",
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


@pytest.mark.parametrize(
    "served, expected",
    [
        (MZB_PIN["consumer_openapi_sha256"], True),
        # c93b2530: b2f4b20 only adds X-Delivered-By on usage links.
        ("a8cb88841430739f16958777062e1c6ed1e01c98583e6a2b0bc14e9864752d03", True),
        # Earlier slices: GET /dishes/{dish_id}/fish changed meaning since each.
        ("f5280e430d4b5f10c3643cb89a6187eacc45fdac7af55b2e81f5e315b7b754dc", False),  # 509a3eb8
        ("f120734c92d7de93091dcb0e0fa88cf411ffc44a1875fccb76a06e0c35bfd99e", False),  # 50e578ea
        ("463def80f58621b3d67db307cd6faee796fe032c0bc8e65d84ed446492ded6b1", False),  # ed39b042
        ("0" * 64, False),
        (None, None),
    ],
)
def test_only_compatible_slices_match(served, expected):
    assert consumer_schema_matches_pin(served) is expected
