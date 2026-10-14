"""Citrus stimulus vocabulary, read from the pinned agent-contracts document.

``contracts/citrus_stimulus_vocabulary_v1.json`` is a byte-for-byte copy of
agent-contracts ``citrus-palette/stimulus_vocabulary/stimulus_vocabulary_v1.json``
(PR 81, 93, 95, 96, 97, 100; main 8927594). Only ``status == "live"`` modes are
accepted; a planned mode stays out of :data:`STIMULUS_MODE` until a contract
revision marks it live, because Palette fails closed on unknown mode ids.
"""

from __future__ import annotations

from hashlib import sha256
from importlib.resources import files as resource_files
import json
from types import MappingProxyType
from typing import Any, Mapping

VOCABULARY_FILE = "citrus_stimulus_vocabulary_v1.json"
VOCABULARY_SHA256 = "7f8dc387802965cea9982cf46b5da2c62395047c406d8ba4fcf91f5f4b364318"


def _load() -> Mapping[str, Any]:
    data = resource_files("fisheye.shared").joinpath("contracts", VOCABULARY_FILE).read_bytes()
    if sha256(data).hexdigest() != VOCABULARY_SHA256:
        raise RuntimeError(f"packaged_contract_drift:{VOCABULARY_FILE}")
    document = json.loads(data)
    if (
        document.get("schema_id") != "citrus.stimulus_vocabulary"
        or document.get("schema_version") != 1
    ):
        raise RuntimeError(f"unsupported stimulus vocabulary in {VOCABULARY_FILE}")
    return document


_DOCUMENT = _load()
_MODES = tuple(_DOCUMENT["modes"])

STIMULUS_MODE: Mapping[int, str] = MappingProxyType(
    {int(mode["id"]): mode["name"] for mode in _MODES if mode["status"] == "live"}
)
STIMULUS_MODE_PLANNED: Mapping[int, str] = MappingProxyType(
    {int(mode["id"]): mode["name"] for mode in _MODES if mode["status"] == "planned"}
)
STIMULUS_FAMILY: Mapping[int, str] = MappingProxyType(
    {int(mode["id"]): mode["family"] for mode in _MODES}
)
QUANTIZATION_DEFAULT: str = _DOCUMENT["color"]["quantization"]["default_when_absent"]
TRIAL_INDEX_ACCEPTED_VERSIONS: frozenset[int] = frozenset(
    _DOCUMENT["trial_index"]["consumer_accepts"]
)
