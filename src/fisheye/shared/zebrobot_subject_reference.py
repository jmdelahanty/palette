"""Orange's per-camera Zebrobot subject reference, resolved at intake.

Orange seals ``subject_references`` into its start snapshot at record start: a
declared status plus the dish identity and revision Zebrobot served then, never
biological fields. At intake Palette fetches the dish from the read-only
MetaZebrobot API, verifies ``dish_uuid``, compares ``(dish_uuid, revision)``
with the recorded pair, computes days post fertilization at the recording date
from ``dof`` (the server's ``dpf`` changes daily) and keeps cross fields
labelled as resolved at intake (``cross.cache_updated_at`` is a cache refresh
time, not a content revision). A declared absence is recorded, never filled.
Zebrobot being unreachable at intake is an error so intake can retry later.

Version 2 references also seal the MetaZebrobot build that served them
(``GET /version``: ``service_commit``, ``service_commit_dirty``,
``consumer_schema_sha256``). Palette records it and whether the served
consumer schema matches the MetaZebrobot pin it relies on; it never refuses on
that alone, because the fields Palette reads are re-fetched and checked here.
MetaZebrobot response shapes and meaning are owned by MetaZebrobot (see
``MZB_PIN``); this module keeps no copy of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from functools import lru_cache
import hashlib
from importlib.resources import files
import json
import re
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

REFERENCE_SCHEMA_ID = "orange.recording_subject_reference"
REFERENCE_SCHEMA_VERSIONS = (1, 2)
API_SCHEMA_VERSION = 2
STATUSES = ("collected", "not_collected", "lookup_failed")
SOURCE_KIND = "zebrobot_api_resolved_at_intake"
_UUID4 = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
_SERVER_TIME = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")


class SubjectReferenceError(ValueError):
    """The recorded reference is malformed or contradicts Zebrobot."""


class ZebrobotUnavailable(RuntimeError):
    """Zebrobot could not be read at intake; retry later rather than finalize."""


# agent-contracts PR 53: v1 at 6098ca47, v2 at 879ae3f6.
SCHEMA_FILES = {
    1: (
        "orange_recording_subject_reference_v1.schema.json",
        "3c4ba74f0f95f76dbb8b3d65aa8bd39fd409383388ee8debec3845ef26c1a930",
    ),
    2: (
        "orange_recording_subject_reference_v2.schema.json",
        "d0f300fdbd71747f44219baf7bb5aea262ebbd275f85f877df9f1a00aa92e935",
    ),
}
# Shapes: metazebrobot docs/api/consumer_openapi.json; meaning:
# docs/zebrobot_snapshot.md; Palette's reliance: agent-contracts PR 54.
MZB_PIN = {
    "repo": "jmdelahanty/metazebrobot",
    "commit": "509a3eb88d6ff44fe07e7ea20d212be5eafe46b7",
    "consumer_openapi_sha256": "f5280e430d4b5f10c3643cb89a6187eacc45fdac7af55b2e81f5e315b7b754dc",
    # agent-contracts metazebrobot-consumers/consumers.json (PR 54, merge 5fc735fe).
    "consumers_json_sha256": "8d2b38780ad08c7111cb6f5bd81723169e5ce96cad44f489e6cf442ee071bda3",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SubjectReferenceError(message)


@lru_cache(maxsize=None)
def _schema(version: int) -> dict[str, Any]:
    name, digest = SCHEMA_FILES[version]
    data = files("fisheye.shared").joinpath("contracts").joinpath(name).read_bytes()
    _require(hashlib.sha256(data).hexdigest() == digest, "packaged_contract_drift:" + name)
    return json.loads(data)


def _schema_version(reference: Any, label: str) -> int:
    version = reference.get("schema_version") if isinstance(reference, Mapping) else None
    _require(
        type(version) is int and version in SCHEMA_FILES,
        f"{label} schema_version {version!r} is not a pinned version",
    )
    return version


def _schema_errors(references: Mapping[str, Any], version: int) -> list[str]:
    from jsonschema import Draft202012Validator

    return [
        "/".join(str(part) for part in error.absolute_path) + ": " + error.message
        for error in Draft202012Validator(_schema(version)).iter_errors(references)
    ]


def _revision(value: Any, label: str) -> int:
    _require(type(value) is int and value >= 1, f"invalid {label}")
    return value


def validate_subject_reference(reference: Mapping[str, Any], camera: str) -> dict[str, Any]:
    """Pinned Orange schema (by its version) plus Palette's semantic checks."""

    label = f"subject_references[{camera}]"
    errors = _schema_errors({camera: reference}, _schema_version(reference, label))
    _require(not errors, f"{label} schema: {errors[0] if errors else ''}")
    if reference["zebrobot"] is not None:
        _recorded_at(reference["zebrobot"]["queried_at_utc"], label)
    if reference["status"] == "lookup_failed":
        _require(reference["zebrobot"]["error"] is not None, f"{label} lookup_failed without error")
    if reference["status"] != "collected":
        return dict(reference)
    dish = reference["dish"]
    _require(_UUID4.match(dish["dish_uuid"]) is not None, f"{label} dish_uuid is not a lowercase UUID4")
    _revision(dish["revision"], f"{label} dish revision")
    _require(_SERVER_TIME.match(dish["updated_at"]) is not None, f"{label} dish updated_at")
    lookup = reference["dish_fish_lookup"]
    _require(lookup is not None, f"{label} collected without dish_fish_lookup")
    _require(
        lookup["status"] == "complete" or reference["dish_fish"] == [],
        f"{label} incomplete fish lookup carries fish",
    )
    seen: set[str] = set()
    for item in reference["dish_fish"]:
        _require(item["fish_id"] not in seen, f"{label} duplicate fish_id")
        seen.add(item["fish_id"])
        _revision(item["revision"], f"{label} fish revision")
    return dict(reference)


def validate_subject_references(
    references: Any, cameras: list[str]
) -> dict[str, dict[str, Any]]:
    """Every camera parent must carry an entry: absence is never implied."""

    _require(isinstance(references, Mapping), "subject_references must be an object")
    _require(
        set(references) == set(cameras),
        "subject_references must declare exactly every camera parent",
    )
    validated = {
        camera: validate_subject_reference(references[camera], camera) for camera in cameras
    }
    _require(
        len({ref["schema_version"] for ref in validated.values()}) <= 1,
        "subject_references mixes schema versions across cameras",
    )
    return validated


def _recorded_at(value: Any, label: str) -> datetime:
    _require(isinstance(value, str) and value.endswith("Z"), f"{label} queried_at_utc")
    try:
        moment = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise SubjectReferenceError(f"{label} queried_at_utc") from exc
    return moment.astimezone(timezone.utc)


Fetch = Callable[[str], tuple[int, Any]]


def http_fetch(url: str, *, timeout: float = 10.0) -> tuple[int, Any]:
    """GET JSON; raise ZebrobotUnavailable for transport/5xx (retryable)."""

    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed http(s) base
            return response.status, json.loads(response.read())
    except HTTPError as exc:
        if exc.code >= 500:
            raise ZebrobotUnavailable(f"zebrobot {exc.code} for {url}") from exc
        try:
            body = json.loads(exc.read())
        except ValueError:
            body = None
        return exc.code, body
    except (URLError, TimeoutError, OSError) as exc:
        raise ZebrobotUnavailable(f"zebrobot unreachable: {url}: {exc}") from exc


def _server_date(value: Any, label: str) -> date:
    _require(isinstance(value, str) and re.fullmatch(r"\d{8}", value or "") is not None, f"invalid {label}")
    return date(int(value[:4]), int(value[4:6]), int(value[6:]))


@dataclass(frozen=True)
class ResolvedSubjectReference:
    status: str
    reason: str
    metadata: dict[str, Any] | None
    source: dict[str, Any]


def _served_build(zebrobot: Mapping[str, Any]) -> dict[str, Any]:
    """The MetaZebrobot build sealed at record start (v2), against Palette's pin."""

    if "consumer_schema_sha256" not in zebrobot:
        return {}
    served = zebrobot["consumer_schema_sha256"]
    return {
        "zebrobot_service_commit": zebrobot["service_commit"],
        "zebrobot_service_commit_dirty": zebrobot["service_commit_dirty"],
        "zebrobot_consumer_schema_sha256": served,
        "zebrobot_consumer_schema_matches_pin": (
            None if served is None else served == MZB_PIN["consumer_openapi_sha256"]
        ),
        "zebrobot_consumer_pin": dict(MZB_PIN),
    }


def resolve_subject_reference(
    reference: Mapping[str, Any], *, camera: str, fetch: Fetch = http_fetch
) -> ResolvedSubjectReference:
    """Fetch and verify a collected reference; pass a declared absence through."""

    ref = validate_subject_reference(reference, camera)
    if ref["status"] != "collected":
        return ResolvedSubjectReference(
            ref["status"], ref["reason"], None,
            {"kind": SOURCE_KIND, "status": ref["status"], "reason": ref["reason"],
             "subject_count": ref["subject_count"],
             "reference_schema_version": ref["schema_version"],
             **(_served_build(ref["zebrobot"]) if ref["zebrobot"] else {})},
        )
    zebrobot, dish = ref["zebrobot"], ref["dish"]
    base = zebrobot["base_url"].rstrip("/")
    dish_id = dish["dish_id"]
    status, snapshot = fetch(f"{base}/dishes/{quote(dish_id, safe='')}/citrus-snapshot")
    _require(
        status == 200 and isinstance(snapshot, Mapping),
        f"recorded dish {dish_id!r} is not served by Zebrobot at intake (HTTP {status})",
    )
    _require(snapshot.get("schema_version") == API_SCHEMA_VERSION, "zebrobot citrus-snapshot schema_version")
    _require(
        snapshot.get("dish_uuid") == dish["dish_uuid"],
        f"dish {dish_id!r} dish_uuid differs from the recording's: the dish was re-created",
    )
    revision = _revision(snapshot.get("revision"), "served dish revision")
    dish_changed = revision != dish["revision"]
    recorded_at = _recorded_at(zebrobot["queried_at_utc"], "reference")
    dof = _server_date(snapshot.get("dof"), "served dof")
    fish_items: list[dict[str, Any]] = []
    fish_changed: list[str] = []
    if ref["dish_fish_lookup"]["status"] == "complete":
        status, body = fetch(f"{base}/dishes/{quote(dish_id, safe='')}/fish")
        _require(status == 200 and isinstance(body, Mapping), f"dish {dish_id!r} fish list (HTTP {status})")
        served = {str(item.get("fish_id")): item for item in body.get("items") or []}
        for recorded in ref["dish_fish"]:
            item = served.get(recorded["fish_id"])
            if item is None or item.get("revision") != recorded["revision"]:
                fish_changed.append(recorded["fish_id"])
        fish_items = [dict(item) for item in ref["dish_fish"]]
    metadata = {
        "dish_id": dish_id,
        "dish_uuid": dish["dish_uuid"],
        "dish_revision_at_recording": dish["revision"],
        "dish_updated_at": dish["updated_at"],
        "dish_revision_at_intake": revision,
        "dish_changed_since_recording": dish_changed,
        "cross_id": snapshot.get("cross_id"),
        "genotype": snapshot.get("genotype"),
        "species": snapshot.get("species"),
        "sex": snapshot.get("sex"),
        "line_strain": snapshot.get("line_strain"),
        "parents": snapshot.get("parents"),
        "dof": snapshot["dof"],
        "dpf_at_recording": (recorded_at.date() - dof).days,
        "source_dish_population_count": snapshot.get("fish_count"),
        # Registered to the dish at record start; NOT the recorded subjects.
        "dish_fish_ids": [item["fish_id"] for item in fish_items],
        "dish_fish_changed_since_recording": fish_changed,
        "cross_fields_resolved_at": "intake",
        **(
            {"subject_count": ref["subject_count"]}
            if ref["subject_count"] is not None
            else {}
        ),
        "recorded_at_utc": zebrobot["queried_at_utc"],
    }
    source = {
        "kind": SOURCE_KIND,
        "status": "collected",
        "reference_schema_version": ref["schema_version"],
        **_served_build(zebrobot),
        # subject_count is operator-declared in Orange's reference, not Zebrobot's.
        "count_field": "subject_count",
        "zebrobot_base_url": base,
        "api_schema_version": API_SCHEMA_VERSION,
        "dish_uuid": dish["dish_uuid"],
        "dish_revision_at_recording": dish["revision"],
        "dish_revision_at_intake": revision,
        "dish_changed_since_recording": dish_changed,
        "recorded_at_utc": zebrobot["queried_at_utc"],
    }
    return ResolvedSubjectReference("collected", "", metadata, source)


__all__ = [
    "MZB_PIN",
    "REFERENCE_SCHEMA_ID",
    "ResolvedSubjectReference",
    "SubjectReferenceError",
    "ZebrobotUnavailable",
    "http_fetch",
    "resolve_subject_reference",
    "validate_subject_reference",
    "validate_subject_references",
]
