"""Canonical subject fields: one name, type and meaning per fact, whatever the source.

Design: ``docs/design/2026-10-06-canonical-subject-fields/README.md``.

Every subject source publishes a ``palette.subject_metadata`` record whose
``subject_metadata`` mapping is shaped by that source (H5 attributes, an
Orange MetaZebrobot reference resolved at intake, or a manual assertion).
This module owns the one translation from each source shape into the
canonical fields; consumers read only the canonical fields. A value that
cannot be translated is left out and its reason is recorded under
``unresolved``; nothing is coerced or guessed.
"""

from __future__ import annotations

from datetime import date, datetime
import json
import re
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

SUBJECT_TYPES = ("individual", "dish_group")
# Fertilization dates are lab calendar days, so the recording day must be too:
# an evening recording is already "tomorrow" in UTC (Citrus counts locally).
# Owner's statement: metazebrobot docs/zebrobot_snapshot.md "Dates and times"
# (commit 49cc930): calendar days are America/New_York; updated_at/created_at
# are UTC without an offset; *_utc fields carry an explicit Z.
LAB_TIMEZONE = ZoneInfo("America/New_York")
LOOKUP_STATUSES = ("collected", "not_collected", "lookup_failed")
# The manual CLI's "group" means a dish group (decided 2026-10-06).
_SUBJECT_TYPE_ALIASES = {"individual": "individual", "dish_group": "dish_group", "group": "dish_group"}
_PARENT_WITH_SEX = re.compile(r"^(?P<identifier>.*?)\s*\[(?P<sex>[^\]]*)\]$")
_YYYYMMDD = re.compile(r"^\d{8}$")
# The ``source`` values the manual writers stamp into their mappings:
# set_recording_subject_metadata, backfill_subject_context, migrate_count_only_subject_context.
_MANUAL_SOURCES = frozenset(
    {"recording_manifest_manual_assertion", "manual_subject_context_backfill",
     "manual_operator_assertion"}
)

TEXT_FIELDS = ("species", "sex", "genotype", "line_strain", "cross_id", "dish_id", "dish_uuid",
               "dish_updated_at", "mzb_fish_id", "mzb_fish_updated_at")


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    text = str(value).strip()
    return text or None


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    text = _text(value)
    if text is not None and re.fullmatch(r"-?\d+", text):
        return int(text)
    return None


def parse_parents(value: Any) -> list[dict[str, str | None]] | None:
    """Parents as ``[{identifier, sex}]`` from a list, JSON text or ``"ID [sex]; ID [sex]"``."""

    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", "replace")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        if isinstance(parsed, list):
            return parse_parents(parsed)
        value = [part for part in text.split(";") if part.strip()]
    if not isinstance(value, (list, tuple)):
        return None
    parents = []
    for item in value:
        if isinstance(item, Mapping):
            identifier, sex = _text(item.get("identifier")), _text(item.get("sex"))
        else:
            text = _text(item)
            if text is None:
                continue
            match = _PARENT_WITH_SEX.match(text)
            identifier, sex = (match["identifier"].strip(), _text(match["sex"])) if match else (text, None)
        if identifier:
            parents.append({"identifier": identifier, "sex": sex})
    return parents


def _fertilization_date(value: Any) -> date | None:
    text = _text(value)
    if text is None or not _YYYYMMDD.match(text):
        return None
    try:
        return date(int(text[:4]), int(text[4:6]), int(text[6:]))
    except ValueError:
        return None


def lab_recording_date(moment: datetime) -> date:
    """The lab calendar day of an instant (naive instants are taken as lab-local)."""

    return moment.astimezone(LAB_TIMEZONE).date() if moment.tzinfo is not None else moment.date()


def _recording_date(value: Any) -> date | None:
    text = _text(value)
    if text is None:
        return None
    try:
        moment = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError:
        return None
    return lab_recording_date(moment)


class _Builder:
    def __init__(self) -> None:
        self.fields: dict[str, Any] = {}
        self.unresolved: dict[str, str] = {}

    def text(self, name: str, *values: Any) -> None:
        for value in values:
            text = _text(value)
            if text is not None:
                self.fields[name] = text
                return

    def integer(self, name: str, *values: Any) -> None:
        for value in values:
            if value is None or _text(value) is None:
                continue
            number = _int(value)
            if number is None:
                self.unresolved[name] = f"not an integer: {value!r}"
            else:
                self.fields[name] = number
                self.unresolved.pop(name, None)
            return

    def status(self, name: str, status: Any, reason: Any) -> None:
        status = _text(status)
        if status is None:
            return
        if status not in LOOKUP_STATUSES:
            self.unresolved[name] = f"unknown status: {status!r}"
            return
        self.fields[name] = {"status": status, "reason": _text(reason) or ""}

    def subject_type(self, value: Any) -> None:
        text = _text(value)
        if text is None:
            return
        canonical = _SUBJECT_TYPE_ALIASES.get(text)
        if canonical is None:
            self.unresolved["subject_type"] = f"unknown subject_type: {text!r}"
        else:
            self.fields["subject_type"] = canonical

    def age(self, fertilization: Any, recorded_at: Any, *declared: Any) -> None:
        """``dpf_at_acquisition`` = recording date - fertilization date when both are known."""

        dof = _fertilization_date(fertilization)
        if _text(fertilization) is not None:
            if dof is None:
                self.unresolved["date_of_fertilization"] = f"not YYYYMMDD: {fertilization!r}"
            else:
                self.fields["date_of_fertilization"] = dof.strftime("%Y%m%d")
        recorded = _recording_date(recorded_at)
        if dof is not None and recorded is not None:
            self.fields["dpf_at_acquisition"] = (recorded - dof).days
            return
        self.integer("dpf_at_acquisition", *declared)

    def finish(self, subject_ids: Sequence[str]) -> dict[str, Any]:
        if subject_ids:
            self.fields["subject_ids"] = [str(value) for value in subject_ids]
        count = self.fields.get("subject_count")
        if "subject_type" not in self.fields and "subject_type" not in self.unresolved and count:
            self.fields["subject_type"] = "individual" if count == 1 else "dish_group"
        result = {key: self.fields[key] for key in sorted(self.fields)}
        if self.unresolved:
            result["unresolved"] = dict(sorted(self.unresolved.items()))
        return result


def from_h5_attributes(attrs: Mapping[str, Any], subject_ids: Sequence[str] = ()) -> dict[str, Any]:
    """Legacy ``/subject_metadata`` and unified ``/metadata/subject`` attributes."""

    out = _Builder()
    out.subject_type(attrs.get("subject_type"))
    for name in TEXT_FIELDS:
        out.text(name, attrs.get(name))
    out.text("cross_id", attrs.get("cross_id"), attrs.get("source_cross_id"))
    out.text("dish_id", attrs.get("dish_id"), attrs.get("source_dish_id"))
    for name in ("subject_count", "dish_revision", "mzb_fish_revision"):
        out.integer(name, attrs.get(name))
    out.integer("source_dish_population_count",
                attrs.get("source_dish_population_count"), attrs.get("fish_count"))
    parents = parse_parents(attrs.get("parents"))
    if parents is not None:
        out.fields["parents"] = parents
    out.age(attrs.get("date_of_fertilization"), attrs.get("queried_at_utc"),
            attrs.get("dpf_at_acquisition"), attrs.get("days_post_fertilization"))
    out.status("subject_lookup", attrs.get("subject_lookup_status"), attrs.get("subject_lookup_reason"))
    out.status("fish_reference", attrs.get("fish_reference_status"), attrs.get("fish_reference_reason"))
    return out.finish(subject_ids)


def from_orange_reference(metadata: Mapping[str, Any], subject_ids: Sequence[str] = ()) -> dict[str, Any]:
    """``zebrobot_subject_reference.resolve_subject_reference(...).metadata``."""

    out = _Builder()
    for name in ("species", "sex", "genotype", "line_strain", "cross_id", "dish_id", "dish_uuid",
                 "dish_updated_at"):
        out.text(name, metadata.get(name))
    out.integer("subject_count", metadata.get("subject_count"))
    out.integer("dish_revision", metadata.get("dish_revision_at_recording"))
    out.integer("source_dish_population_count", metadata.get("source_dish_population_count"))
    parents = parse_parents(metadata.get("parents"))
    if parents is not None:
        out.fields["parents"] = parents
    out.age(metadata.get("dof"), metadata.get("recorded_at_utc"), metadata.get("dpf_at_recording"))
    out.fields["subject_lookup"] = {"status": "collected", "reason": ""}
    observations = {
        name: metadata[name]
        for name in ("dish_revision_at_intake", "dish_changed_since_recording",
                     "dish_fish_changed_since_recording", "cross_fields_resolved_at")
        if name in metadata
    }
    if observations:
        out.fields["intake_observations"] = observations
    return out.finish(subject_ids)


def from_manual_assertion(metadata: Mapping[str, Any], subject_ids: Sequence[str] = ()) -> dict[str, Any]:
    """Manual assertions and backfills (``set_recording_subject_metadata`` and friends)."""

    out = _Builder()
    out.subject_type(metadata.get("subject_type"))
    for name in ("species", "sex", "genotype", "line_strain", "cross_id", "dish_id"):
        out.text(name, metadata.get(name))
    # e.g. "recording_local_placeholder": ids that name no real animal.
    out.text("identity_scope", metadata.get("identity_scope"))
    out.integer("subject_count", metadata.get("subject_count"))
    out.age(metadata.get("date_of_fertilization"), None,
            metadata.get("dpf_at_acquisition"), metadata.get("days_post_fertilization"))
    return out.finish(subject_ids)


def from_declared_absence(metadata: Mapping[str, Any], subject_ids: Sequence[str] = ()) -> dict[str, Any]:
    """A session that declares no subject: only the lookup statuses (no dish)."""

    out = _Builder()
    out.status("subject_lookup", metadata.get("subject_lookup_status"), metadata.get("subject_lookup_reason"))
    out.status("fish_reference", metadata.get("fish_reference_status"), metadata.get("fish_reference_reason"))
    return out.finish(subject_ids)


def from_legacy_zebrobot_snapshot(
    snapshot: Mapping[str, Any], subject_ids: Sequence[str] = ()
) -> dict[str, Any]:
    """The pre-``subject_metadata`` ``analysis_metadata`` ``zebrobot_snapshot`` attr.

    Dish and cross fields are nested (``dish``/``cross``); the dish's
    ``cross_id`` wins, the cross's ``line_strain`` and ``parents`` win.
    """

    dish = snapshot.get("dish") if isinstance(snapshot.get("dish"), Mapping) else {}
    cross = snapshot.get("cross") if isinstance(snapshot.get("cross"), Mapping) else {}
    flat = {
        key: value for key, value in snapshot.items() if not isinstance(value, Mapping)
    }
    flat.update(cross)
    flat.update(dish)
    for name in ("line_strain", "parents"):
        if cross.get(name) is not None:
            flat[name] = cross[name]
    if dish.get("cross_id") is None and cross.get("cross_id") is not None:
        flat["cross_id"] = cross["cross_id"]
    return from_h5_attributes(flat, subject_ids)


def source_kind(metadata: Mapping[str, Any]) -> str:
    """Which translator a stored v1 mapping needs (v1 records carry no source tag)."""

    if "dish_revision_at_recording" in metadata and "recorded_at_utc" in metadata:
        return "orange_reference"
    if _text(metadata.get("source")) in _MANUAL_SOURCES or "identity_scope" in metadata:
        return "manual_assertion"
    return "h5_attributes"


TRANSLATORS: dict[str, Callable[..., dict[str, Any]]] = {
    "h5_attributes": from_h5_attributes,
    "orange_reference": from_orange_reference,
    "manual_assertion": from_manual_assertion,
    "declared_absence": from_declared_absence,
    "legacy_zebrobot_snapshot": from_legacy_zebrobot_snapshot,
}


def canonical_subject_fields(metadata: Mapping[str, Any], subject_ids: Sequence[str] = ()) -> dict[str, Any]:
    return TRANSLATORS[source_kind(metadata)](metadata, subject_ids)


__all__ = [
    "LOOKUP_STATUSES",
    "SUBJECT_TYPES",
    "TRANSLATORS",
    "canonical_subject_fields",
    "from_declared_absence",
    "from_h5_attributes",
    "lab_recording_date",
    "from_legacy_zebrobot_snapshot",
    "from_manual_assertion",
    "from_orange_reference",
    "parse_parents",
    "source_kind",
]
