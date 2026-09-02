"""Small exact chaser-appearance dimension for grouped-statistics consumers."""

from __future__ import annotations

from collections import defaultdict
import math
from types import MappingProxyType
from typing import Any, Mapping

from fisheye.analytics_exports.validated_behavior_phase_c_contracts import (
    CHASER_OCCURRENCES_V2_SPEC,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

SCHEMA_ID = "palette.analytics.validated_behavior.chaser_appearance_dimension"
SCHEMA_VERSION = 1
METHOD_ID = "phase_c_chaser_occurrence_projection_v1"
NEUTRAL_MIXED_COLOR = "#555555"

APPEARANCE_COLUMNS = (
    "recording_id",
    "chaser_identity_code",
    "chaser_index",
    "chaser_identity",
    "behavior_role_code",
    "behavior_role",
    "stimulus_run_path",
    "source_protocol_sha256",
    "experimental_color_r",
    "experimental_color_g",
    "experimental_color_b",
    "experimental_color_a",
    "experimental_color_hex",
    "experimental_color_css",
    "contrast_outline_hex",
    "plotly_role_symbol",
    "matplotlib_role_marker",
    "appearance_schema_id",
    "appearance_schema_version",
    "appearance_policy_id",
    "appearance_projection_sha256",
    "occurrence_binding_sha256",
    "color_semantics",
    "role_semantics",
    "color_role_independence",
)
_APPEARANCE_EXTENSION_COLUMNS = {
    "behavior_role_code",
    "experimental_color_r",
    "experimental_color_g",
    "experimental_color_b",
    "experimental_color_a",
    "experimental_color_hex",
    "experimental_color_css",
    "contrast_outline_hex",
    "plotly_role_symbol",
    "matplotlib_role_marker",
    "appearance_schema_id",
    "appearance_schema_version",
    "appearance_policy_id",
    "appearance_projection_sha256",
    "occurrence_binding_sha256",
    "color_semantics",
    "role_semantics",
    "color_role_independence",
}

_DIGEST_FIELDS = (
    "source_protocol_sha256",
    "appearance_projection_sha256",
    "occurrence_binding_sha256",
)
_TEXT_FIELDS = tuple(
    name
    for name in APPEARANCE_COLUMNS
    if name
    not in {
        "chaser_identity_code",
        "chaser_index",
        "behavior_role_code",
        "experimental_color_r",
        "experimental_color_g",
        "experimental_color_b",
        "experimental_color_a",
        "appearance_schema_version",
        "color_role_independence",
    }
)


class ValidatedBehaviorAppearanceError(ValueError):
    """The exported appearance dimension is absent, stale, or ambiguous."""


def _fail(message: str) -> None:
    raise ValidatedBehaviorAppearanceError(message)


def _digest(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"Chaser appearance has an invalid digest: {field}")
    return value


def _hex_color(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 7
        or value[0] != "#"
        or any(character not in "0123456789abcdef" for character in value[1:])
    ):
        _fail(f"Chaser appearance has an invalid RGB color: {field}")
    return value


def validate_chaser_appearance_dimension(
    value: object,
    *,
    expected_export_manifest_sha256: str | None = None,
) -> Mapping[str, object]:
    """Validate one self-contained projection copied into statistics provenance."""

    if not isinstance(value, Mapping):
        _fail("Chaser appearance dimension must be one object")
    expected_fields = {
        "schema_id",
        "schema_version",
        "method_id",
        "status",
        "source_table",
        "join_fields",
        "color_semantics",
        "role_semantics",
        "color_role_independence",
        "source_query_identity",
        "rows",
        "record_sha256",
    }
    if set(value) != expected_fields:
        _fail("Chaser appearance dimension has an unexpected field set")
    if (
        value.get("schema_id") != SCHEMA_ID
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("method_id") != METHOD_ID
        or value.get("status") != "complete"
        or value.get("source_table") != "chaser_occurrences"
        or value.get("join_fields") != ["recording_id", "chaser_identity_code"]
        or value.get("color_semantics") != "experimental_protocol_rgba"
        or value.get("role_semantics") != "independent_marker_shape_and_text"
        or value.get("color_role_independence") is not True
    ):
        _fail("Chaser appearance dimension semantics are unsupported")
    body = {key: item for key, item in value.items() if key != "record_sha256"}
    if value.get("record_sha256") != canonical_json_sha256(body):
        _fail("Chaser appearance dimension digest is stale")

    query = value.get("source_query_identity")
    if (
        not isinstance(query, Mapping)
        or query.get("table_name") != "chaser_occurrences"
        or query.get("selected_columns") != list(APPEARANCE_COLUMNS)
    ):
        _fail("Chaser appearance source-query identity is invalid")
    for field in (
        "export_manifest_record_sha256",
        "export_plan_sha256",
        "table_contract_sha256",
        "analysis_unit_policy_sha256",
    ):
        _digest(query.get(field), field=field)
    if (
        expected_export_manifest_sha256 is not None
        and query.get("export_manifest_record_sha256")
        != _digest(
            expected_export_manifest_sha256,
            field="expected_export_manifest_sha256",
        )
    ):
        _fail("Chaser appearance belongs to another source export manifest")

    rows = value.get("rows")
    if not isinstance(rows, list) or not rows:
        _fail("Chaser appearance dimension has no exact occurrence rows")
    keys: list[tuple[str, int]] = []
    glyphs_by_role: dict[str, tuple[str, str]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != set(APPEARANCE_COLUMNS):
            _fail("Chaser appearance row has an unexpected field set")
        for field in _TEXT_FIELDS:
            text = row.get(field)
            if type(text) is not str or not text or text != text.strip():
                _fail(f"Chaser appearance text is invalid: {field}")
        for field in _DIGEST_FIELDS:
            _digest(row.get(field), field=field)
        for field in (
            "chaser_identity_code",
            "chaser_index",
            "behavior_role_code",
            "appearance_schema_version",
        ):
            if type(row.get(field)) is not int:
                _fail(f"Chaser appearance integer is invalid: {field}")
        if (
            int(row["chaser_identity_code"]) < 1
            or int(row["chaser_index"]) < 0
            or int(row["behavior_role_code"]) < 1
            or int(row["appearance_schema_version"]) < 1
        ):
            _fail("Chaser appearance integer lies outside its valid range")
        for field in (
            "experimental_color_r",
            "experimental_color_g",
            "experimental_color_b",
            "experimental_color_a",
        ):
            channel = row.get(field)
            if (
                type(channel) not in {int, float}
                or not math.isfinite(float(channel))
                or not 0.0 <= float(channel) <= 1.0
            ):
                _fail(f"Chaser appearance channel is invalid: {field}")
        _hex_color(row.get("experimental_color_hex"), field="experimental_color_hex")
        _hex_color(row.get("contrast_outline_hex"), field="contrast_outline_hex")
        if row.get("color_role_independence") is not True:
            _fail("Chaser appearance row does not preserve color/role independence")
        if row.get("color_semantics") != value.get("color_semantics") or row.get(
            "role_semantics"
        ) != value.get("role_semantics"):
            _fail("Chaser appearance row semantics differ from the dimension")
        key = (str(row["recording_id"]), int(row["chaser_identity_code"]))
        keys.append(key)
        role = str(row["behavior_role"])
        glyph = (
            str(row["plotly_role_symbol"]),
            str(row["matplotlib_role_marker"]),
        )
        previous = glyphs_by_role.setdefault(role, glyph)
        if previous != glyph:
            _fail("One behavior role resolves to multiple glyph definitions")
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        _fail("Chaser appearance occurrence keys are not strictly ordered and unique")
    return MappingProxyType(dict(value))


def build_chaser_appearance_dimension(dataset: Any) -> Mapping[str, object] | None:
    """Copy the small Phase-C dimension with its exact lazy-query identity."""

    if "chaser_occurrences" not in dataset.table_names:
        return None
    table = dataset.table("chaser_occurrences")
    known = {field.name for field in table.spec.contract.fields}
    present = _APPEARANCE_EXTENSION_COLUMNS & known
    if not present:
        return None
    missing = sorted(set(APPEARANCE_COLUMNS) - known)
    if missing:
        _fail(f"Chaser appearance contract is only partially installed: {missing}")
    if (
        table.spec.contract.payload_sha256
        != CHASER_OCCURRENCES_V2_SPEC.contract.payload_sha256
        or dict(table.spec.semantic_metadata)
        != dict(CHASER_OCCURRENCES_V2_SPEC.semantic_metadata)
    ):
        _fail("Chaser appearance table does not use the installed Phase-C contract")
    frame = (
        table.scan(columns=APPEARANCE_COLUMNS)
        .sort("recording_id", "chaser_identity_code")
        .collect()
    )
    rows = [json_attr_safe(row) for row in frame.to_dicts()]
    body: dict[str, object] = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "status": "complete",
        "source_table": "chaser_occurrences",
        "join_fields": ["recording_id", "chaser_identity_code"],
        "color_semantics": "experimental_protocol_rgba",
        "role_semantics": "independent_marker_shape_and_text",
        "color_role_independence": True,
        "source_query_identity": table.query_identity(
            columns=APPEARANCE_COLUMNS,
            predicate_description=(
                "all manifest-selected chaser occurrences; strictly ordered by "
                "recording_id and chaser_identity_code"
            ),
        ),
        "rows": rows,
    }
    result = {**body, "record_sha256": canonical_json_sha256(body)}
    return validate_chaser_appearance_dimension(
        result,
        expected_export_manifest_sha256=dataset.cache_identity,
    )


def behavior_role_styles(
    dimension: Mapping[str, object] | None,
    *,
    legacy_display_colors: Mapping[str, str],
) -> Mapping[str, Mapping[str, object]]:
    """Resolve truthful aggregate color policy plus independent role glyphs."""

    if dimension is None:
        return MappingProxyType(
            {
                role: MappingProxyType(
                    {
                        "aggregate_color_hex": color,
                        "aggregate_color_css": color,
                        "aggregate_color_policy": (
                            "legacy_display_palette_not_protocol_color"
                        ),
                        "experimental_color_hex_values": [],
                        "experimental_color_css_values": [],
                        "plotly_role_symbol": None,
                        "matplotlib_role_marker": None,
                        "color_role_independence": True,
                    }
                )
                for role, color in sorted(legacy_display_colors.items())
            }
        )
    resolved = validate_chaser_appearance_dimension(dimension)
    rows_by_role: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    raw_rows = resolved["rows"]
    assert isinstance(raw_rows, list)
    for row in raw_rows:
        assert isinstance(row, Mapping)
        rows_by_role[str(row["behavior_role"])].append(row)
    styles: dict[str, Mapping[str, object]] = {}
    for role, rows in sorted(rows_by_role.items()):
        colors = sorted({str(row["experimental_color_hex"]) for row in rows})
        css_colors = sorted({str(row["experimental_color_css"]) for row in rows})
        plotly_symbols = {str(row["plotly_role_symbol"]) for row in rows}
        matplotlib_markers = {str(row["matplotlib_role_marker"]) for row in rows}
        if len(plotly_symbols) != 1 or len(matplotlib_markers) != 1:
            _fail("One behavior role resolves to multiple glyph definitions")
        unique_color = len(colors) == 1 and len(css_colors) == 1
        styles[role] = MappingProxyType(
            {
                "aggregate_color_hex": (
                    colors[0] if unique_color else NEUTRAL_MIXED_COLOR
                ),
                "aggregate_color_css": (
                    css_colors[0] if unique_color else NEUTRAL_MIXED_COLOR
                ),
                "aggregate_color_policy": (
                    "unique_protocol_rgba_across_occurrences"
                    if unique_color
                    else "neutral_due_to_multiple_protocol_colors"
                ),
                "experimental_color_hex_values": colors,
                "experimental_color_css_values": css_colors,
                "plotly_role_symbol": next(iter(plotly_symbols)),
                "matplotlib_role_marker": next(iter(matplotlib_markers)),
                "color_role_independence": True,
            }
        )
    return MappingProxyType(styles)


__all__ = [
    "APPEARANCE_COLUMNS",
    "METHOD_ID",
    "NEUTRAL_MIXED_COLOR",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "ValidatedBehaviorAppearanceError",
    "behavior_role_styles",
    "build_chaser_appearance_dimension",
    "validate_chaser_appearance_dimension",
]
