from __future__ import annotations

from pathlib import Path

from fisheye.registry.db import Registry
from fisheye.registry.migrations import MIGRATION_METHODS
from scripts import generate_registry_schema_reference as mod


def test_schema_reference_is_deterministic_and_version_bound() -> None:
    registry = Registry(Path(":memory:"))
    try:
        first = mod._generate_markdown(registry)
        second = mod._generate_markdown(registry)
    finally:
        registry.close()

    assert first == second
    assert "Generated at:" not in first
    assert f"> Registry schema migration: `{MIGRATION_METHODS[-1][0]}`" in first


def test_view_column_reference_ignores_sqlite_inferred_types() -> None:
    common = {"name": "computed_value", "notnull": 0, "pk": 0, "dflt_value": None}
    sqlite_345 = [{**common, "type": ""}]
    sqlite_352 = [{**common, "type": "TEXT"}]

    expected = [
        "| Column | Type | Nullable | PK | Default |",
        "|---|---|---|---|---|",
        "| `computed_value` | `` | yes | no | `` |",
    ]
    assert mod._render_view_column_table(sqlite_345) == expected
    assert mod._render_view_column_table(sqlite_352) == expected


def test_check_mode_is_read_only_and_detects_drift(tmp_path: Path) -> None:
    output = tmp_path / "registry_schema_reference.md"

    assert mod.main(["--output", str(output)]) == 0
    original = output.read_text(encoding="utf-8")
    assert mod.main(["--output", str(output), "--check"]) == 0
    assert output.read_text(encoding="utf-8") == original

    output.write_text(original + "drift\n", encoding="utf-8")
    assert mod.main(["--output", str(output), "--check"]) == 1
    assert output.read_text(encoding="utf-8") == original + "drift\n"
