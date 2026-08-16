from __future__ import annotations

from pathlib import Path

from scripts import check_observed_metadata_literals as mod


def test_rejects_observation_value_literal(tmp_path: Path) -> None:
    source = tmp_path / "bad.py"
    source.write_text(
        'attrs = {"container_color_range_observed": "tv"}\n',
        encoding="utf-8",
    )

    assert mod.main([str(source)]) == 1


def test_accepts_runtime_observation_and_schema_type_literal(tmp_path: Path) -> None:
    source = tmp_path / "good.py"
    source.write_text(
        "observed = probe()\n"
        'attrs = {"container_color_range_observed": observed}\n'
        'schema = {"sample_observed": "bool"}\n',
        encoding="utf-8",
    )

    assert mod.main([str(source)]) == 0
