from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys

from fisheye.utils.registry_integrity import main


def _create_registry(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE marker (value TEXT NOT NULL);")
        connection.execute("INSERT INTO marker VALUES ('present');")
        connection.commit()


def test_registry_integrity_records_exact_palette_sqlite_runtime(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry.sqlite"
    result_json = tmp_path / "validation.json"
    _create_registry(registry)

    status = main(
        [
            "--registry",
            str(registry),
            "--result-json",
            str(result_json),
        ]
    )

    assert status == 0
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    assert payload["schema_id"] == "palette.registry_integrity_validation"
    assert payload["status"] == "complete"
    validation = payload["validation"]
    assert validation["integrity_check"] == "ok"
    assert validation["foreign_key_issue_count"] == 0
    assert validation["sqlite_runtime_version"] == sqlite3.sqlite_version
    assert validation["sqlite_python_module_version"] == sqlite3.version
    assert validation["python_executable"] == sys.executable
    assert validation["validation_backend"] == "python_stdlib_sqlite3"
