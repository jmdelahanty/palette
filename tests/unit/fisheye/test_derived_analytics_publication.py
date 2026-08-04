from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
import threading

import pyarrow.parquet as pq
import pytest

from fisheye.analytics_exports.arrow_contract_core import (
    ArrowTableContract,
    contract_envelope,
    exact_schema,
    field,
)
from fisheye.analytics_exports.derived_publication import (
    derived_manifest_selected_parts,
    publish_derived_table_generation,
)


TABLE = "example_metrics"
ENVELOPE_SCHEMA_ID = "palette.test.derived_arrow_contracts"
CONTRACT = ArrowTableContract(
    table_name=TABLE,
    fields=(
        field("analysis_run_id", "string"),
        field("recording_id", "string"),
        field("value", "float64", nullable=True),
    ),
    schema_version=2,
    schema_namespace="palette.test.derived_arrow_table",
    primary_key=("analysis_run_id", "recording_id"),
)
CONTRACTS = {TABLE: CONTRACT}


def _envelope() -> dict[str, object]:
    return contract_envelope(
        (TABLE,),
        known_table_names=(TABLE,),
        contracts=CONTRACTS,
        schema_id=ENVELOPE_SCHEMA_ID,
        schema_version=1,
    )


def _publish(
    root: Path,
    rows: list[dict[str, object]],
    *,
    generation_id: str = "generation-a",
) -> dict[str, object]:
    return publish_derived_table_generation(
        output_root=root,
        analysis_run_id="run-a",
        rows_by_table={TABLE: rows},
        table_names=(TABLE,),
        contracts=CONTRACTS,
        arrow_contract_envelope=_envelope(),
        arrow_envelope_schema_id=ENVELOPE_SCHEMA_ID,
        arrow_envelope_schema_version=1,
        manifest_fields={
            "schema_id": "palette.test.derived",
            "schema_version": 2,
        },
        footer_metadata={
            b"palette.schema_id": b"palette.test.derived",
            b"palette.schema_version": b"2",
            b"palette.table_name": TABLE.encode(),
        },
        generation_id=generation_id,
    )


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [
            {
                "analysis_run_id": "run-a",
                "recording_id": "recording-a",
                "value": 1.25,
            }
        ],
    ],
)
def test_exact_generation_publishes_manifest_selected_zero_or_nonzero_part(
    tmp_path: Path,
    rows: list[dict[str, object]],
) -> None:
    result = _publish(tmp_path, rows)

    manifest_path = Path(str(result["manifest_path"]))
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    parts = derived_manifest_selected_parts(
        tmp_path,
        payload,
        TABLE,
        table_names=(TABLE,),
    )

    assert len(parts) == 1
    parquet_file = pq.ParquetFile(parts[0])
    assert parquet_file.metadata.num_rows == len(rows)
    assert parquet_file.schema_arrow.remove_metadata() == exact_schema(
        CONTRACT,
        metadata={},
    ).remove_metadata()
    assert not list((tmp_path / "v2" / ".staging").glob("*"))


def test_selected_part_rejects_content_tampering(tmp_path: Path) -> None:
    result = _publish(
        tmp_path,
        [
            {
                "analysis_run_id": "run-a",
                "recording_id": "recording-a",
                "value": 1.25,
            }
        ],
    )
    manifest_path = Path(str(result["manifest_path"]))
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    parts = derived_manifest_selected_parts(
        tmp_path,
        payload,
        TABLE,
        table_names=(TABLE,),
    )
    parts[0].write_bytes(parts[0].read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="size mismatch"):
        derived_manifest_selected_parts(
            tmp_path,
            payload,
            TABLE,
            table_names=(TABLE,),
        )


def test_selected_part_rejects_in_root_symlink_alias(tmp_path: Path) -> None:
    result = _publish(
        tmp_path,
        [
            {
                "analysis_run_id": "run-a",
                "recording_id": "recording-a",
                "value": 1.25,
            }
        ],
    )
    manifest_path = Path(str(result["manifest_path"]))
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    parts = derived_manifest_selected_parts(
        tmp_path,
        payload,
        TABLE,
        table_names=(TABLE,),
    )
    real_part = parts[0].with_name("backing.parquet")
    parts[0].rename(real_part)
    parts[0].symlink_to(real_part.name)

    with pytest.raises(ValueError, match="symbolic-link alias"):
        derived_manifest_selected_parts(
            tmp_path,
            payload,
            TABLE,
            table_names=(TABLE,),
        )


def test_failed_staged_validation_leaves_no_visible_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fisheye.analytics_exports.derived_publication as publication

    def reject(*args: object, **kwargs: object) -> None:
        raise ValueError("injected staged validation failure")

    monkeypatch.setattr(publication, "_validate_staged_generation", reject)
    with pytest.raises(ValueError, match="injected staged validation failure"):
        _publish(
            tmp_path,
            [
                {
                    "analysis_run_id": "run-a",
                    "recording_id": "recording-a",
                    "value": 1.25,
                }
            ],
        )

    assert not (tmp_path / "v2" / "manifests").exists()
    assert not list((tmp_path / "v2" / ".generations").rglob("*.parquet"))
    assert not list((tmp_path / "v2" / ".staging").glob("*"))


def test_lost_manifest_race_removes_unselected_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fisheye.analytics_exports.publication as publication

    @contextmanager
    def competing_manifest(
        publication_root: Path,
        manifest_path: Path,
        *,
        lock_directory: Path,
    ):
        del publication_root, lock_directory
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text('{"winner":"other-writer"}\n', encoding="utf-8")
        yield

    monkeypatch.setattr(
        publication,
        "immutable_manifest_commit_lock",
        competing_manifest,
    )
    with pytest.raises(RuntimeError, match="manifest changed"):
        _publish(
            tmp_path,
            [
                {
                    "analysis_run_id": "run-a",
                    "recording_id": "recording-a",
                    "value": 1.25,
                }
            ],
        )

    manifest_path = tmp_path / "v2" / "manifests" / "analysis_run_id=run-a.json"
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == {
        "winner": "other-writer"
    }
    assert not list((tmp_path / "v2" / ".generations").rglob("*.parquet"))
    assert not list((tmp_path / "v2" / ".staging").glob("*"))


def test_concurrent_commit_cannot_select_another_writers_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fisheye.analytics_exports.publication as publication

    writer_a_ready = threading.Event()
    writer_b_ready = threading.Event()
    writer_a_committed = threading.Event()

    @contextmanager
    def ordered_lock(
        publication_root: Path,
        manifest_path: Path,
        *,
        lock_directory: Path,
    ):
        del publication_root, manifest_path, lock_directory
        if threading.current_thread().name == "writer-a":
            writer_a_ready.set()
            assert writer_b_ready.wait(timeout=5)
            try:
                yield
            finally:
                writer_a_committed.set()
            return
        writer_b_ready.set()
        assert writer_a_ready.wait(timeout=5)
        assert writer_a_committed.wait(timeout=5)
        yield

    monkeypatch.setattr(publication, "immutable_manifest_commit_lock", ordered_lock)
    manifest_path = tmp_path / "manifests" / "run.json"
    errors: dict[str, BaseException] = {}

    def commit(label: str) -> None:
        stage = tmp_path / f"stage-{label}"
        generation = tmp_path / f"generation-{label}"
        stage.mkdir()
        try:
            publication.commit_validated_immutable_generation(
                tmp_path,
                stage,
                generation,
                manifest_path,
                {"generation_id": label},
                baseline_manifest_identity=None,
                lock_directory=tmp_path / "locks",
                validate_staging=lambda: None,
            )
        except BaseException as exc:  # preserve the worker failure for assertion
            errors[label] = exc

    writer_a = threading.Thread(target=commit, args=("a",), name="writer-a")
    writer_b = threading.Thread(target=commit, args=("b",), name="writer-b")
    writer_a.start()
    writer_b.start()
    writer_a.join(timeout=10)
    writer_b.join(timeout=10)

    assert not writer_a.is_alive()
    assert not writer_b.is_alive()
    assert "a" not in errors
    assert isinstance(errors.get("b"), RuntimeError)
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == {
        "generation_id": "a"
    }
    assert (tmp_path / "generation-a").is_dir()
    assert not (tmp_path / "generation-b").exists()
    assert not list(manifest_path.parent.glob(".*.tmp"))
