"""Historical detect/pose metric rebinding to the registered best.pt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

import pandas as pd
import pytest
import torch

from fisheye.registry.db import Registry
from fisheye.training.training_run_shared import (
    BEST_CHECKPOINT_METRICS_SOURCE,
    ULTRALYTICS_BOX_METRIC_COLUMNS,
    ULTRALYTICS_POSE_METRIC_COLUMNS,
)
from fisheye.utils import backfill_training_best_metrics as tool

# Shaped like detect v004: best epoch 27, training ran to 37.
_DETECT_ROWS = [
    [26, 0.97369, 0.97369, 0.97833, 0.64270],
    [27, 0.97678, 0.97583, 0.98086, 0.71357],
    [37, 0.97530, 0.97440, 0.97827, 0.70829],
]
_POSE_ROWS = [
    [9, 0.95, 0.94, 0.97, 0.70, 0.90, 0.60],
    [10, 0.96, 0.95, 0.98, 0.72, 0.93, 0.66],
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_run(
    root: Path,
    name: str,
    *,
    columns: dict[str, str],
    rows: list[list[float]],
    best_index: int,
) -> tuple[Path, Path]:
    run_dir = root / name
    (run_dir / "weights").mkdir(parents=True)
    results = pd.DataFrame(rows, columns=["epoch", *columns.values()])
    results_csv = run_dir / "results.csv"
    results.to_csv(results_csv, index=False)
    best = results.iloc[best_index]
    torch.save(
        {"epoch": -1, "train_metrics": {column: float(best[column]) for column in columns.values()}},
        run_dir / "weights" / "best.pt",
    )
    return run_dir / "weights" / "best.pt", results_csv


def _register(
    registry_path: Path,
    run_id: str,
    task_type: str,
    model: Path,
    metrics: Path,
    final_metrics: dict,
) -> None:
    registry = Registry(registry_path)
    try:
        registry.record_training_run(
            run_id=run_id,
            set_id=f"{task_type}_set",
            task_type=task_type,
            config_path=None,
            manifest_path=None,
            model_path=model,
            metrics_path=metrics,
            model_sha256=_sha(model),
            metrics_sha256=_sha(metrics),
            status="success",
            final_metrics=final_metrics,
        )
    finally:
        registry.close()


def _row(registry_path: Path, table: str, run_id: str) -> dict:
    con = sqlite3.connect(registry_path)
    con.row_factory = sqlite3.Row
    try:
        row = con.execute(f"SELECT * FROM {table} WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row)
    finally:
        con.close()


@pytest.fixture
def populated(tmp_path: Path) -> Path:
    registry_path = tmp_path / "registry.sqlite"
    runs = tmp_path / "runs"

    model, metrics = _write_run(
        runs, "detect_v004", columns=ULTRALYTICS_BOX_METRIC_COLUMNS, rows=_DETECT_ROWS, best_index=1
    )
    _register(
        registry_path, "detect_v004", "detect", model, metrics,
        {"precision": 0.9753, "recall": 0.9744, "mAP50": 0.97827, "mAP50_95": 0.70829,
         "stage": "completed", "imgsz_h": 640},
    )
    # A promotion record in metadata must survive the backfill.
    con = sqlite3.connect(registry_path)
    with con:
        meta = json.loads(con.execute(
            "SELECT metadata_json FROM training_models WHERE run_id='detect_v004'").fetchone()[0])
        meta["canonical_package_promotion"] = {"schema_version": 1}
        con.execute("UPDATE training_models SET metadata_json=? WHERE run_id='detect_v004'",
                    (json.dumps(meta),))
    con.close()

    model, metrics = _write_run(
        runs, "pose_confirmed", columns=ULTRALYTICS_POSE_METRIC_COLUMNS, rows=_POSE_ROWS, best_index=1
    )
    _register(
        registry_path, "pose_confirmed", "pose", model, metrics,
        {"precision": 0.96, "recall": 0.95, "mAP50": 0.98, "mAP50_95": 0.72,
         "pose_mAP50": 0.93, "pose_mAP50_95": 0.66, "stage": "completed"},
    )

    model, metrics = _write_run(
        runs, "pose_empty", columns=ULTRALYTICS_POSE_METRIC_COLUMNS, rows=_POSE_ROWS, best_index=1
    )
    _register(registry_path, "pose_empty", "pose", model, metrics, {})

    model, metrics = _write_run(
        runs, "detect_bound", columns=ULTRALYTICS_BOX_METRIC_COLUMNS, rows=_DETECT_ROWS, best_index=1
    )
    _register(
        registry_path, "detect_bound", "detect", model, metrics,
        {"precision": 0.97678, "recall": 0.97583, "mAP50": 0.98086, "mAP50_95": 0.71357,
         "metrics_source": BEST_CHECKPOINT_METRICS_SOURCE},
    )

    model, metrics = _write_run(
        runs, "detect_tampered", columns=ULTRALYTICS_BOX_METRIC_COLUMNS, rows=_DETECT_ROWS, best_index=1
    )
    _register(registry_path, "detect_tampered", "detect", model, metrics, {"mAP50_95": 0.70829})
    torch.save({"epoch": -1, "train_metrics": {"metrics/mAP50-95(B)": 0.99}}, model)
    return registry_path


def test_plan_classifies_rows_and_skips_tampered_weights(populated: Path) -> None:
    planned, skipped = tool.plan(populated, applied_utc="2026-11-17T00:00:00+00:00")
    by_run = {item.run_id: item for item in planned}

    assert by_run["detect_v004"].classification == "corrected"
    assert by_run["detect_v004"].best_epoch == 27
    assert by_run["detect_v004"].last_epoch == 37
    assert by_run["pose_confirmed"].classification == "confirmed"
    assert by_run["pose_empty"].classification == "filled"
    assert by_run["detect_bound"].classification == "already_bound"
    assert by_run["detect_bound"].updates == []
    assert [entry["run_id"] for entry in skipped] == ["detect_tampered"]
    assert skipped[0]["reason"] == "model_sha256_mismatch"
    assert {update.table for update in by_run["detect_v004"].updates} == {
        "training_runs", "training_models"}


def test_apply_rebinds_metrics_and_keeps_previous_values(populated: Path) -> None:
    planned, _ = tool.plan(populated, applied_utc="2026-11-17T00:00:00+00:00")
    result = tool.apply_plan(populated, planned)
    assert result["row_updates"] == 6  # 3 runs x (training_runs + training_models)

    run = json.loads(_row(populated, "training_runs", "detect_v004")["final_metrics_json"])
    assert run["mAP50_95"] == pytest.approx(0.71357)
    assert run["best_epoch"] == 27
    assert run["last_epoch"] == 37
    assert run["last_epoch_metrics"]["mAP50_95"] == pytest.approx(0.70829)
    assert run["metrics_source"] == BEST_CHECKPOINT_METRICS_SOURCE
    assert run["metrics_backfill"]["classification"] == "corrected"
    assert run["metrics_backfill"]["previous_values"]["mAP50_95"] == pytest.approx(0.70829)
    assert run["stage"] == "completed" and run["imgsz_h"] == 640  # other keys preserved

    model_row = _row(populated, "training_models", "detect_v004")
    assert json.loads(model_row["final_metrics_json"])["mAP50_95"] == pytest.approx(0.71357)
    metadata = json.loads(model_row["metadata_json"])
    assert metadata["best_epoch"] == 27
    assert metadata["canonical_package_promotion"] == {"schema_version": 1}

    confirmed = json.loads(_row(populated, "training_runs", "pose_confirmed")["final_metrics_json"])
    assert confirmed["pose_mAP50_95"] == pytest.approx(0.66)
    assert confirmed["metrics_backfill"]["classification"] == "confirmed"

    filled = json.loads(_row(populated, "training_runs", "pose_empty")["final_metrics_json"])
    assert filled["pose_mAP50_95"] == pytest.approx(0.66)
    assert filled["metrics_backfill"]["previous_values"]["mAP50_95"] is None

    bound = json.loads(_row(populated, "training_runs", "detect_bound")["final_metrics_json"])
    assert "metrics_backfill" not in bound
    tampered = json.loads(_row(populated, "training_runs", "detect_tampered")["final_metrics_json"])
    assert tampered == {"mAP50_95": 0.70829}


def test_apply_is_idempotent(populated: Path) -> None:
    planned, _ = tool.plan(populated)
    tool.apply_plan(populated, planned)
    replanned, _ = tool.plan(populated)
    assert {item.classification for item in replanned} == {"already_bound"}
    assert all(item.updates == [] for item in replanned)


def test_apply_refuses_rows_changed_since_planning(populated: Path) -> None:
    planned, _ = tool.plan(populated)
    con = sqlite3.connect(populated)
    with con:
        con.execute(
            "UPDATE training_runs SET final_metrics_json='{\"edited\": true}' WHERE run_id='detect_v004'")
    con.close()
    with pytest.raises(tool.StaleBackfillPlanError, match="changed since planning"):
        tool.apply_plan(populated, planned)
    # The transaction rolled back: earlier rows in the plan were not written either.
    assert "metrics_backfill" not in json.loads(
        _row(populated, "training_runs", "pose_confirmed")["final_metrics_json"])


def test_dry_run_does_not_modify_registry(populated: Path, capsys: pytest.CaptureFixture) -> None:
    before = _sha(populated)
    assert tool.main(["--registry", str(populated)]) == 0
    assert _sha(populated) == before
    report = json.loads(capsys.readouterr().out)
    assert report["summary"]["planned"] == {
        "corrected": 1, "confirmed": 1, "filled": 1, "already_bound": 1}
    assert report["summary"]["skipped"] == 1


def test_apply_publishes_through_shadow_gateway_and_writes_receipt(
    populated: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LSB_JOBID", raising=False)
    receipts = tmp_path / "receipts"
    assert tool.main([
        "--registry", str(populated), "--apply",
        "--backup", str(tmp_path / "backup.sqlite"),
        "--receipt-dir", str(receipts),
    ]) == 0
    receipt_files = sorted(receipts.glob("receipt_*.json"))
    assert len(receipt_files) == 1 and sorted(receipts.glob("plan_*.json"))
    receipt = json.loads(receipt_files[0].read_text(encoding="utf-8"))
    assert receipt["schema_id"] == tool.RECEIPT_SCHEMA
    assert receipt["publication"]["mutation_result"]["row_updates"] == 6
    corrected = next(row for row in receipt["rows"] if row["run_id"] == "detect_v004")
    before = json.loads(corrected["updates"][0]["before_final_metrics_json"])
    assert before["mAP50_95"] == pytest.approx(0.70829)
    assert (tmp_path / "backup.sqlite").exists()
    run = json.loads(_row(populated, "training_runs", "detect_v004")["final_metrics_json"])
    assert run["mAP50_95"] == pytest.approx(0.71357)


def test_apply_refuses_inside_lsf_job(populated: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LSB_JOBID", "1")
    before = _sha(populated)
    assert tool.main([
        "--registry", str(populated), "--apply",
        "--backup", str(tmp_path / "b.sqlite"), "--receipt-dir", str(tmp_path / "r"),
    ]) == 2
    assert _sha(populated) == before
