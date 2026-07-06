"""Backfill recording subject context into Zarr metadata and the registry.

This utility is intentionally narrow. It repairs archives where the biological
subject context is known after import by writing the Zarr metadata mirror and
the normalized registry subject rows in one apply-gated operation.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from fisheye.shared.json_safety import write_json_atomic, write_jsonl_atomic


BACKFILL_SOURCE = "manual_subject_context_backfill"
DPF_KEYS = ("dpf_at_acquisition", "days_post_fertilization")
FERTILIZATION_DATE_KEYS = ("date_of_fertilization", "fertilization_date", "fertilized_on")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _node_attrs_path(node_dir: Path) -> Path | None:
    attrs_path = node_dir / "zarr.json"
    return attrs_path if attrs_path.is_file() else None


def _read_node_attrs(node_dir: Path) -> dict[str, Any]:
    attrs_path = _node_attrs_path(node_dir)
    if attrs_path is None:
        return {}
    payload = _read_json(attrs_path) or {}
    attrs = payload.get("attributes")
    return dict(attrs) if isinstance(attrs, Mapping) else {}


def _write_node_attrs(node_dir: Path, attrs: Mapping[str, Any]) -> None:
    attrs_path = _node_attrs_path(node_dir)
    if attrs_path is None:
        raise FileNotFoundError(f"missing Zarr v3 metadata: {node_dir / 'zarr.json'}")
    payload = _read_json(attrs_path) or {}
    payload["attributes"] = dict(attrs)
    write_json_atomic(attrs_path, payload)


def _ensure_group_node(node_dir: Path) -> None:
    attrs_path = _node_attrs_path(node_dir)
    if attrs_path is not None:
        return
    node_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        node_dir / "zarr.json",
        {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {},
        },
    )


def _decode_subject_metadata(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return dict(payload) if isinstance(payload, Mapping) else None
    return None


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    text = str(value).strip()
    return text or None


def _recording_dir_for_zarr(zarr_path: Path) -> Path:
    if zarr_path.parent.name == "zarr":
        return zarr_path.parent.parent
    return zarr_path.parent


def _first_int(payload: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    for key in keys:
        value = _as_int(payload.get(key))
        if value is not None:
            return value
    return None


def _first_text(payload: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = _as_text(payload.get(key))
        if value:
            return value
    return None


def _read_h5_subject_metadata(recording_dir: Path) -> dict[str, Any] | None:
    raw_dir = recording_dir / "raw"
    h5_paths = sorted(path for path in raw_dir.glob("*.h5") if path.is_file())
    if len(h5_paths) != 1:
        return None
    try:
        import h5py
    except Exception:
        return None
    try:
        with h5py.File(h5_paths[0], "r") as handle:
            if "subject_metadata" not in handle:
                return None
            attrs: dict[str, Any] = {}
            for key, value in handle["subject_metadata"].attrs.items():
                if hasattr(value, "item"):
                    try:
                        value = value.item()
                    except Exception:
                        pass
                attrs[str(key)] = _as_text(value) if isinstance(value, bytes) else value
            return attrs
    except Exception:
        return None


def _derive_subject_context_from_metadata(target: DatasetTarget) -> dict[str, Any]:
    """Return DPF/fertilization metadata that already exists near the recording."""

    candidates: list[tuple[str, Mapping[str, Any]]] = []
    analysis_attrs = _read_node_attrs(target.zarr_path / "analysis_metadata")
    zarr_subject = _decode_subject_metadata(analysis_attrs.get("subject_metadata"))
    if zarr_subject:
        candidates.append(("zarr_subject_metadata", zarr_subject))

    recording_dir = _recording_dir_for_zarr(target.zarr_path)
    manifest = _read_json(recording_dir / "recording_manifest.json")
    if manifest:
        candidates.append(("recording_manifest", manifest))
        manifest_subject = manifest.get("subject_metadata")
        if isinstance(manifest_subject, Mapping):
            candidates.append(("recording_manifest.subject_metadata", manifest_subject))

    h5_subject = _read_h5_subject_metadata(recording_dir)
    if h5_subject:
        candidates.append(("raw_h5.subject_metadata", h5_subject))

    derived: dict[str, Any] = {
        "dpf_at_acquisition": None,
        "date_of_fertilization": None,
        "dpf_source": None,
        "date_of_fertilization_source": None,
    }
    for source, payload in candidates:
        if derived["dpf_at_acquisition"] is None:
            dpf = _first_int(payload, DPF_KEYS)
            if dpf is not None:
                derived["dpf_at_acquisition"] = dpf
                derived["dpf_source"] = source
        if derived["date_of_fertilization"] is None:
            date_value = _first_text(payload, FERTILIZATION_DATE_KEYS)
            if date_value:
                derived["date_of_fertilization"] = date_value
                derived["date_of_fertilization_source"] = source
        if derived["dpf_at_acquisition"] is not None and derived["date_of_fertilization"] is not None:
            break
    return derived


def _subject_type(subject_count: int) -> str:
    return "individual" if subject_count == 1 else "group"


def _subject_id_for_recording(recording_id: str, index: int, template: str) -> str:
    return template.format(recording_id=recording_id, index=index, index1=index + 1)


def _subject_ids(recording_id: str, subject_count: int, template: str) -> list[str]:
    return [_subject_id_for_recording(recording_id, idx, template) for idx in range(subject_count)]


def _subject_metadata_payload(
    *,
    recording_id: str,
    species: str,
    subject_count: int,
    subject_ids: Sequence[str],
    dpf_at_acquisition: int | None,
    date_of_fertilization: str | None,
    created_at_utc: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source": BACKFILL_SOURCE,
        "status": "manual_backfill",
        "recording_id": recording_id,
        "species": species,
        "subject_count": subject_count,
        "subject_type": _subject_type(subject_count),
        "subject_ids": list(subject_ids),
        "identity_scope": "recording_local_placeholder",
        "backfilled_at_utc": created_at_utc,
    }
    if subject_count == 1:
        payload["subject_id"] = subject_ids[0]
    if dpf_at_acquisition is not None:
        payload["dpf_at_acquisition"] = dpf_at_acquisition
        payload["days_post_fertilization"] = dpf_at_acquisition
    if date_of_fertilization:
        payload["date_of_fertilization"] = date_of_fertilization
    return payload


def _experiment_setup_payload(
    *,
    subject_count: int,
    created_at_utc: str,
) -> dict[str, Any]:
    return {
        "num_dishes": 1,
        "fish_per_dish": subject_count,
        "total_expected_fish": subject_count,
        "setup_type": "single_dish",
        "source": BACKFILL_SOURCE,
        "configured_at": created_at_utc,
        "subject_count": subject_count,
        "subject_type": _subject_type(subject_count),
    }


def _compatible_subject_metadata(existing: Mapping[str, Any], *, species: str, subject_count: int) -> bool:
    existing_species = str(existing.get("species") or "").strip()
    if existing_species and existing_species != species:
        return False
    existing_count = existing.get("subject_count")
    if existing_count not in (None, ""):
        try:
            if int(existing_count) != subject_count:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _compatible_optional_int(existing: Mapping[str, Any], keys: Sequence[str], expected: int | None) -> bool:
    if expected is None:
        return True
    for key in keys:
        value = existing.get(key)
        if value in (None, ""):
            continue
        try:
            return int(value) == expected
        except (TypeError, ValueError):
            return False
    return True


def _compatible_optional_text(existing: Mapping[str, Any], key: str, expected: str | None) -> bool:
    if not expected:
        return True
    value = str(existing.get(key) or "").strip()
    return not value or value == expected


def _merge_subject_metadata(
    existing: Mapping[str, Any] | None,
    backfill_payload: Mapping[str, Any],
    *,
    overwrite: bool,
) -> tuple[dict[str, Any], bool, str]:
    if existing is None:
        return dict(backfill_payload), True, "created"
    if not overwrite and not _compatible_subject_metadata(
        existing,
        species=str(backfill_payload["species"]),
        subject_count=int(backfill_payload["subject_count"]),
    ):
        return dict(existing), False, "conflict"
    merged = dict(backfill_payload if overwrite else existing)
    if not overwrite:
        for key, value in backfill_payload.items():
            if merged.get(key) in (None, "", []):
                merged[key] = value
    changed = dict(existing) != merged
    return merged, changed, "updated" if changed else "already_present"


def _merge_experiment_setup(
    existing: Mapping[str, Any] | None,
    backfill_payload: Mapping[str, Any],
    *,
    overwrite: bool,
) -> tuple[dict[str, Any], bool, str]:
    if existing is None:
        return dict(backfill_payload), True, "created"
    existing_count = existing.get("subject_count") or existing.get("total_expected_fish")
    if existing_count not in (None, ""):
        try:
            if int(existing_count) != int(backfill_payload["subject_count"]) and not overwrite:
                return dict(existing), False, "conflict"
        except (TypeError, ValueError):
            if not overwrite:
                return dict(existing), False, "conflict"
    merged = dict(backfill_payload if overwrite else existing)
    if not overwrite:
        for key, value in backfill_payload.items():
            if merged.get(key) in (None, ""):
                merged[key] = value
    changed = dict(existing) != merged
    return merged, changed, "updated" if changed else "already_present"


@dataclass(frozen=True)
class DatasetTarget:
    dataset_id: str
    recording_id: str
    zarr_path: Path
    zarr_use: str | None
    status: str | None
    recording_started_utc: str | None


def load_targets_from_registry(
    registry_path: Path,
    *,
    recording_id_contains: str | None = None,
    path_contains: str | None = None,
    zarr_use: str = "analysis",
    include_inactive: bool = False,
    limit: int | None = None,
) -> list[DatasetTarget]:
    clauses = ["d.recording_id IS NOT NULL"]
    params: list[Any] = []
    if not include_inactive:
        clauses.append("(d.status IS NULL OR d.status = 'active')")
    if zarr_use != "any":
        clauses.append("d.zarr_use = ?")
        params.append(zarr_use)
    if recording_id_contains:
        clauses.append("d.recording_id LIKE ?")
        params.append(f"%{recording_id_contains}%")
    if path_contains:
        clauses.append("d.zarr_path LIKE ?")
        params.append(f"%{path_contains}%")
    sql = f"""
        SELECT
            d.dataset_id,
            d.recording_id,
            d.zarr_path,
            d.zarr_use,
            d.status,
            r.started_utc AS recording_started_utc
        FROM datasets d
        LEFT JOIN recordings r ON r.recording_id = d.recording_id
        WHERE {' AND '.join(clauses)}
        ORDER BY d.recording_id, d.zarr_use, d.dataset_id
    """
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    conn = sqlite3.connect(str(registry_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [
        DatasetTarget(
            dataset_id=str(row["dataset_id"]),
            recording_id=str(row["recording_id"]),
            zarr_path=Path(str(row["zarr_path"])),
            zarr_use=str(row["zarr_use"]) if row["zarr_use"] is not None else None,
            status=str(row["status"]) if row["status"] is not None else None,
            recording_started_utc=(
                str(row["recording_started_utc"]) if row["recording_started_utc"] is not None else None
            ),
        )
        for row in rows
    ]


def _parse_date(value: str | None) -> date | None:
    text = _as_text(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _recording_date_for_target(target: DatasetTarget) -> tuple[date | None, str | None]:
    started_date = _parse_date(target.recording_started_utc)
    if started_date is not None:
        return started_date, "recordings.started_utc"
    id_date = _parse_date(target.recording_id)
    if id_date is not None:
        return id_date, "recording_id_prefix"
    return None, None


def _derive_dpf_from_fertilization_date(
    target: DatasetTarget,
    date_of_fertilization: str | None,
) -> tuple[int | None, dict[str, Any]]:
    fertilization_date = _parse_date(date_of_fertilization)
    if fertilization_date is None:
        return None, {
            "dpf_from_date_status": "unavailable",
            "dpf_from_date_reason": "missing_or_invalid_date_of_fertilization",
        }
    recording_date, recording_date_source = _recording_date_for_target(target)
    if recording_date is None:
        return None, {
            "dpf_from_date_status": "unavailable",
            "dpf_from_date_reason": "missing_recording_date",
            "date_of_fertilization": fertilization_date.isoformat(),
        }
    dpf = (recording_date - fertilization_date).days
    if dpf < 0:
        return None, {
            "dpf_from_date_status": "invalid",
            "dpf_from_date_reason": "recording_before_fertilization",
            "date_of_fertilization": fertilization_date.isoformat(),
            "recording_date": recording_date.isoformat(),
            "recording_date_source": recording_date_source,
        }
    return dpf, {
        "dpf_from_date_status": "derived",
        "dpf_from_date_reason": "calendar_day_difference",
        "date_of_fertilization": fertilization_date.isoformat(),
        "recording_date": recording_date.isoformat(),
        "recording_date_source": recording_date_source,
    }


def _existing_registry_subject_rows(conn: sqlite3.Connection, recording_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT subject_id, species, dpf_at_acquisition, metadata_json
        FROM recording_subjects
        WHERE recording_id = ?
        ORDER BY subject_id;
        """,
        (recording_id,),
    ).fetchall()


def _registry_rows_compatible(
    rows: Sequence[sqlite3.Row],
    *,
    subject_ids: Sequence[str],
    species: str,
    dpf_at_acquisition: int | None,
) -> bool:
    if not rows:
        return True
    existing_ids = {str(row["subject_id"]) for row in rows}
    expected_ids = set(subject_ids)
    if existing_ids != expected_ids:
        return False
    for row in rows:
        existing_species = str(row["species"] or "").strip()
        if existing_species and existing_species != species:
            return False
        if dpf_at_acquisition is not None and row["dpf_at_acquisition"] is not None:
            try:
                if int(row["dpf_at_acquisition"]) != dpf_at_acquisition:
                    return False
            except (TypeError, ValueError):
                return False
    return True


def _registry_rows_complete(
    rows: Sequence[sqlite3.Row],
    *,
    subject_ids: Sequence[str],
    species: str,
    dpf_at_acquisition: int | None,
    date_of_fertilization: str | None,
) -> bool:
    if not rows:
        return False
    if {str(row["subject_id"]) for row in rows} != set(subject_ids):
        return False
    for row in rows:
        if str(row["species"] or "").strip() != species:
            return False
        if dpf_at_acquisition is not None:
            if row["dpf_at_acquisition"] is None:
                return False
            try:
                if int(row["dpf_at_acquisition"]) != dpf_at_acquisition:
                    return False
            except (TypeError, ValueError):
                return False
        if date_of_fertilization:
            try:
                metadata = json.loads(str(row["metadata_json"] or "{}"))
            except json.JSONDecodeError:
                return False
            if not isinstance(metadata, Mapping):
                return False
            if str(metadata.get("date_of_fertilization") or "").strip() != date_of_fertilization:
                return False
    return True


def _upsert_registry_subjects(
    conn: sqlite3.Connection,
    *,
    target: DatasetTarget,
    species: str,
    subject_ids: Sequence[str],
    dpf_at_acquisition: int | None,
    date_of_fertilization: str | None,
    apply: bool,
    overwrite: bool,
    now: str,
) -> dict[str, Any]:
    existing = _existing_registry_subject_rows(conn, target.recording_id)
    if existing and not overwrite and not _registry_rows_compatible(
        existing,
        subject_ids=subject_ids,
        species=species,
        dpf_at_acquisition=dpf_at_acquisition,
    ):
        return {
            "status": "conflict",
            "reason": "existing_recording_subjects_conflict",
            "existing_subject_count": len(existing),
            "planned_subject_count": len(subject_ids),
        }
    if not apply:
        complete = _registry_rows_complete(
            existing,
            subject_ids=subject_ids,
            species=species,
            dpf_at_acquisition=dpf_at_acquisition,
            date_of_fertilization=date_of_fertilization,
        )
        return {
            "status": "skipped" if complete else "planned",
            "reason": "already_present" if complete else "dry_run",
            "existing_subject_count": len(existing),
            "planned_subject_count": len(subject_ids),
        }
    if not overwrite and _registry_rows_complete(
        existing,
        subject_ids=subject_ids,
        species=species,
        dpf_at_acquisition=dpf_at_acquisition,
        date_of_fertilization=date_of_fertilization,
    ):
        return {
            "status": "skipped",
            "reason": "already_present",
            "existing_subject_count": len(existing),
            "planned_subject_count": len(subject_ids),
        }
    for subject_id in subject_ids:
        metadata = {
            "source": BACKFILL_SOURCE,
            "dataset_id": target.dataset_id,
            "zarr_path": str(target.zarr_path),
            "identity_scope": "recording_local_placeholder",
        }
        if date_of_fertilization:
            metadata["date_of_fertilization"] = date_of_fertilization
        conn.execute(
            """
            INSERT INTO recording_subjects (
                recording_id, subject_id, dataset_id, species, dpf_at_acquisition,
                metadata_json, created_utc, updated_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(recording_id, subject_id) DO UPDATE SET
                dataset_id=excluded.dataset_id,
                species=COALESCE(recording_subjects.species, excluded.species),
                dpf_at_acquisition=COALESCE(recording_subjects.dpf_at_acquisition, excluded.dpf_at_acquisition),
                metadata_json=excluded.metadata_json,
                updated_utc=excluded.updated_utc;
            """,
            (
                target.recording_id,
                subject_id,
                target.dataset_id,
                species,
                dpf_at_acquisition,
                json.dumps(metadata, sort_keys=True),
                now,
                now,
            ),
        )
        conn.execute(
            """
            INSERT INTO subjects (
                subject_id, species, metadata_json, created_utc, updated_utc
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(subject_id) DO UPDATE SET
                species=COALESCE(subjects.species, excluded.species),
                metadata_json=COALESCE(subjects.metadata_json, excluded.metadata_json),
                updated_utc=excluded.updated_utc;
            """,
            (
                subject_id,
                species,
                json.dumps({"source": BACKFILL_SOURCE, "recording_id": target.recording_id}, sort_keys=True),
                now,
                now,
            ),
        )
    return {
        "status": "updated",
        "reason": "updated" if not existing else "refreshed",
        "existing_subject_count": len(existing),
        "planned_subject_count": len(subject_ids),
    }


def _backfill_zarr_subject_context(
    target: DatasetTarget,
    *,
    species: str,
    subject_count: int,
    subject_ids: Sequence[str],
    dpf_at_acquisition: int | None,
    date_of_fertilization: str | None,
    apply: bool,
    overwrite: bool,
    now: str,
) -> dict[str, Any]:
    if _node_attrs_path(target.zarr_path) is None:
        return {"status": "skipped", "reason": "missing_root_zarr_json", "fields": []}

    root_attrs = _read_node_attrs(target.zarr_path)
    analysis_dir = target.zarr_path / "analysis_metadata"
    if apply:
        _ensure_group_node(analysis_dir)
    elif _node_attrs_path(analysis_dir) is None:
        return {
            "status": "planned",
            "reason": "would_create_analysis_metadata",
            "fields": ["analysis_metadata.subject_metadata", "experiment_setup", "subject_count"],
        }
    analysis_attrs = _read_node_attrs(analysis_dir)

    subject_payload = _subject_metadata_payload(
        recording_id=target.recording_id,
        species=species,
        subject_count=subject_count,
        subject_ids=subject_ids,
        dpf_at_acquisition=dpf_at_acquisition,
        date_of_fertilization=date_of_fertilization,
        created_at_utc=now,
    )
    experiment_payload = _experiment_setup_payload(subject_count=subject_count, created_at_utc=now)

    existing_subject_meta = _decode_subject_metadata(analysis_attrs.get("subject_metadata"))
    merged_subject_meta, subject_changed, subject_reason = _merge_subject_metadata(
        existing_subject_meta,
        subject_payload,
        overwrite=overwrite,
    )
    existing_experiment_setup = root_attrs.get("experiment_setup")
    if not isinstance(existing_experiment_setup, Mapping):
        existing_experiment_setup = None
    merged_experiment_setup, experiment_changed, experiment_reason = _merge_experiment_setup(
        existing_experiment_setup,
        experiment_payload,
        overwrite=overwrite,
    )

    root_subject_count = root_attrs.get("subject_count")
    root_subject_count_changed = root_subject_count in (None, "") or overwrite
    if root_subject_count not in (None, ""):
        try:
            root_subject_count_conflict = int(root_subject_count) != subject_count
        except (TypeError, ValueError):
            root_subject_count_conflict = True
        if root_subject_count_conflict and not overwrite:
            return {
                "status": "conflict",
                "reason": "root_subject_count_conflict",
                "fields": ["subject_count"],
            }

    if existing_subject_meta is not None and not overwrite:
        if not _compatible_optional_int(
            existing_subject_meta,
            ("dpf_at_acquisition", "days_post_fertilization"),
            dpf_at_acquisition,
        ):
            subject_reason = "conflict"
        if not _compatible_optional_text(existing_subject_meta, "date_of_fertilization", date_of_fertilization):
            subject_reason = "conflict"

    conflict_reasons = [reason for reason in (subject_reason, experiment_reason) if reason == "conflict"]
    if conflict_reasons:
        return {
            "status": "conflict",
            "reason": ",".join(sorted(set(conflict_reasons))),
            "fields": ["analysis_metadata.subject_metadata", "experiment_setup"],
        }

    changed_fields: list[str] = []
    if subject_changed:
        changed_fields.append("analysis_metadata.subject_metadata")
    if experiment_changed:
        changed_fields.append("experiment_setup")
    if root_subject_count_changed and root_attrs.get("subject_count") != subject_count:
        changed_fields.append("subject_count")

    if not changed_fields:
        return {"status": "skipped", "reason": "already_present", "fields": []}
    if not apply:
        return {"status": "planned", "reason": "dry_run", "fields": sorted(changed_fields)}

    analysis_attrs["subject_metadata"] = json.dumps(merged_subject_meta, sort_keys=True)
    root_attrs["experiment_setup"] = merged_experiment_setup
    root_attrs["subject_count"] = subject_count
    _write_node_attrs(target.zarr_path, root_attrs)
    _write_node_attrs(analysis_dir, analysis_attrs)
    return {"status": "updated", "reason": "updated", "fields": sorted(changed_fields)}


def _row_status(zarr_result: Mapping[str, Any], registry_result: Mapping[str, Any]) -> str:
    statuses = {str(zarr_result.get("status")), str(registry_result.get("status"))}
    if "conflict" in statuses:
        return "conflict"
    if "updated" in statuses:
        return "updated"
    if "planned" in statuses:
        return "planned"
    return "skipped"


def backfill_subject_context_for_targets(
    registry_path: Path,
    targets: Sequence[DatasetTarget],
    *,
    species: str,
    subject_count: int,
    subject_id_template: str,
    apply: bool,
    overwrite: bool,
    dpf_at_acquisition: int | None = None,
    date_of_fertilization: str | None = None,
    derive_dpf_from_metadata: bool = False,
) -> list[dict[str, Any]]:
    now = _utc_now()
    conn = sqlite3.connect(str(registry_path))
    conn.row_factory = sqlite3.Row
    rows: list[dict[str, Any]] = []
    try:
        for target in targets:
            derived = _derive_subject_context_from_metadata(target) if derive_dpf_from_metadata else {}
            effective_date_of_fertilization = (
                date_of_fertilization
                if date_of_fertilization
                else _as_text(derived.get("date_of_fertilization"))
            )
            date_derived_dpf, date_derived_context = _derive_dpf_from_fertilization_date(
                target,
                effective_date_of_fertilization,
            )
            metadata_dpf = _as_int(derived.get("dpf_at_acquisition"))
            effective_dpf = (
                dpf_at_acquisition
                if dpf_at_acquisition is not None
                else (metadata_dpf if metadata_dpf is not None else date_derived_dpf)
            )
            if effective_dpf is not None and dpf_at_acquisition is None and metadata_dpf is None:
                derived["dpf_at_acquisition"] = effective_dpf
                derived["dpf_source"] = "date_of_fertilization_and_recording_date"
            if date_derived_context:
                derived["dpf_from_date"] = date_derived_context
            subject_ids = _subject_ids(target.recording_id, subject_count, subject_id_template)
            if apply:
                existing = _existing_registry_subject_rows(conn, target.recording_id)
                if existing and not overwrite and not _registry_rows_compatible(
                    existing,
                    subject_ids=subject_ids,
                    species=species,
                    dpf_at_acquisition=effective_dpf,
                ):
                    registry_result = {
                        "status": "conflict",
                        "reason": "existing_recording_subjects_conflict",
                        "existing_subject_count": len(existing),
                        "planned_subject_count": len(subject_ids),
                    }
                    zarr_result = {
                        "status": "skipped",
                        "reason": "registry_conflict_not_applied",
                        "fields": [],
                    }
                    rows.append(
                        {
                            "record_type": "subject_context_backfill_action",
                            "dataset_id": target.dataset_id,
                            "recording_id": target.recording_id,
                            "zarr_path": str(target.zarr_path),
                            "zarr_use": target.zarr_use,
                            "status": "conflict",
                            "species": species,
                            "subject_count": subject_count,
                            "subject_ids": list(subject_ids),
                            "zarr": zarr_result,
                            "registry": registry_result,
                        }
                    )
                    continue
            zarr_result = _backfill_zarr_subject_context(
                target,
                species=species,
                subject_count=subject_count,
                subject_ids=subject_ids,
                dpf_at_acquisition=effective_dpf,
                date_of_fertilization=effective_date_of_fertilization,
                apply=apply,
                overwrite=overwrite,
                now=now,
            )
            registry_result = _upsert_registry_subjects(
                conn,
                target=target,
                species=species,
                subject_ids=subject_ids,
                dpf_at_acquisition=effective_dpf,
                date_of_fertilization=effective_date_of_fertilization,
                apply=apply,
                overwrite=overwrite,
                now=now,
            )
            row = {
                "record_type": "subject_context_backfill_action",
                "dataset_id": target.dataset_id,
                "recording_id": target.recording_id,
                "zarr_path": str(target.zarr_path),
                "zarr_use": target.zarr_use,
                "status": _row_status(zarr_result, registry_result),
                "species": species,
                "subject_count": subject_count,
                "dpf_at_acquisition": effective_dpf,
                "date_of_fertilization": effective_date_of_fertilization,
                "derived_context": dict(derived),
                "subject_ids": list(subject_ids),
                "zarr": dict(zarr_result),
                "registry": dict(registry_result),
            }
            rows.append(row)
        if apply and all(row["status"] != "conflict" for row in rows):
            conn.commit()
        elif apply:
            conn.rollback()
    finally:
        conn.close()
    return rows


def _summarize(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    total = 0
    for row in rows:
        total += 1
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return {"total": total, "status_counts": counts}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True, help="Palette registry sqlite path.")
    parser.add_argument("--recording-id-contains", help="Filter registry datasets by recording id substring.")
    parser.add_argument("--path-contains", help="Filter registry datasets by zarr path substring.")
    parser.add_argument("--zarr-use", default="analysis", help="Registry zarr_use filter, or 'any'.")
    parser.add_argument("--include-inactive", action="store_true", help="Include inactive registry dataset rows.")
    parser.add_argument("--limit", type=int, help="Limit selected dataset rows.")
    parser.add_argument("--species", required=True, help="Species display name to stamp, e.g. 'Danionella cerebrum'.")
    parser.add_argument("--subject-count", type=int, default=1, help="Expected subject count per recording.")
    parser.add_argument("--dpf-at-acquisition", type=int, help="Days post fertilization at acquisition.")
    parser.add_argument(
        "--derive-dpf-from-metadata",
        action="store_true",
        help="Use existing Zarr/manifest/H5 subject metadata to fill DPF when --dpf-at-acquisition is omitted.",
    )
    parser.add_argument(
        "--date-of-fertilization",
        help="Literal fertilization date to preserve in metadata, preferably YYYY-MM-DD.",
    )
    parser.add_argument(
        "--subject-id-template",
        default="{recording_id}:subject_{index}",
        help="Format template for deterministic recording-local subject IDs.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite conflicting existing Zarr subject metadata.")
    parser.add_argument("--apply", action="store_true", help="Apply changes. Default is dry-run.")
    parser.add_argument("--output-jsonl", type=Path, help="Optional JSONL report path.")
    parser.add_argument("--summary-json", type=Path, help="Optional summary JSON path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.subject_count < 1:
        parser.error("--subject-count must be >= 1")
    targets = load_targets_from_registry(
        args.registry,
        recording_id_contains=args.recording_id_contains,
        path_contains=args.path_contains,
        zarr_use=str(args.zarr_use),
        include_inactive=bool(args.include_inactive),
        limit=args.limit,
    )
    rows = backfill_subject_context_for_targets(
        args.registry,
        targets,
        species=str(args.species).strip(),
        subject_count=int(args.subject_count),
        subject_id_template=str(args.subject_id_template),
        apply=bool(args.apply),
        overwrite=bool(args.overwrite),
        dpf_at_acquisition=args.dpf_at_acquisition,
        date_of_fertilization=str(args.date_of_fertilization).strip() if args.date_of_fertilization else None,
        derive_dpf_from_metadata=bool(args.derive_dpf_from_metadata),
    )
    summary = _summarize(rows)
    if args.output_jsonl:
        write_jsonl_atomic(args.output_jsonl, rows)
    if args.summary_json:
        write_json_atomic(args.summary_json, summary)
    print(json.dumps(summary, sort_keys=True))
    if any(row["status"] == "conflict" for row in rows):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
