"""SQLite-backed registry for datasets, provenance, and training runs."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import zarr
import yaml


@dataclass(frozen=True)
class RegistryPaths:
    path: Path

    @staticmethod
    def from_env(default_root: Path) -> "RegistryPaths":
        env_path = os.environ.get("PALETTE_REGISTRY_PATH")
        if env_path:
            return RegistryPaths(path=Path(env_path))
        config_path = _load_registry_path(default_root)
        if config_path:
            return RegistryPaths(path=config_path)
        return RegistryPaths(path=default_root / "runs" / "registry" / "palette_registry.sqlite")


def _load_registry_path(default_root: Path) -> Optional[Path]:
    config_path = default_root / "configs" / "fisheye" / "registry.yaml"
    if not config_path.exists():
        return None
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    path_value = None
    if "registry_path" in data:
        path_value = data.get("registry_path")
    elif isinstance(data.get("registry"), dict):
        path_value = data["registry"].get("path")
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _json_dumps(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True)


def _json_loads(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8")
        except Exception:
            return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return None


def _normalize_parents(value: Any) -> List[Dict[str, Optional[str]]]:
    if value is None:
        return []
    if isinstance(value, list):
        parents: List[Dict[str, Optional[str]]] = []
        for item in value:
            if isinstance(item, dict):
                parents.append(
                    {
                        "identifier": item.get("identifier"),
                        "sex": item.get("sex"),
                    }
                )
            elif isinstance(item, str):
                parents.append({"identifier": item, "sex": None})
        return parents
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8")
        except Exception:
            return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        parsed = _json_loads(value)
        if isinstance(parsed, list):
            return _normalize_parents(parsed)
        parents = []
        for part in value.split(";"):
            ident = part.strip()
            if ident:
                parents.append({"identifier": ident, "sex": None})
        return parents
    return []


def _compute_path_hash(path: Path) -> str:
    return sha256(str(path.resolve()).encode("utf-8")).hexdigest()


def _extract_session_uuid(root: zarr.Group) -> Optional[str]:
    for key in ("session_uuid", "session_id"):
        value = root.attrs.get(key)
        if value:
            return str(value)
    analysis = root.get("analysis_metadata")
    if analysis is not None:
        value = analysis.attrs.get("session_uuid")
        if value:
            return str(value)
    return None


def resolve_dataset_id(root: zarr.Group, zarr_path: Path) -> Tuple[str, Optional[str]]:
    session_uuid = _extract_session_uuid(root)
    dataset_id = session_uuid or f"path-{_compute_path_hash(zarr_path)[:12]}"
    return dataset_id, session_uuid


def _extract_protocol(root: zarr.Group) -> Tuple[Optional[str], Optional[str]]:
    stim_parent = None
    if "analysis" in root and "stimulus_runs" in root["analysis"]:
        stim_parent = root["analysis"]["stimulus_runs"]
    if stim_parent is None:
        return None, None
    latest = stim_parent.attrs.get("latest")
    if not latest or latest not in stim_parent:
        return None, None
    stim_group = stim_parent[latest]
    raw = stim_group.attrs.get("protocol_json")
    payload = _json_loads(raw)
    if not payload:
        return None, None
    name = payload.get("protocol_name")
    proto_hash = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return str(name) if name else None, proto_hash


def _extract_snapshot(root: zarr.Group) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    analysis = root.get("analysis_metadata")
    if analysis is not None:
        for key in ("zebrobot_snapshot", "subject_metadata"):
            raw = analysis.attrs.get(key)
            payload = _json_loads(raw)
            if payload:
                return payload, key
    return None, None


def _extract_session_context(root: zarr.Group) -> Dict[str, Any]:
    analysis = root.get("analysis_metadata")
    if analysis is None:
        return {}
    raw = analysis.attrs.get("session_context")
    payload = _json_loads(raw)
    return payload if isinstance(payload, dict) else {}


def _extract_provenance(snapshot: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not snapshot:
        return {}
    dish = snapshot.get("dish") or snapshot
    cross = snapshot.get("cross") or {}
    return {
        "dish_id": snapshot.get("dish_id") or dish.get("dish_id"),
        "cross_id": dish.get("cross_id") or cross.get("cross_id"),
        "line_strain": cross.get("line_strain") or dish.get("line_strain"),
        "genotype": dish.get("genotype"),
        "parents": _normalize_parents(cross.get("parents") or dish.get("parents")),
        "species": dish.get("species"),
        "sex": dish.get("sex"),
        "dpf_at_acquisition": snapshot.get("dpf_at_acquisition"),
        "snapshot_status": snapshot.get("status"),
        "snapshot_missing": snapshot.get("missing"),
    }


class Registry:
    def __init__(self, path: Path):
        self.path = path
        _ensure_parent(self.path)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON;")
        cur.execute("PRAGMA user_version = 1;")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id TEXT PRIMARY KEY,
                session_uuid TEXT,
                zarr_path TEXT NOT NULL,
                path_hash TEXT,
                created_utc TEXT,
                last_seen_utc TEXT,
                status TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS provenance (
                dataset_id TEXT PRIMARY KEY,
                dish_id TEXT,
                cross_id TEXT,
                line_strain TEXT,
                genotype TEXT,
                parents_json TEXT,
                species TEXT,
                sex TEXT,
                dpf_at_acquisition INTEGER,
                rig_id TEXT,
                arena_id TEXT,
                camera_id TEXT,
                canvas_name TEXT,
                protocol_name TEXT,
                protocol_hash TEXT,
                snapshot_status TEXT,
                snapshot_missing_json TEXT,
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS detection_sources (
                dataset_id TEXT NOT NULL,
                refined_run TEXT,
                source_type TEXT,
                counts_json TEXT,
                created_utc TEXT,
                PRIMARY KEY (dataset_id, refined_run, source_type),
                FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS training_sets (
                set_id TEXT PRIMARY KEY,
                name TEXT,
                query_filter TEXT,
                dataset_ids_json TEXT,
                created_utc TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS training_runs (
                run_id TEXT PRIMARY KEY,
                set_id TEXT,
                config_path TEXT,
                manifest_path TEXT,
                model_path TEXT,
                metrics_path TEXT,
                created_utc TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS model_exports (
                run_id TEXT NOT NULL,
                export_type TEXT NOT NULL,
                path TEXT,
                manifest_path TEXT,
                metadata_json TEXT,
                created_utc TEXT,
                PRIMARY KEY (run_id, export_type),
                FOREIGN KEY(run_id) REFERENCES training_runs(run_id) ON DELETE CASCADE
            );
            """
        )
        self.conn.commit()
        self._ensure_columns(
            "provenance",
            {
                "rig_id": "TEXT",
                "arena_id": "TEXT",
                "camera_id": "TEXT",
                "canvas_name": "TEXT",
            },
        )

    def _ensure_columns(self, table: str, columns: Dict[str, str]) -> None:
        existing = {
            row["name"]
            for row in self.conn.execute(f"PRAGMA table_info({table});").fetchall()
        }
        for name, ddl in columns.items():
            if name in existing:
                continue
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl};")
        self.conn.commit()

    def upsert_dataset(self, dataset_id: str, *, session_uuid: Optional[str], zarr_path: Path) -> None:
        now = _utc_now()
        payload = {
            "dataset_id": dataset_id,
            "session_uuid": session_uuid,
            "zarr_path": str(zarr_path),
            "path_hash": _compute_path_hash(zarr_path),
            "created_utc": now,
            "last_seen_utc": now,
            "status": "active",
        }
        self.conn.execute(
            """
            INSERT INTO datasets (dataset_id, session_uuid, zarr_path, path_hash, created_utc, last_seen_utc, status)
            VALUES (:dataset_id, :session_uuid, :zarr_path, :path_hash, :created_utc, :last_seen_utc, :status)
            ON CONFLICT(dataset_id) DO UPDATE SET
                session_uuid=excluded.session_uuid,
                zarr_path=excluded.zarr_path,
                path_hash=excluded.path_hash,
                last_seen_utc=excluded.last_seen_utc,
                status=excluded.status;
            """,
            payload,
        )
        self.conn.commit()

    def upsert_provenance(
        self,
        dataset_id: str,
        *,
        provenance: Dict[str, Any],
        context: Dict[str, Any],
        protocol_name: Optional[str],
        protocol_hash: Optional[str],
    ) -> None:
        payload = {
            "dataset_id": dataset_id,
            "dish_id": provenance.get("dish_id"),
            "cross_id": provenance.get("cross_id"),
            "line_strain": provenance.get("line_strain"),
            "genotype": provenance.get("genotype"),
            "parents_json": _json_dumps(provenance.get("parents")),
            "species": provenance.get("species"),
            "sex": provenance.get("sex"),
            "dpf_at_acquisition": provenance.get("dpf_at_acquisition"),
            "rig_id": context.get("rig_id"),
            "arena_id": context.get("arena_id"),
            "camera_id": context.get("camera_id"),
            "canvas_name": context.get("canvas_name"),
            "protocol_name": protocol_name,
            "protocol_hash": protocol_hash,
            "snapshot_status": provenance.get("snapshot_status"),
            "snapshot_missing_json": _json_dumps(provenance.get("snapshot_missing")),
        }
        self.conn.execute(
            """
            INSERT INTO provenance (
                dataset_id, dish_id, cross_id, line_strain, genotype, parents_json,
                species, sex, dpf_at_acquisition, protocol_name, protocol_hash,
                snapshot_status, snapshot_missing_json
            )
            VALUES (
                :dataset_id, :dish_id, :cross_id, :line_strain, :genotype, :parents_json,
                :species, :sex, :dpf_at_acquisition, :protocol_name, :protocol_hash,
                :snapshot_status, :snapshot_missing_json
            )
            ON CONFLICT(dataset_id) DO UPDATE SET
                dish_id=excluded.dish_id,
                cross_id=excluded.cross_id,
                line_strain=excluded.line_strain,
                genotype=excluded.genotype,
                parents_json=excluded.parents_json,
                species=excluded.species,
                sex=excluded.sex,
                dpf_at_acquisition=excluded.dpf_at_acquisition,
                protocol_name=excluded.protocol_name,
                protocol_hash=excluded.protocol_hash,
                snapshot_status=excluded.snapshot_status,
                snapshot_missing_json=excluded.snapshot_missing_json;
            """,
            payload,
        )
        self.conn.commit()

    def register_from_root(self, root: zarr.Group, zarr_path: Path) -> str:
        dataset_id, session_uuid = resolve_dataset_id(root, zarr_path)
        self.upsert_dataset(dataset_id, session_uuid=session_uuid, zarr_path=zarr_path)

        protocol_name, protocol_hash = _extract_protocol(root)
        snapshot, _ = _extract_snapshot(root)
        provenance = _extract_provenance(snapshot)
        context = _extract_session_context(root)
        self.upsert_provenance(
            dataset_id,
            provenance=provenance,
            context=context,
            protocol_name=protocol_name,
            protocol_hash=protocol_hash,
        )
        return dataset_id

    def record_training_run(
        self,
        *,
        run_id: str,
        set_id: Optional[str],
        config_path: Optional[Path],
        manifest_path: Optional[Path],
        model_path: Optional[Path],
        metrics_path: Optional[Path],
    ) -> None:
        payload = {
            "run_id": run_id,
            "set_id": set_id,
            "config_path": str(config_path) if config_path else None,
            "manifest_path": str(manifest_path) if manifest_path else None,
            "model_path": str(model_path) if model_path else None,
            "metrics_path": str(metrics_path) if metrics_path else None,
            "created_utc": _utc_now(),
        }
        self.conn.execute(
            """
            INSERT INTO training_runs (run_id, set_id, config_path, manifest_path, model_path, metrics_path, created_utc)
            VALUES (:run_id, :set_id, :config_path, :manifest_path, :model_path, :metrics_path, :created_utc)
            ON CONFLICT(run_id) DO UPDATE SET
                set_id=excluded.set_id,
                config_path=excluded.config_path,
                manifest_path=excluded.manifest_path,
                model_path=excluded.model_path,
                metrics_path=excluded.metrics_path,
                created_utc=excluded.created_utc;
            """,
            payload,
        )
        self.conn.commit()

    def record_model_export(
        self,
        *,
        run_id: str,
        export_type: str,
        path: Optional[Path],
        manifest_path: Optional[Path] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "run_id": run_id,
            "export_type": export_type,
            "path": str(path) if path else None,
            "manifest_path": str(manifest_path) if manifest_path else None,
            "metadata_json": _json_dumps(metadata),
            "created_utc": _utc_now(),
        }
        self.conn.execute(
            """
            INSERT INTO model_exports (run_id, export_type, path, manifest_path, metadata_json, created_utc)
            VALUES (:run_id, :export_type, :path, :manifest_path, :metadata_json, :created_utc)
            ON CONFLICT(run_id, export_type) DO UPDATE SET
                path=excluded.path,
                manifest_path=excluded.manifest_path,
                metadata_json=excluded.metadata_json,
                created_utc=excluded.created_utc;
            """,
            payload,
        )
        self.conn.commit()

    def scan_zarr(self, zarr_path: Path) -> Optional[str]:
        if not zarr_path.exists():
            return None
        root = zarr.open(str(zarr_path), mode="r")
        return self.register_from_root(root, zarr_path)


def scan_paths(
    registry: Registry,
    paths: Iterable[Path],
    *,
    recursive: bool = False,
) -> List[str]:
    dataset_ids: List[str] = []
    for path in paths:
        if path.is_dir() and _is_zarr_root(path):
            dataset_id = registry.scan_zarr(path)
            if dataset_id:
                dataset_ids.append(dataset_id)
            continue
        if path.is_dir() and recursive:
            for candidate in _find_zarr_roots(path):
                dataset_id = registry.scan_zarr(candidate)
                if dataset_id:
                    dataset_ids.append(dataset_id)
    return dataset_ids


def _is_zarr_root(path: Path) -> bool:
    return (path / "zarr.json").exists() or (path / ".zgroup").exists()


def _find_zarr_roots(root: Path) -> List[Path]:
    roots: List[Path] = []
    for entry in root.rglob("zarr.json"):
        roots.append(entry.parent)
    for entry in root.rglob(".zgroup"):
        if entry.parent not in roots:
            roots.append(entry.parent)
    return roots
