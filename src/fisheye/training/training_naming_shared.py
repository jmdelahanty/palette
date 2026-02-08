"""Shared helpers for training run naming and project directory resolution."""

from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Optional

from rich.console import Console


def strip_manifest_suffixes(value: str) -> str:
    text = str(value).strip()
    while text.endswith(".manifest"):
        text = text[: -len(".manifest")]
    return text


def sanitize_run_component(value: Optional[str], fallback: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text or fallback


def infer_set_slug(set_id: Optional[str], config_path: Optional[Path], fallback: str) -> str:
    if set_id:
        slug = strip_manifest_suffixes(set_id)
        return slug or fallback
    if config_path is not None:
        stem = strip_manifest_suffixes(config_path.stem)
        return stem or fallback
    return fallback


def resolve_project_dir(
    *,
    args,
    training_params: dict,
    set_id: Optional[str],
    config_path: Optional[Path],
    task_subdir: str,
    default_slug: str,
    console: Console,
) -> None:
    if args.project:
        training_params["project"] = str(Path(args.project).expanduser().resolve())
        return

    configured_project = training_params.get("project")
    if isinstance(configured_project, str) and configured_project.strip():
        training_params["project"] = str(Path(configured_project).expanduser().resolve())
        return

    nvme_root = Path("/nvme1")
    if not nvme_root.exists():
        return

    slug = infer_set_slug(set_id, config_path, default_slug)
    project_dir = (nvme_root / "models" / task_subdir / slug).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    training_params["project"] = str(project_dir)
    console.print(f"[cyan]Using default model output directory:[/cyan] {project_dir}")


def _resolve_manifest_version_token(manifest_summary: dict) -> str:
    for key in ("manifest_set_id", "manifest_set_slug"):
        raw = str(manifest_summary.get(key) or "").strip().lower()
        if not raw:
            continue
        match = re.search(r"_v(?P<num>\d+)$", raw)
        if not match:
            continue
        try:
            version_num = int(match.group("num"))
        except Exception:
            continue
        if version_num >= 0:
            return f"v{version_num:03d}"
    return "v001"


def _resolve_run_hash(
    *,
    manifest_summary: dict,
    task: str,
    stamp: str,
    pid: int,
) -> str:
    manifest_sha = str(manifest_summary.get("manifest_sha256") or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{8,}", manifest_sha):
        return manifest_sha[:8]

    seed = "|".join(
        [
            str(manifest_summary.get("manifest_set_id") or ""),
            str(manifest_summary.get("manifest_set_slug") or ""),
            str(manifest_summary.get("manifest_rig_name") or ""),
            str(manifest_summary.get("manifest_dish_design") or ""),
            str(manifest_summary.get("manifest_canvas_name") or ""),
            task,
            stamp,
            str(pid),
        ]
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8]


def _infer_dish_canvas_from_set_slug(manifest_summary: dict) -> tuple[Optional[str], Optional[str]]:
    raw = str(manifest_summary.get("manifest_set_slug") or manifest_summary.get("manifest_set_id") or "").strip()
    if not raw:
        return None, None
    slug = sanitize_run_component(raw, "")
    if not slug:
        return None, None
    slug = re.sub(r"_v\d+$", "", slug)
    if slug.startswith("detect_"):
        slug = slug[len("detect_") :]
    elif slug.startswith("pose_"):
        slug = slug[len("pose_") :]
    parts = [part for part in slug.split("_") if part]
    if len(parts) < 2:
        return None, None
    return parts[0], parts[1]


def build_default_detect_run_name(
    *,
    manifest_summary: dict,
    task_fallback: str,
    timestamp: Optional[str] = None,
    pid: Optional[int] = None,
) -> str:
    fallback_dish, fallback_canvas = _infer_dish_canvas_from_set_slug(manifest_summary)
    rig = sanitize_run_component(manifest_summary.get("manifest_rig_name"), "unknown_rig")
    dish = sanitize_run_component(manifest_summary.get("manifest_dish_design") or fallback_dish, "unknown_dish")
    canvas = sanitize_run_component(
        manifest_summary.get("manifest_canvas_name") or fallback_canvas,
        "unknown_canvas",
    )
    version = _resolve_manifest_version_token(manifest_summary)
    task = sanitize_run_component(manifest_summary.get("manifest_task") or task_fallback, task_fallback)
    stamp = timestamp or time.strftime("%Y%m%d-%H%M%S")
    process_id = int(os.getpid() if pid is None else pid)
    short_hash = _resolve_run_hash(
        manifest_summary=manifest_summary,
        task=task,
        stamp=stamp,
        pid=process_id,
    )
    return f"{rig}_{dish}_{canvas}_{version}_{task}_{stamp}_{short_hash}"


def build_default_pose_run_name(
    *,
    manifest_hints: dict,
    task_fallback: str,
    timestamp: Optional[str] = None,
    pid: Optional[int] = None,
) -> str:
    manifest_summary = {
        "manifest_set_id": manifest_hints.get("set_id"),
        "manifest_set_slug": manifest_hints.get("set_slug") or manifest_hints.get("set_id"),
        "manifest_task": manifest_hints.get("task"),
        "manifest_rig_name": manifest_hints.get("rig_name"),
        "manifest_dish_design": manifest_hints.get("dish_design"),
        "manifest_canvas_name": manifest_hints.get("canvas_name"),
        "manifest_sha256": manifest_hints.get("manifest_sha256"),
    }
    return build_default_detect_run_name(
        manifest_summary=manifest_summary,
        task_fallback=task_fallback,
        timestamp=timestamp,
        pid=pid,
    )
