#!/usr/bin/env python3
from __future__ import annotations

from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr
import argparse
from copy import deepcopy
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import h5py
import zarr

from fisheye.shared.type_conversions import normalize_attr as _normalize_attr


DEFAULT_KEYS = [
    "dish_mask",
    "detection_tuning",
    "keypoint_tuning",
    "subject_mask_tuning",
    "subdish_mask_tuning",
]
SUBJECT_MASK_TUNING_KEY = "subject_mask_tuning"
SUBJECT_TUNING_VERSION = "2.0"
SUPPORTED_SUBJECT_MASK_COMPONENTS = {
    "subject_body",
    "eyes_union",
    "eye_left",
    "eye_right",
    "swim_bladder",
}
SOURCE_SPECIFIC_SUBJECT_TUNING_CONTEXT_KEYS = {
    "crop_run",
    "keypoint_run",
    "keypoint_source",
    "subject_preview_run",
    "frame_index",
    "detection_index",
    "roi_index",
    "roi_index_one_based",
    "total_rois",
    "roi_success",
    "patch_center_source",
}


@dataclass
class TargetPlan:
    zarr_path: Path
    camera_id: Optional[str]
    zarr_use: Optional[str]
    status: str
    reason: Optional[str] = None


def _derive_camera_id(ipc_source_name: object) -> Optional[str]:
    if ipc_source_name is None:
        return None
    text = _normalize_attr(ipc_source_name)
    if text is None:
        return None
    match = re.search(r"cam_(\d+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    digits = re.findall(r"\d+", text)
    return digits[-1] if digits else None


def _read_camera_id(h5_path: Path) -> Optional[str]:
    with h5py.File(h5_path, "r") as h5:
        root = h5.attrs
        if "camera_id" in root:
            cam = _normalize_attr(root.get("camera_id"))
            if cam:
                return cam
        ipc = _normalize_attr(root.get("ipc_source_name"))
        return _derive_camera_id(ipc)


def _resolve_root(paths: Optional[List[Path]]) -> List[Path]:
    if paths:
        return paths
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return [Path(env_root)]
    return [Path("/nvme1/recordings")]


def _open_group(path: Path, mode: str) -> zarr.Group:
    try:
        return zarr.open_group(str(path), mode=mode, use_consolidated=False)
    except TypeError:
        return zarr.open_group(str(path), mode=mode)


def _infer_zarr_use(root: zarr.Group, zarr_path: Path) -> Optional[str]:
    for key in ("zarr_use", "zarr_purpose"):
        raw = _normalize_attr(root.attrs.get(key))
        if raw in {"analysis", "training"}:
            return raw
    name = zarr_path.name.lower()
    if name.endswith("_analysis.zarr"):
        return "analysis"
    if name.endswith("_training.zarr"):
        return "training"
    return None


def _parse_keys(keys: Optional[str]) -> List[str]:
    if keys is None or keys.strip() == "" or keys.strip().lower() == "all":
        return list(DEFAULT_KEYS)
    return [key.strip() for key in keys.split(",") if key.strip()]


def _normalize_subject_mask_component_name(value: object) -> Optional[str]:
    text = _normalize_attr(value)
    if text is None:
        return None
    normalized = text.strip().lower().replace(" ", "_")
    if not normalized:
        return None
    aliases = {
        "subject": "subject_body",
        "body": "subject_body",
        "subject-body": "subject_body",
        "eyes": "eyes_union",
        "eye_union": "eyes_union",
        "eye-union": "eyes_union",
        "left_eye": "eye_left",
        "left-eye": "eye_left",
        "right_eye": "eye_right",
        "right-eye": "eye_right",
        "swimbladder": "swim_bladder",
        "swim-bladder": "swim_bladder",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_SUBJECT_MASK_COMPONENTS:
        return None
    return normalized


def _parse_subject_mask_components(values: Optional[Sequence[str]]) -> List[str]:
    components: List[str] = []
    seen: set[str] = set()
    for raw_value in values or ():
        for token in str(raw_value).split(","):
            token = token.strip()
            if not token:
                continue
            normalized = _normalize_subject_mask_component_name(token)
            if normalized is None:
                supported = ", ".join(sorted(SUPPORTED_SUBJECT_MASK_COMPONENTS))
                raise RuntimeError(
                    f"Unsupported --subject-mask-components value {token!r}. "
                    f"Expected one of: {supported}."
                )
            if normalized in seen:
                continue
            seen.add(normalized)
            components.append(normalized)
    return components


def _normalize_subject_mask_tuning_payload(raw: object) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {"version": SUBJECT_TUNING_VERSION, "components": {}}

    components_raw = raw.get("components")
    if isinstance(components_raw, Mapping):
        components: Dict[str, Dict[str, Any]] = {}
        for key, value in components_raw.items():
            normalized = _normalize_subject_mask_component_name(key)
            if normalized is None or not isinstance(value, Mapping):
                continue
            components[normalized] = {str(entry_key): deepcopy(entry_value) for entry_key, entry_value in value.items()}
        payload = {str(key): deepcopy(value) for key, value in raw.items()}
        payload["version"] = str(payload.get("version") or SUBJECT_TUNING_VERSION)
        payload["components"] = components
        return payload

    tuned = raw.get("tuned_parameters")
    if not isinstance(tuned, Mapping):
        return {"version": SUBJECT_TUNING_VERSION, "components": {}}
    return {
        "version": SUBJECT_TUNING_VERSION,
        "components": {
            "subject_body": {
                "method": str(raw.get("method") or "traditional_subject_mask_seed"),
                "version": str(raw.get("version") or "1.0"),
                "tuned_timestamp": raw.get("tuned_timestamp"),
                "tuned_parameters": {str(key): deepcopy(value) for key, value in tuned.items()},
                "context": (
                    {str(key): deepcopy(value) for key, value in raw.get("context", {}).items()}
                    if isinstance(raw.get("context"), Mapping)
                    else {}
                ),
            }
        },
        "migrated_from_flat": True,
    }


def _filter_subject_mask_tuning_payload(
    raw: object,
    components: Sequence[str],
) -> tuple[Dict[str, Any], List[str]]:
    payload = _normalize_subject_mask_tuning_payload(raw)
    source_components = payload.get("components", {})
    filtered_components: Dict[str, Any] = {}
    missing: List[str] = []
    for component_name in components:
        if component_name not in source_components:
            missing.append(component_name)
            continue
        filtered_components[component_name] = _scrub_subject_tuning_component_for_camera_propagation(
            component_name,
            deepcopy(source_components[component_name]),
        )
    return {
        "version": str(payload.get("version") or SUBJECT_TUNING_VERSION),
        "components": filtered_components,
    }, missing


def _scrub_subject_tuning_component_for_camera_propagation(
    component_name: str,
    component_value: object,
) -> object:
    if not isinstance(component_value, Mapping):
        return component_value

    normalized = {str(key): deepcopy(value) for key, value in component_value.items()}
    context_raw = normalized.get("context")
    if not isinstance(context_raw, Mapping):
        return normalized

    context = {str(key): deepcopy(value) for key, value in context_raw.items()}
    removed_any = False
    for key in SOURCE_SPECIFIC_SUBJECT_TUNING_CONTEXT_KEYS:
        if key in context:
            context.pop(key, None)
            removed_any = True

    if removed_any:
        if context:
            normalized["context"] = context
        else:
            normalized.pop("context", None)
        normalized["propagated_by_camera"] = True
        normalized["propagated_component"] = str(component_name)
    return normalized


def _format_tuning_targets(
    tuning: Mapping[str, object],
    *,
    subject_mask_components: Optional[Sequence[str]] = None,
) -> List[str]:
    targets: List[str] = []
    for key in tuning.keys():
        if key == SUBJECT_MASK_TUNING_KEY and subject_mask_components:
            targets.extend(f"{key}.components.{component_name}" for component_name in subject_mask_components)
        else:
            targets.append(str(key))
    return targets


def _apply_subject_mask_component_updates(
    attrs: Any,
    value: object,
    *,
    overwrite: bool,
    merge_dicts: bool,
) -> tuple[List[str], List[str]]:
    target_raw = attrs.get(SUBJECT_MASK_TUNING_KEY)
    target_payload = _normalize_subject_mask_tuning_payload(target_raw)
    target_components = dict(target_payload.get("components", {}))
    source_payload = _normalize_subject_mask_tuning_payload(value)
    source_components = dict(source_payload.get("components", {}))

    updated_paths: List[str] = []
    skipped_paths: List[str] = []
    for component_name, component_value in source_components.items():
        path = f"{SUBJECT_MASK_TUNING_KEY}.components.{component_name}"
        if component_name in target_components and not overwrite:
            skipped_paths.append(path)
            continue
        if component_name in target_components:
            new_value = _compute_tuning_value(
                target_components[component_name],
                component_value,
                merge_dicts=merge_dicts,
            )
            if new_value == target_components[component_name]:
                skipped_paths.append(path)
                continue
            target_components[component_name] = new_value
        else:
            target_components[component_name] = deepcopy(component_value)
        updated_paths.append(path)

    if not updated_paths:
        return updated_paths, skipped_paths

    target_payload["version"] = str(target_payload.get("version") or source_payload.get("version") or SUBJECT_TUNING_VERSION)
    target_payload["components"] = target_components
    attrs[SUBJECT_MASK_TUNING_KEY] = target_payload
    return updated_paths, skipped_paths


def _read_camera_id_from_zarr(root: zarr.Group) -> Optional[str]:
    camera_id = _normalize_attr(root.attrs.get("camera_id"))
    if camera_id:
        return camera_id

    raw_video = root.get("raw_video")
    if raw_video is not None:
        raw_camera_id = _normalize_attr(raw_video.attrs.get("camera_id"))
        if raw_camera_id:
            return raw_camera_id

    analysis = root.get("analysis_metadata")
    if analysis is not None:
        raw = analysis.attrs.get("camera_metadata")
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "ignore")
        if isinstance(raw, str):
            try:
                payload = json.loads(raw)
            except Exception:
                payload = None
        elif isinstance(raw, dict):
            payload = raw
        else:
            payload = None
        if isinstance(payload, dict):
            for key in ("camera_id", "device_serial_number", "device_snmp_comm_read"):
                value = _normalize_attr(payload.get(key))
                if value:
                    return value

    return None


def _candidate_h5_stems(zarr_stem: str) -> List[str]:
    stems = [zarr_stem]
    for suffix in ("_analysis", "_training"):
        if zarr_stem.endswith(suffix):
            stems.append(zarr_stem[: -len(suffix)])
    out: List[str] = []
    seen: set[str] = set()
    for stem in stems:
        if stem and stem not in seen:
            seen.add(stem)
            out.append(stem)
    return out


def _find_h5_for_zarr(zarr_path: Path) -> Optional[Path]:
    if zarr_path.parent.name == "zarr":
        recording_dir = zarr_path.parent.parent
    else:
        recording_dir = zarr_path.parent
    raw_dir = recording_dir / "raw"
    if not raw_dir.exists():
        return None

    for stem in _candidate_h5_stems(zarr_path.stem):
        for suffix in (".h5", ".hdf5"):
            candidate = raw_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    return None


def _camera_id_for_zarr(zarr_path: Path, root: zarr.Group) -> Optional[str]:
    camera_id = _read_camera_id_from_zarr(root)
    if camera_id:
        return camera_id
    h5_path = _find_h5_for_zarr(zarr_path)
    if h5_path is None:
        return None
    try:
        return _read_camera_id(h5_path)
    except Exception:
        return None


def _load_source_tuning(
    source_zarr: Path,
    keys: Sequence[str],
    *,
    subject_mask_components: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    root = _open_group(source_zarr, mode="r")
    analysis = root.get("analysis_metadata")
    if analysis is None:
        return {}
    attrs = dict(analysis.attrs)
    if subject_mask_components and SUBJECT_MASK_TUNING_KEY in keys and SUBJECT_MASK_TUNING_KEY not in attrs:
        raise RuntimeError("Source Zarr is missing subject_mask_tuning.")
    tuning: Dict[str, object] = {}
    for key in keys:
        if key not in attrs:
            continue
        value = attrs[key]
        if key == SUBJECT_MASK_TUNING_KEY and subject_mask_components:
            filtered_value, missing_components = _filter_subject_mask_tuning_payload(value, subject_mask_components)
            if missing_components:
                missing_text = ", ".join(missing_components)
                raise RuntimeError(
                    f"Source Zarr is missing requested subject_mask_tuning components: {missing_text}."
                )
            tuning[key] = filtered_value
            continue
        tuning[key] = value
    return tuning


def _merge_mapping_values(base: Mapping[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {str(key): deepcopy(value) for key, value in base.items()}
    for key, value in update.items():
        key_text = str(key)
        existing = merged.get(key_text)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key_text] = _merge_mapping_values(existing, value)
        else:
            merged[key_text] = deepcopy(value)
    return merged


def _compute_tuning_value(existing: object, source: object, *, merge_dicts: bool) -> object:
    if merge_dicts and isinstance(existing, Mapping) and isinstance(source, Mapping):
        return _merge_mapping_values(existing, source)
    return deepcopy(source)


def _resolve_target_use(mode: str, source_use: Optional[str]) -> Optional[str]:
    normalized = mode.strip().lower()
    if normalized == "all":
        return None
    if normalized in {"analysis", "training"}:
        return normalized
    if normalized == "source":
        if source_use in {"analysis", "training"}:
            return source_use
        raise RuntimeError(
            "Unable to infer source zarr use. Pass --zarr-use training|analysis|all explicitly."
        )
    raise RuntimeError(f"Unsupported --zarr-use value: {mode}")


def _build_plans(
    roots: List[Path],
    recursive: bool,
    camera_id: str,
    target_use: Optional[str],
    source: Path,
) -> List[TargetPlan]:
    plans: List[TargetPlan] = []
    seen: set[Path] = set()
    for zarr_path in _iter_zarr(roots, recursive):
        resolved = zarr_path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved == source.resolve():
            continue

        try:
            root = _open_group(resolved, mode="r")
        except Exception as exc:
            plans.append(
                TargetPlan(
                    zarr_path=resolved,
                    camera_id=None,
                    zarr_use=None,
                    status="error",
                    reason=f"zarr open failed: {exc}",
                )
            )
            continue

        zarr_use = _infer_zarr_use(root, resolved)
        if target_use is not None and zarr_use != target_use:
            continue

        target_camera = _camera_id_for_zarr(resolved, root)
        if target_camera is None:
            plans.append(
                TargetPlan(
                    zarr_path=resolved,
                    camera_id=None,
                    zarr_use=zarr_use,
                    status="skipped",
                    reason="camera_id missing",
                )
            )
            continue
        if target_camera != camera_id:
            continue

        plans.append(
            TargetPlan(
                zarr_path=resolved,
                camera_id=target_camera,
                zarr_use=zarr_use,
                status="ok",
            )
        )
    return plans


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Apply tuning parameters from a source Zarr to other Zarr archives "
            "that share the same camera_id."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Recording root(s) or zarr path(s) to scan (default: $PALETTE_RECORDINGS_ROOT or /nvme1/recordings).",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Source Zarr with tuned analysis_metadata attrs.",
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        help="Camera ID to match (default: inferred from source zarr metadata/h5).",
    )
    parser.add_argument(
        "--zarr-use",
        default="source",
        choices=["source", "analysis", "training", "all"],
        help="Target zarr use scope (default: source).",
    )
    parser.add_argument(
        "--keys",
        type=str,
        help=f"Comma-separated tuning keys to copy (default: {', '.join(DEFAULT_KEYS)}; use 'all').",
    )
    parser.add_argument(
        "--subject-mask-components",
        action="append",
        help=(
            "Optional comma-separated subject_mask_tuning component names to copy "
            "(for example swim_bladder). Preserves unrelated target components and "
            "skips matching target components unless --overwrite is given."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan for zarr archives under each root.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply updates (default: dry-run).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing tuning keys in target zarrs.",
    )
    parser.add_argument(
        "--merge-dicts",
        action="store_true",
        help=(
            "When both source and target tuning values are dict-like, recursively merge them "
            "with source paths overriding matching target paths while preserving unrelated target entries."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON lines for each plan/apply event.",
    )

    args = parser.parse_args(argv)
    roots = _resolve_root(args.paths)
    source = args.source.expanduser().resolve()
    if not source.exists():
        print(f"Source Zarr not found: {source}")
        return 1

    try:
        source_root = _open_group(source, mode="r")
    except Exception as exc:
        print(f"Failed to open source zarr: {source} ({exc})")
        return 1

    source_use = _infer_zarr_use(source_root, source)
    try:
        target_use = _resolve_target_use(args.zarr_use, source_use)
    except RuntimeError as exc:
        print(str(exc))
        return 1

    camera_id = args.camera_id or _camera_id_for_zarr(source, source_root)
    if not camera_id:
        print("Unable to infer camera_id from source; pass --camera-id explicitly.")
        return 1

    keys = _parse_keys(args.keys)
    try:
        subject_mask_components = _parse_subject_mask_components(args.subject_mask_components)
    except RuntimeError as exc:
        print(str(exc))
        return 1
    if subject_mask_components and SUBJECT_MASK_TUNING_KEY not in keys:
        print("--subject-mask-components requires subject_mask_tuning to be included in --keys.")
        return 1
    try:
        tuning = _load_source_tuning(
            source,
            keys,
            subject_mask_components=subject_mask_components,
        )
    except RuntimeError as exc:
        print(str(exc))
        return 1
    if not tuning:
        print("No matching tuning keys found in source analysis_metadata.")
        return 1
    tuning_targets = _format_tuning_targets(
        tuning,
        subject_mask_components=subject_mask_components,
    )

    plans = _build_plans(roots, bool(args.recursive), camera_id, target_use, source)
    if not plans:
        use_label = target_use or "all"
        print(f"No target zarrs found for camera_id={camera_id} (zarr_use={use_label}).")
        return 0

    ok_plans = [plan for plan in plans if plan.status == "ok"]
    pre_skipped = [plan for plan in plans if plan.status == "skipped"]
    pre_errors = [plan for plan in plans if plan.status == "error"]

    updated_targets = 0
    unchanged_targets = 0
    apply_errors = 0

    for plan in plans:
        if plan.status != "ok":
            payload = {
                "status": plan.status,
                "zarr": str(plan.zarr_path),
                "camera_id": plan.camera_id,
                "zarr_use": plan.zarr_use,
                "reason": plan.reason,
            }
            if args.json:
                print(json.dumps(payload, sort_keys=True))
            else:
                print(f"Skipping ({plan.reason}): {plan.zarr_path}")
            continue

        if not args.apply:
            payload = {
                "status": "plan",
                "zarr": str(plan.zarr_path),
                "camera_id": plan.camera_id,
                "zarr_use": plan.zarr_use,
                "keys": tuning_targets,
            }
            if args.json:
                print(json.dumps(payload, sort_keys=True))
            else:
                print(
                    f"Would apply {tuning_targets} to {plan.zarr_path} "
                    f"(camera_id={plan.camera_id}, zarr_use={plan.zarr_use or 'unknown'})"
                )
            continue

        try:
            root = _open_group(plan.zarr_path, mode="a")
            analysis = root.require_group("analysis_metadata")
            attrs = analysis.attrs
            updated: List[str] = []
            skipped_keys: List[str] = []
            for key, value in tuning.items():
                if key == SUBJECT_MASK_TUNING_KEY and subject_mask_components:
                    component_updates, component_skips = _apply_subject_mask_component_updates(
                        attrs,
                        value,
                        overwrite=bool(args.overwrite),
                        merge_dicts=bool(args.merge_dicts),
                    )
                    updated.extend(component_updates)
                    skipped_keys.extend(component_skips)
                    continue
                if key in attrs and args.merge_dicts and isinstance(attrs[key], Mapping) and isinstance(value, Mapping):
                    merged_value = _compute_tuning_value(attrs[key], value, merge_dicts=True)
                    if merged_value == attrs[key]:
                        skipped_keys.append(key)
                        continue
                    attrs[key] = merged_value
                    updated.append(key)
                    continue
                if not args.overwrite and key in attrs:
                    skipped_keys.append(key)
                    continue
                new_value = _compute_tuning_value(attrs.get(key), value, merge_dicts=bool(args.merge_dicts))
                if key in attrs and new_value == attrs[key]:
                    skipped_keys.append(key)
                    continue
                attrs[key] = new_value
                updated.append(key)
            status = "applied" if updated else "unchanged"
            if updated:
                updated_targets += 1
            else:
                unchanged_targets += 1
            payload = {
                "status": status,
                "zarr": str(plan.zarr_path),
                "updated": updated,
                "skipped_keys": skipped_keys,
                "camera_id": plan.camera_id,
                "zarr_use": plan.zarr_use,
            }
            if args.json:
                print(json.dumps(payload, sort_keys=True))
            else:
                if updated:
                    print(f"Applied {updated} to {plan.zarr_path}")
                if skipped_keys:
                    print(f"  Skipped existing keys: {skipped_keys}")
        except Exception as exc:
            apply_errors += 1
            payload = {
                "status": "error",
                "zarr": str(plan.zarr_path),
                "reason": str(exc),
                "camera_id": plan.camera_id,
                "zarr_use": plan.zarr_use,
            }
            if args.json:
                print(json.dumps(payload, sort_keys=True))
            else:
                print(f"Failed to apply tuning to {plan.zarr_path}: {exc}")

    if not args.apply:
        print(
            f"\nDry-run complete. targets={len(plans)} ok={len(ok_plans)} "
            f"skipped={len(pre_skipped)} errors={len(pre_errors)}"
        )
        return 0

    print("\nSummary")
    print(f"  source: {source}")
    print(f"  camera_id: {camera_id}")
    print(f"  zarr_use: {target_use or 'all'}")
    print(f"  keys: {tuning_targets}")
    print(f"  planned_ok: {len(ok_plans)}")
    print(f"  pre_skipped: {len(pre_skipped)}")
    print(f"  pre_errors: {len(pre_errors)}")
    print(f"  updated: {updated_targets}")
    print(f"  unchanged: {unchanged_targets}")
    print(f"  apply_errors: {apply_errors}")
    return 0 if apply_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
