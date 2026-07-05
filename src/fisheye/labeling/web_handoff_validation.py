"""User-handoff validation/evidence file helpers for web labeling."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _require_validation_dependencies(
    dependencies: Mapping[str, object],
    names: tuple[str, ...],
) -> dict[str, Any]:
    missing = [name for name in names if name not in dependencies]
    if missing:
        raise KeyError(f"missing user handoff validation dependencies: {', '.join(missing)}")
    return {name: dependencies[name] for name in names}


def _write_user_handoff_validation_log_impl(
    manifest: dict[str, object],
    output_path: Path,
    *,
    dependencies: Mapping[str, object],
) -> None:
    validation_dependencies = _require_validation_dependencies(
        dependencies, ('_write_web_labeling_validation_log',)
    )
    _write_web_labeling_validation_log = validation_dependencies['_write_web_labeling_validation_log']
    _write_web_labeling_validation_log(manifest, output_path, bundle_label="single-user handoff bundle")



def _write_user_handoff_validation_checklist_impl(
    manifest: dict[str, object],
    output_path: Path,
    *,
    dependencies: Mapping[str, object],
) -> None:
    validation_dependencies = _require_validation_dependencies(
        dependencies, ('_write_web_labeling_validation_checklist',)
    )
    _write_web_labeling_validation_checklist = validation_dependencies['_write_web_labeling_validation_checklist']
    _write_web_labeling_validation_checklist(manifest, output_path, bundle_label="single-user handoff bundle")



def _inspect_handoff_validation_log(path: Path) -> dict[str, object]:
    required_name = "validation-log-template.md"
    if path.is_dir():
        required_path = path / required_name
        related_paths = sorted(str(candidate) for candidate in path.rglob(required_name) if candidate.is_file())
        return {
            "required": True,
            "present": required_path.exists(),
            "required_path": str(required_path),
            "matched_paths": [str(required_path)] if required_path.exists() else [],
            "related_paths": related_paths,
        }
    if path.is_file() and path.suffix.lower() == ".zip":
        import zipfile

        with zipfile.ZipFile(path) as archive:
            names = sorted(
                name
                for name in archive.namelist()
                if name == required_name or name.endswith(f"/{required_name}")
            )
        top_level_names = [
            name
            for name in names
            if name == required_name or len([part for part in name.split("/") if part]) == 2
        ]
        return {
            "required": True,
            "present": bool(top_level_names),
            "required_path": f"*/{required_name}",
            "matched_paths": top_level_names,
            "related_paths": names,
        }
    return {
        "required": True,
        "present": False,
        "required_path": required_name,
        "matched_paths": [],
        "related_paths": [],
    }



def _operator_evidence_command_sheet_boundary_status(text: str) -> dict[str, object]:
    required_phrases = [
        "Boundary: operator-only",
        "not labeler instructions",
        "Do not send this command sheet",
        "Labelers should use only their guarded browser links",
    ]
    missing_phrases = [phrase for phrase in required_phrases if phrase not in text]
    return {
        "operator_only_boundary_present": not missing_phrases,
        "operator_only_boundary_required_phrases": required_phrases,
        "operator_only_boundary_missing_phrases": missing_phrases,
    }



def _inspect_handoff_operator_evidence_commands(path: Path, *, required: bool = True) -> dict[str, object]:
    required_name = "operator-evidence-commands.txt"
    if path.is_dir():
        required_path = path / required_name
        related_paths = sorted(str(candidate) for candidate in path.rglob(required_name) if candidate.is_file())
        present = required_path.is_file()
        boundary_status = _operator_evidence_command_sheet_boundary_status(
            required_path.read_text(encoding="utf-8") if present else ""
        )
        return {
            "required": required,
            "present": present,
            "valid": present and bool(boundary_status["operator_only_boundary_present"]),
            "required_path": str(required_path),
            "matched_paths": [str(required_path)] if present else [],
            "related_paths": related_paths,
            "error": "",
            **boundary_status,
        }
    if path.is_file() and path.suffix.lower() == ".zip":
        import zipfile

        with zipfile.ZipFile(path) as archive:
            names = sorted(
                name
                for name in archive.namelist()
                if name == required_name or name.endswith(f"/{required_name}")
            )
            top_level_names = [
                name
                for name in names
                if name == required_name or len([part for part in name.split("/") if part]) == 2
            ]
            boundary_text = ""
            if top_level_names:
                try:
                    boundary_text = archive.read(top_level_names[0]).decode("utf-8")
                except (KeyError, UnicodeDecodeError):
                    boundary_text = ""
            boundary_status = _operator_evidence_command_sheet_boundary_status(boundary_text)
        return {
            "required": required,
            "present": bool(top_level_names),
            "valid": bool(top_level_names) and bool(boundary_status["operator_only_boundary_present"]),
            "required_path": f"*/{required_name}",
            "matched_paths": top_level_names,
            "related_paths": names,
            "error": "",
            **boundary_status,
        }
    return {
        "required": required,
        "present": False,
        "valid": False,
        "required_path": required_name,
        "matched_paths": [],
        "related_paths": [],
        "error": "",
        **_operator_evidence_command_sheet_boundary_status(""),
    }



def _launch_evidence_execution_checklist_status(text: str) -> dict[str, object]:
    required_phrases = [
        "Palette web-labeling launch evidence execution checklist",
        "Operator-only checklist",
        "record-zarr-backup-evidence",
        "record-browser-response-security-evidence",
        "record-identity-source-evidence",
        "record-browser-smoke-evidence",
        "record-disposable-zarr-mutation-smoke-evidence",
        "apply-operator-evidence-templates",
        "inspect-handoff --path PACKAGE_PATH --require-shareable",
        "labeler_links_safe_to_share=true",
    ]
    missing_phrases = [phrase for phrase in required_phrases if phrase not in text]
    return {
        "checklist_contract_present": not missing_phrases,
        "checklist_required_phrases": required_phrases,
        "checklist_missing_phrases": missing_phrases,
    }



def _inspect_handoff_launch_evidence_execution_checklist(
    path: Path, *, required: bool = True
) -> dict[str, object]:
    required_name = "launch-evidence-execution-checklist.txt"
    if path.is_dir():
        required_path = path / required_name
        related_paths = sorted(str(candidate) for candidate in path.rglob(required_name) if candidate.is_file())
        present = required_path.is_file()
        checklist_status = _launch_evidence_execution_checklist_status(
            required_path.read_text(encoding="utf-8") if present else ""
        )
        return {
            "required": required,
            "present": present,
            "valid": present and bool(checklist_status["checklist_contract_present"]),
            "required_path": str(required_path),
            "matched_paths": [str(required_path)] if present else [],
            "related_paths": related_paths,
            "error": "",
            **checklist_status,
        }
    if path.is_file() and path.suffix.lower() == ".zip":
        import zipfile

        with zipfile.ZipFile(path) as archive:
            names = sorted(
                name
                for name in archive.namelist()
                if name == required_name or name.endswith(f"/{required_name}")
            )
            top_level_names = [
                name
                for name in names
                if name == required_name or len([part for part in name.split("/") if part]) == 2
            ]
            checklist_text = ""
            if top_level_names:
                try:
                    checklist_text = archive.read(top_level_names[0]).decode("utf-8")
                except (KeyError, UnicodeDecodeError):
                    checklist_text = ""
            checklist_status = _launch_evidence_execution_checklist_status(checklist_text)
        return {
            "required": required,
            "present": bool(top_level_names),
            "valid": bool(top_level_names) and bool(checklist_status["checklist_contract_present"]),
            "required_path": f"*/{required_name}",
            "matched_paths": top_level_names,
            "related_paths": names,
            "error": "",
            **checklist_status,
        }
    return {
        "required": required,
        "present": False,
        "valid": False,
        "required_path": required_name,
        "matched_paths": [],
        "related_paths": [],
        "error": "",
        **_launch_evidence_execution_checklist_status(""),
    }


def _load_operator_evidence_template_from_directory(root: Path, template_path: str) -> tuple[dict[str, object] | None, bool, bool, str]:
    candidates: list[Path] = []
    if template_path:
        raw = Path(template_path)
        candidates.append(raw if raw.is_absolute() else root / raw)
        candidates.append(root / raw.name)
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file():
            continue
        try:
            loaded = json.loads(candidate.read_text(encoding="utf-8"))
            return (loaded if isinstance(loaded, dict) else {}, True, isinstance(loaded, dict), "")
        except (OSError, json.JSONDecodeError) as exc:
            return None, True, False, str(exc)
    return None, False, False, ""



def _load_operator_evidence_template_from_zip(archive, template_path: str) -> tuple[dict[str, object] | None, bool, bool, str]:
    basename = Path(str(template_path or "")).name
    if not basename:
        return None, False, False, ""
    matches = sorted(
        name
        for name in archive.namelist()
        if name == basename or name.endswith(f"/{basename}")
    )
    if not matches:
        return None, False, False, ""
    chosen = min(matches, key=lambda name: (len([part for part in name.split("/") if part]), name))
    try:
        loaded = json.loads(archive.read(chosen).decode("utf-8"))
        return (loaded if isinstance(loaded, dict) else {}, True, isinstance(loaded, dict), "")
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, True, False, str(exc)



def _operator_evidence_commands_public_summary(
    operator_evidence_commands: Mapping[str, object],
) -> dict[str, object]:
    required = bool(operator_evidence_commands.get("required"))
    present = bool(operator_evidence_commands.get("present"))
    valid = bool(operator_evidence_commands.get("valid"))
    boundary_present = bool(operator_evidence_commands.get("operator_only_boundary_present"))
    missing_phrases = [
        str(phrase)
        for phrase in (
            operator_evidence_commands.get("operator_only_boundary_missing_phrases")
            if isinstance(
                operator_evidence_commands.get("operator_only_boundary_missing_phrases"),
                list,
            )
            else []
        )
        if str(phrase)
    ]
    matched_paths = (
        operator_evidence_commands.get("matched_paths")
        if isinstance(operator_evidence_commands.get("matched_paths"), list)
        else []
    )
    related_paths = (
        operator_evidence_commands.get("related_paths")
        if isinstance(operator_evidence_commands.get("related_paths"), list)
        else []
    )
    blocking_reason_id = ""
    if required and not valid:
        blocking_reason_id = (
            "operator_evidence_commands_missing"
            if not present
            else (
                "operator_evidence_commands_boundary_missing"
                if not boundary_present
                else "operator_evidence_commands_invalid"
            )
        )
    return {
        "schema": "palette.web_labeling_operator_evidence_commands_summary.v1",
        "required": required,
        "present": present,
        "valid": valid,
        "operator_only_boundary_present": boundary_present,
        "operator_only_boundary_missing_phrases": missing_phrases,
        "operator_only_boundary_missing_phrase_count": len(missing_phrases),
        "matched_path_count": len(matched_paths),
        "related_path_count": len(related_paths),
        "required_path": str(operator_evidence_commands.get("required_path") or ""),
        "blocking_reason_id": blocking_reason_id,
    }



def _launch_evidence_execution_checklist_public_summary(
    launch_evidence_execution_checklist: Mapping[str, object],
) -> dict[str, object]:
    required = bool(launch_evidence_execution_checklist.get("required"))
    present = bool(launch_evidence_execution_checklist.get("present"))
    valid = bool(launch_evidence_execution_checklist.get("valid"))
    contract_present = bool(
        launch_evidence_execution_checklist.get("checklist_contract_present")
    )
    missing_phrases = [
        str(phrase)
        for phrase in (
            launch_evidence_execution_checklist.get("checklist_missing_phrases")
            if isinstance(
                launch_evidence_execution_checklist.get("checklist_missing_phrases"),
                list,
            )
            else []
        )
        if str(phrase)
    ]
    matched_paths = (
        launch_evidence_execution_checklist.get("matched_paths")
        if isinstance(launch_evidence_execution_checklist.get("matched_paths"), list)
        else []
    )
    related_paths = (
        launch_evidence_execution_checklist.get("related_paths")
        if isinstance(launch_evidence_execution_checklist.get("related_paths"), list)
        else []
    )
    blocking_reason_id = ""
    if required and not valid:
        blocking_reason_id = (
            "launch_evidence_execution_checklist_missing"
            if not present
            else (
                "launch_evidence_execution_checklist_incomplete"
                if not contract_present
                else "launch_evidence_execution_checklist_invalid"
            )
        )
    return {
        "schema": "palette.web_labeling_launch_evidence_execution_checklist_summary.v1",
        "required": required,
        "present": present,
        "valid": valid,
        "checklist_contract_present": contract_present,
        "checklist_missing_phrases": missing_phrases,
        "checklist_missing_phrase_count": len(missing_phrases),
        "matched_path_count": len(matched_paths),
        "related_path_count": len(related_paths),
        "required_path": str(launch_evidence_execution_checklist.get("required_path") or ""),
        "blocking_reason_id": blocking_reason_id,
    }



def _launch_evidence_execution_checklist_inspection_target() -> dict[str, object]:
    return {
        "shareability_launch_evidence_execution_checklist_required": True,
        "shareability_launch_evidence_execution_checklist_file": (
            "launch-evidence-execution-checklist.txt"
        ),
        "shareability_launch_evidence_execution_checklist_field": (
            "launch_evidence_execution_checklist"
        ),
        "shareability_launch_evidence_execution_checklist_summary_field": (
            "launch_evidence_execution_checklist_summary"
        ),
        "shareability_launch_evidence_execution_checklist_top_level_fields": [
            "launch_evidence_execution_checklist_required",
            "launch_evidence_execution_checklist_present",
            "launch_evidence_execution_checklist_valid",
            "launch_evidence_execution_checklist_contract_present",
            "launch_evidence_execution_checklist_missing_phrases",
            "launch_evidence_execution_checklist_blocking_reason_id",
        ],
        "shareability_launch_evidence_execution_checklist_required_phrases": [
            "Palette web-labeling launch evidence execution checklist",
            "Operator-only checklist",
            "record-zarr-backup-evidence",
            "record-browser-response-security-evidence",
            "record-identity-source-evidence",
            "record-browser-smoke-evidence",
            "record-disposable-zarr-mutation-smoke-evidence",
            "apply-operator-evidence-templates",
            "inspect-handoff --path PACKAGE_PATH --require-shareable",
            "labeler_links_safe_to_share=true",
        ],
        "shareability_launch_evidence_execution_checklist_blocking_reason_ids": [
            "launch_evidence_execution_checklist_missing",
            "launch_evidence_execution_checklist_incomplete",
            "launch_evidence_execution_checklist_invalid",
        ],
    }


