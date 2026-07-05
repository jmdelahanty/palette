"""User-handoff validation/evidence file helpers for web labeling."""

from __future__ import annotations

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


