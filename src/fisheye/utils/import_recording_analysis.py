#!/usr/bin/env python3
"""Single-recording analysis import helpers (create archive + metadata/stimulus import).

This module intentionally excludes detect/refine orchestration. Use
`fisheye.utils.run_recording_analysis_pipeline` for the full pipeline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional

import zarr

from fisheye.shared.acquisition_video_streams import write_acquisition_video_stream_inventory
from fisheye.shared.acquisition_publication_status import (
    ACQUISITION_AUTHORITY_PUBLISHED,
    load_acquisition_authority_publication_status,
)
from fisheye.shared.experiment_setup import (
    build_experiment_setup_record,
    publish_experiment_setup,
)
from fisheye.shared.import_source_fingerprint import optional_source_stat_fingerprint_attrs
from fisheye.shared.import_video_metadata import (
    probe_video_metadata,
    publish_external_video_acquisition_authority,
    write_video_metadata,
)
from fisheye.shared.recording_preflight import preflight_gate_reason
from fisheye.shared.subject_metadata import (
    publish_subject_metadata,
    read_h5_subject_metadata,
)


@dataclass
class RecordingAnalysisPlan:
    recording_dir: Path
    h5_path: Optional[Path]
    cam_video: Path
    zarr_path: Path


@dataclass
class RecordingImportOptions:
    import_video_metadata: bool
    video_metadata_overwrite: bool
    import_stimulus: bool
    stimulus_always: bool
    stimulus_run_name: Optional[str]
    stimulus_overwrite: bool
    stimulus_quiet: bool
    allow_preflight_failures: bool = False


@dataclass
class RecordingImportResult:
    ok: bool
    failed_step: Optional[str] = None
    error: Optional[str] = None
    returncode: Optional[int] = None


# Backward-compatible aliases during refactor window.
RecordingAnalysisOptions = RecordingImportOptions
RecordingAnalysisResult = RecordingImportResult


EventLogger = Callable[[str], None]


def _load_recording_manifest(recording_dir: Path) -> dict[str, object]:
    manifest_path = recording_dir / "recording_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _manifest_text(manifest: dict[str, object], *keys: str) -> Optional[str]:
    nested_meta = manifest.get("meta")
    candidates: list[object] = []
    for key in keys:
        candidates.append(manifest.get(key))
        if isinstance(nested_meta, dict):
            candidates.append(nested_meta.get(key))
    for value in candidates:
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8", "ignore")
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _producer_video_metadata(plan: RecordingAnalysisPlan) -> dict[str, Any]:
    """Return the organizer's producer-declared full-video fields, when present."""

    manifest = _load_recording_manifest(plan.recording_dir)
    video_streams = manifest.get("video_streams")
    if not isinstance(video_streams, Mapping):
        return {}
    streams = video_streams.get("streams")
    if not isinstance(streams, Mapping):
        return {}
    selected_key: str | None = None
    selected: Mapping[str, Any] | None = None
    wanted = plan.cam_video.expanduser().resolve()
    for key, candidate in streams.items():
        if not isinstance(key, str) or not isinstance(candidate, Mapping):
            continue
        raw_video = candidate.get("video")
        if isinstance(raw_video, str) and raw_video.strip():
            path = Path(raw_video)
            if not path.is_absolute():
                path = plan.recording_dir / path
            if path.expanduser().resolve() == wanted:
                selected_key, selected = key, candidate
                break
    if selected is None:
        fallback = streams.get("full")
        if isinstance(fallback, Mapping):
            selected_key, selected = "full", fallback
    if selected is None or selected_key is None:
        return {}
    source_prefix = f"recording_manifest.video_streams.streams.{selected_key}"
    producer: dict[str, Any] = {
        "_source": source_prefix,
        "total_frames": selected.get("frame_count"),
        "fps": selected.get("frame_rate"),
        "width": selected.get("width"),
        "height": selected.get("height"),
        "codec": selected.get("codec"),
    }
    return {key: value for key, value in producer.items() if value is not None}


def stimulus_runs_present(zarr_path: Path) -> bool:
    try:
        root = zarr.open(str(zarr_path), mode="r")
    except Exception:
        return False
    analysis = root.get("analysis")
    if analysis is None:
        return False
    stim = analysis.get("stimulus_runs")
    if stim is None:
        return False
    try:
        return len(list(stim.group_keys())) > 0
    except Exception:
        return False


def ensure_analysis_archive(plan: RecordingAnalysisPlan) -> Optional[dict[str, object]]:
    plan.zarr_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if plan.zarr_path.exists() else "w"
    root = zarr.open_group(str(plan.zarr_path), mode=mode)
    attrs = dict(root.attrs)
    recording_id = plan.recording_dir.name
    manifest = _load_recording_manifest(plan.recording_dir)
    attrs["zarr_purpose"] = "analysis"
    attrs.setdefault("session_uuid", _manifest_text(manifest, "session_uuid") or recording_id)
    attrs.setdefault("recording_id", recording_id)
    attrs.setdefault("recording_name", _manifest_text(manifest, "recording_name") or recording_id)
    attrs.setdefault("recording_path", str(plan.recording_dir))
    attrs.setdefault("recording_type", _manifest_text(manifest, "recording_type") or "behavior")
    attrs.setdefault("recording_subtype", _manifest_text(manifest, "recording_subtype") or "free")
    attrs.setdefault("behavior_mode", _manifest_text(manifest, "behavior_mode") or "free")
    attrs.setdefault("artifact_schema_id", "recording_analysis_v1")
    for key in (
        "camera_id",
        "rig_id",
        "arena_id",
        "canvas_name",
        "protocol_name",
        "protocol_name_from_definition",
        "dish_design",
        "genotype",
        "dpf_at_acquisition",
        "num_dishes",
        "fish_per_dish",
        "session_start_iso8601_utc",
    ):
        value = _manifest_text(manifest, key)
        if value:
            if key == "camera_id":
                existing = attrs.get(key)
                if existing not in (None, "", value):
                    raise ValueError(
                        "Existing archive camera_id conflicts with recording_manifest.json."
                    )
            attrs.setdefault(key, value)
    manifest_recording_id = _manifest_text(manifest, "recording_id")
    if manifest_recording_id and manifest_recording_id != recording_id:
        attrs.setdefault("organizer_recording_id", manifest_recording_id)
    if manifest:
        attrs.setdefault("recording_manifest_path", str(plan.recording_dir / "recording_manifest.json"))
    attrs.setdefault("source_video", plan.cam_video.name)
    attrs.setdefault("source_video_path", str(plan.cam_video))
    attrs.setdefault("source_path", str(plan.cam_video))
    if plan.h5_path is None:
        attrs["experiment_context_status"] = "absent"
        attrs["experiment_context_source"] = "none"
        attrs["stimulus_runs_available"] = False
        attrs["experiment_context_status_detail"] = (
            "No H5/protocol source was provided; stimulus-dependent analyses are unavailable."
        )
    else:
        attrs["experiment_context_status"] = "present"
        attrs["experiment_context_source"] = "h5"
        attrs["source_h5"] = plan.h5_path.name
        attrs["source_h5_path"] = str(plan.h5_path)
        attrs.pop("experiment_context_status_detail", None)
    root.attrs.put(attrs)
    return write_acquisition_video_stream_inventory(root, plan.recording_dir, manifest)


def apply_video_metadata(plan: RecordingAnalysisPlan, *, overwrite: bool) -> dict[str, int]:
    root = zarr.open_group(str(plan.zarr_path), mode="r+")
    try:
        publication = load_acquisition_authority_publication_status(root)
        authority_published = publication.status == ACQUISITION_AUTHORITY_PUBLISHED
    except Exception:
        authority_published = False
    meta = probe_video_metadata(
        plan.cam_video,
        producer_metadata=_producer_video_metadata(plan),
    )
    updates = write_video_metadata(
        root,
        meta,
        overwrite=bool(overwrite or not authority_published),
        import_purpose="analysis",
        recording_path=plan.recording_dir,
    )
    publish_external_video_acquisition_authority(root)
    h5_updates = _write_source_h5_fingerprint(root, plan.h5_path, overwrite=overwrite)
    return {
        "root_attrs_updated": len(updates.get("root", {})) + h5_updates["root_attrs_updated"],
        "raw_video_attrs_updated": len(updates.get("raw_video", {})) + h5_updates["raw_video_attrs_updated"],
    }


def _set_attr_if_needed(attrs: object, key: str, value: object, *, overwrite: bool) -> bool:
    if value is None:
        return False
    try:
        current = attrs.get(key)  # type: ignore[attr-defined]
    except Exception:
        current = None
    if overwrite or current in (None, ""):
        attrs[key] = value  # type: ignore[index]
        return True
    return False


def _write_source_h5_fingerprint(
    root: zarr.Group,
    h5_path: Path | None,
    *,
    overwrite: bool,
) -> dict[str, int]:
    if h5_path is None:
        return {"root_attrs_updated": 0, "raw_video_attrs_updated": 0}
    raw = root.require_group("raw_video")
    payload = {
        "source_h5": h5_path.name,
        "source_h5_path": str(h5_path),
        **optional_source_stat_fingerprint_attrs(h5_path, attr_prefix="source_h5"),
    }
    root_updated = 0
    raw_updated = 0
    for key, value in payload.items():
        if _set_attr_if_needed(root.attrs, key, value, overwrite=overwrite):
            root_updated += 1
        if _set_attr_if_needed(raw.attrs, key, value, overwrite=overwrite):
            raw_updated += 1
    return {"root_attrs_updated": root_updated, "raw_video_attrs_updated": raw_updated}


def import_experiment_setup(plan: RecordingAnalysisPlan) -> Optional[dict[str, Any]]:
    """Publish canonical subject metadata and experiment setup from the H5."""

    if plan.h5_path is None:
        return None
    subject_metadata = read_h5_subject_metadata(plan.h5_path)
    if not subject_metadata:
        return None
    root = zarr.open_group(str(plan.zarr_path), mode="r+")
    subject_authority = publish_subject_metadata(
        root,
        subject_metadata,
        source_h5_path=plan.h5_path,
    )
    record = build_experiment_setup_record(
        subject_metadata,
        source_h5_path=plan.h5_path,
        subject_metadata_sha256=subject_authority.record_sha256,
        subject_metadata_ref=subject_authority.group_path,
    )
    resolved = publish_experiment_setup(
        root,
        record,
        source_h5_path=plan.h5_path,
    )
    return {
        "run_name": resolved.run_name,
        "group_path": resolved.group_path,
        "record_sha256": resolved.record_sha256,
        "expected_subject_count": resolved.expected_subject_count,
        "assigned_subject_count": resolved.assigned_subject_count,
        "subject_metadata_run": subject_authority.run_name,
        "subject_metadata_path": subject_authority.group_path,
        "subject_metadata_sha256": subject_authority.record_sha256,
    }


def _log(logger: Optional[Callable[..., None]], event: str, **fields: object) -> None:
    if logger is not None:
        logger(event, **fields)


def run_stimulus_import(plan: RecordingAnalysisPlan, opts: RecordingImportOptions) -> tuple[bool, int, List[str]]:
    if plan.h5_path is None:
        return False, 2, ["missing_h5_for_stimulus_import"]
    cmd = [
        sys.executable,
        "-m",
        "fisheye.analysis.import_stimulus_to_zarr",
        str(plan.h5_path),
        str(plan.zarr_path),
    ]
    if opts.stimulus_run_name:
        cmd.extend(["--run-name", opts.stimulus_run_name])
    if opts.stimulus_overwrite:
        cmd.append("--overwrite")
    if opts.stimulus_quiet:
        cmd.append("--quiet")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0, result.returncode, cmd


def process_recording_import(
    plan: RecordingAnalysisPlan,
    opts: RecordingImportOptions,
    *,
    logger: Optional[Callable[..., None]] = None,
) -> RecordingImportResult:
    gate_reason = preflight_gate_reason(
        plan.recording_dir,
        allow_failures=bool(opts.allow_preflight_failures),
    )
    if gate_reason is not None:
        _log(
            logger,
            "preflight_gate_blocked",
            recording_dir=str(plan.recording_dir),
            reason=gate_reason,
        )
        return RecordingImportResult(ok=False, failed_step="preflight_gate", error=gate_reason)

    try:
        stream_inventory = ensure_analysis_archive(plan)
    except Exception as exc:
        return RecordingImportResult(ok=False, failed_step="ensure_analysis_archive", error=str(exc))
    if isinstance(stream_inventory, dict):
        _log(
            logger,
            "acquisition_video_streams_imported",
            recording_dir=str(plan.recording_dir),
            zarr_path=str(plan.zarr_path),
            stream_count=stream_inventory.get("stream_count"),
            stream_keys=stream_inventory.get("stream_keys"),
            crop_stream_available=stream_inventory.get("crop_stream_available"),
            inventory_status=stream_inventory.get("inventory_status"),
        )

    if opts.import_video_metadata:
        try:
            updates = apply_video_metadata(plan, overwrite=bool(opts.video_metadata_overwrite))
        except Exception as exc:
            return RecordingImportResult(ok=False, failed_step="import_video_metadata", error=str(exc))
        _log(
            logger,
            "video_metadata_imported",
            recording_dir=str(plan.recording_dir),
            zarr_path=str(plan.zarr_path),
            root_attrs_updated=updates["root_attrs_updated"],
            raw_video_attrs_updated=updates["raw_video_attrs_updated"],
            overwrite=bool(opts.video_metadata_overwrite),
        )

    if plan.h5_path is not None:
        try:
            setup = import_experiment_setup(plan)
        except Exception as exc:
            return RecordingImportResult(ok=False, failed_step="import_experiment_setup", error=str(exc))
        if setup is not None:
            _log(
                logger,
                "experiment_setup_imported",
                recording_dir=str(plan.recording_dir),
                zarr_path=str(plan.zarr_path),
                **setup,
            )

    if opts.import_stimulus:
        if plan.h5_path is None:
            return RecordingImportResult(
                ok=False,
                failed_step="import_stimulus_to_zarr",
                returncode=2,
                error="stimulus import requested but no H5/protocol source is available",
            )
        stim_present = stimulus_runs_present(plan.zarr_path)
        if stim_present and not opts.stimulus_always:
            _log(
                logger,
                "stimulus_skipped",
                recording_dir=str(plan.recording_dir),
                zarr_path=str(plan.zarr_path),
                reason="stimulus_runs already present",
            )
        else:
            stim_ok, stim_rc, stim_cmd = run_stimulus_import(plan, opts)
            _log(
                logger,
                "stimulus_result",
                recording_dir=str(plan.recording_dir),
                zarr_path=str(plan.zarr_path),
                returncode=int(stim_rc),
                cmd=stim_cmd,
            )
            if not stim_ok:
                return RecordingImportResult(
                    ok=False,
                    failed_step="import_stimulus_to_zarr",
                    returncode=int(stim_rc),
                    error="stimulus import failed",
                )

    return RecordingImportResult(ok=True)


# Backward-compatible alias name.
process_recording_analysis = process_recording_import


def resolve_single_recording_plan(
    *,
    recording_dir: Path,
    video: Optional[Path] = None,
    h5: Optional[Path] = None,
    output: Optional[Path] = None,
    require_h5: bool = True,
) -> RecordingAnalysisPlan:
    rec_dir = recording_dir.expanduser().resolve()
    if not rec_dir.exists() or not rec_dir.is_dir():
        raise ValueError(f"recording_dir not found: {rec_dir}")

    if video is None:
        cams_dir = rec_dir / "cams"
        mp4s = sorted(cams_dir.glob("*.mp4"))
        if not mp4s:
            raise ValueError(f"no .mp4 files found under {cams_dir}")
        if len(mp4s) > 1:
            raise ValueError(
                f"multiple .mp4 files found under {cams_dir}; pass --video explicitly for single-recording mode"
            )
        cam_video = mp4s[0].resolve()
    else:
        cam_video = video.expanduser().resolve()
        if not cam_video.exists() or not cam_video.is_file():
            raise ValueError(f"video not found: {cam_video}")

    if h5 is None:
        raw_dir = rec_dir / "raw"
        h5s = sorted(raw_dir.glob("*.h5"))
        if not h5s:
            if require_h5:
                raise ValueError(f"no .h5 files found under {raw_dir}")
            h5_path = None
        if len(h5s) > 1:
            if require_h5:
                raise ValueError(
                    f"multiple .h5 files found under {raw_dir}; pass --h5 explicitly for single-recording mode"
                )
            h5_path = None
        elif h5s:
            h5_path = h5s[0].resolve()
    else:
        h5_path = h5.expanduser().resolve()
        if not h5_path.exists() or not h5_path.is_file():
            raise ValueError(f"h5 not found: {h5_path}")

    if output is None:
        zarr_path = (rec_dir / "zarr" / f"{rec_dir.name}_analysis.zarr").resolve()
    else:
        zarr_path = output.expanduser().resolve()

    return RecordingAnalysisPlan(
        recording_dir=rec_dir,
        h5_path=h5_path,
        cam_video=cam_video,
        zarr_path=zarr_path,
    )


def _build_options(args: argparse.Namespace) -> RecordingImportOptions:
    return RecordingImportOptions(
        import_video_metadata=bool(args.import_video_metadata),
        video_metadata_overwrite=bool(args.video_metadata_overwrite),
        import_stimulus=bool(args.import_stimulus),
        stimulus_always=bool(args.stimulus_always),
        stimulus_run_name=args.stimulus_run_name,
        stimulus_overwrite=bool(args.stimulus_overwrite),
        stimulus_quiet=bool(args.stimulus_quiet),
        allow_preflight_failures=bool(args.allow_preflight_failures),
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Import/create analysis zarr for one recording (archive + metadata + optional stimulus).",
    )
    parser.add_argument("--recording-dir", type=Path, required=True, help="Recording directory to process.")
    parser.add_argument("--video", type=Path, help="Optional explicit camera video path.")
    parser.add_argument("--h5", type=Path, help="Optional explicit stimulus H5 path.")
    parser.add_argument("--output", type=Path, help="Optional explicit analysis zarr output path.")

    parser.add_argument("--apply", action="store_true", help="Execute import steps.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved plan only.")

    parser.add_argument(
        "--import-video-metadata",
        dest="import_video_metadata",
        action="store_true",
        help="Import source-video metadata into root/raw_video attrs (default).",
    )
    parser.add_argument(
        "--no-import-video-metadata",
        dest="import_video_metadata",
        action="store_false",
        help="Skip source-video metadata import.",
    )
    parser.add_argument(
        "--video-metadata-overwrite",
        action="store_true",
        help="Overwrite existing source-video metadata attrs when importing.",
    )

    parser.add_argument(
        "--import-stimulus",
        dest="import_stimulus",
        action="store_true",
        help="Import H5 stimulus metadata into analysis/stimulus_runs (default).",
    )
    parser.add_argument(
        "--no-import-stimulus",
        dest="import_stimulus",
        action="store_false",
        help="Skip stimulus import.",
    )
    parser.add_argument(
        "--recording-only",
        action="store_true",
        help="Process a camera-video-only recording without requiring an H5/protocol source.",
    )
    parser.set_defaults(import_stimulus=True, import_video_metadata=True)
    parser.add_argument("--stimulus-always", action="store_true", help="Run stimulus import even if runs exist.")
    parser.add_argument("--stimulus-run-name", type=str, help="Optional stimulus run name.")
    parser.add_argument("--stimulus-overwrite", action="store_true", help="Overwrite existing stimulus run name.")
    parser.add_argument("--stimulus-quiet", action="store_true", help="Suppress verbose stimulus import output.")
    parser.add_argument(
        "--allow-preflight-failures",
        action="store_true",
        help="Proceed even if recording_manifest.json marks preflight.status=fail.",
    )

    args = parser.parse_args(argv)
    if args.recording_only:
        args.import_stimulus = False
    if not args.apply:
        args.dry_run = True

    try:
        plan = resolve_single_recording_plan(
            recording_dir=args.recording_dir,
            video=args.video,
            h5=args.h5,
            output=args.output,
            require_h5=bool(args.import_stimulus),
        )
    except ValueError as exc:
        print(f"Plan resolution failed: {exc}")
        return 1

    print("Single recording import plan")
    print(f"  recording_dir: {plan.recording_dir}")
    print(f"  video: {plan.cam_video}")
    print(f"  h5: {plan.h5_path if plan.h5_path is not None else 'none (recording-only)'}")
    print(f"  output: {plan.zarr_path}")
    print(f"  import_video_metadata: {bool(args.import_video_metadata)}")
    print(f"  import_stimulus: {bool(args.import_stimulus)}")
    print(f"  allow_preflight_failures: {bool(args.allow_preflight_failures)}")
    if args.dry_run:
        print("Dry run: no changes were made.")
        return 0

    opts = _build_options(args)
    result = process_recording_import(plan, opts, logger=None)

    if not result.ok:
        print(
            f"Failed: step={result.failed_step or 'unknown'}"
            + (f" returncode={result.returncode}" if result.returncode is not None else "")
            + (f" error={result.error}" if result.error else "")
        )
        return 1

    print("Success: analysis import completed.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
