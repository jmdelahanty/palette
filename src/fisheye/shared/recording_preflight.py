from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from fisheye.shared.source_recording_identity import load_strict_json_object

PRECHECK_NOT_RUN = "not_run"
PRECHECK_PASS = "pass"
PRECHECK_WARN = "warn"
PRECHECK_FAIL = "fail"


@dataclass(frozen=True)
class RecordedPreflight:
    manifest_path: Optional[Path]
    manifest_exists: bool
    status: str
    checked_at_utc: Optional[str] = None
    video_status: Optional[str] = None
    video_media_status: Optional[str] = None
    video_tooling_status: Optional[str] = None
    h5_status: Optional[str] = None
    h5_core_status: Optional[str] = None
    h5_optional_status: Optional[str] = None
    h5_tooling_status: Optional[str] = None
    error: Optional[str] = None


def _section_status_from_video(payload: Mapping[str, Any]) -> str:
    return _combine_statuses(*(payload.get(key, PRECHECK_NOT_RUN) for key in (
        "status", "media_status", "tooling_status",
    )))


def _section_status_from_h5(payload: Mapping[str, Any]) -> str:
    return _combine_statuses(*(payload.get(key, PRECHECK_NOT_RUN) for key in (
        "status", "core_status", "optional_status", "tooling_status",
    )))


def _combine_statuses(*statuses: str) -> str:
    allowed = {PRECHECK_NOT_RUN, PRECHECK_PASS, PRECHECK_WARN, PRECHECK_FAIL, "error", "skip"}
    if any(type(status) is not str or status not in allowed for status in statuses):
        raise ValueError("preflight contains an invalid diagnostic status")
    effective = [status for status in statuses if status not in {PRECHECK_NOT_RUN, "skip"}]
    if not effective:
        return PRECHECK_NOT_RUN
    if PRECHECK_FAIL in effective or "error" in effective:
        return PRECHECK_FAIL
    if PRECHECK_WARN in effective:
        return PRECHECK_WARN
    return PRECHECK_PASS


def default_preflight_payload() -> dict[str, Any]:
    return {
        "status": PRECHECK_NOT_RUN,
        "checked_at_utc": None,
        "video": None,
        "h5": None,
    }


def build_video_preflight_payload(
    *,
    status: str,
    media_status: str,
    tooling_status: str,
    videos_scanned: int,
    finding_codes: list[str],
    error: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "media_status": media_status,
        "tooling_status": tooling_status,
        "videos_scanned": int(videos_scanned),
        "finding_codes": list(finding_codes),
        "error": error,
    }


def build_h5_preflight_payload(
    *,
    status: str,
    core_status: str,
    optional_status: str,
    tooling_status: str,
    finding_codes: list[str],
    error: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "core_status": core_status,
        "optional_status": optional_status,
        "tooling_status": tooling_status,
        "finding_codes": list(finding_codes),
        "error": error,
    }


def build_manifest_preflight_payload(
    *,
    checked_at_utc: Optional[str],
    video: Optional[Mapping[str, Any]] = None,
    h5: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    video_payload = dict(video) if video is not None else None
    h5_payload = dict(h5) if h5 is not None else None
    return {
        "status": _combine_statuses(
            _section_status_from_video(video_payload or {}),
            _section_status_from_h5(h5_payload or {}),
        ),
        "checked_at_utc": checked_at_utc,
        "video": video_payload,
        "h5": h5_payload,
    }


def read_manifest_payload(recording_dir: Path) -> Optional[dict[str, Any]]:
    manifest_path = recording_dir / 'recording_manifest.json'
    if not manifest_path.exists():
        return None
    return load_strict_json_object(manifest_path)


def read_recorded_preflight(recording_dir: Path) -> RecordedPreflight:
    manifest_path = recording_dir / 'recording_manifest.json'
    if not manifest_path.exists():
        return RecordedPreflight(manifest_path=manifest_path, manifest_exists=False, status=PRECHECK_NOT_RUN)

    try:
        payload = load_strict_json_object(manifest_path)
    except Exception as exc:
        return RecordedPreflight(
            manifest_path=manifest_path,
            manifest_exists=True,
            status=PRECHECK_FAIL,
            error=f'failed to read manifest JSON: {exc}',
        )

    preflight = payload.get('preflight')
    if preflight is None:
        return RecordedPreflight(manifest_path=manifest_path, manifest_exists=True, status=PRECHECK_NOT_RUN)

    try:
        if not isinstance(preflight, dict):
            raise ValueError('preflight must be a JSON object or null')
        video = preflight.get('video')
        h5 = preflight.get('h5')
        for name, section in (("preflight", preflight), ("video", video), ("h5", h5)):
            if section is None:
                continue
            if not isinstance(section, dict):
                raise ValueError(f'{name} must be a JSON object or null')
            if section.get('error') is not None:
                raise ValueError(f'{name} recorded a diagnostic error: {section["error"]}')
        status = _combine_statuses(
            preflight.get('status', PRECHECK_NOT_RUN),
            _section_status_from_video(video or {}),
            _section_status_from_h5(h5 or {}),
        )
    except ValueError as exc:
        return RecordedPreflight(
            manifest_path=manifest_path, manifest_exists=True,
            status=PRECHECK_FAIL, error=str(exc),
        )
    return RecordedPreflight(
        manifest_path=manifest_path,
        manifest_exists=True,
        status=status or PRECHECK_NOT_RUN,
        checked_at_utc=str(preflight.get('checked_at_utc')) if preflight.get('checked_at_utc') else None,
        video_status=_section_status_from_video(video or {}) if video is not None else None,
        video_media_status=str(video.get('media_status')) if video and video.get('media_status') else None,
        video_tooling_status=str(video.get('tooling_status')) if video and video.get('tooling_status') else None,
        h5_status=_section_status_from_h5(h5 or {}) if h5 is not None else None,
        h5_core_status=str(h5.get('core_status')) if h5 and h5.get('core_status') else None,
        h5_optional_status=str(h5.get('optional_status')) if h5 and h5.get('optional_status') else None,
        h5_tooling_status=str(h5.get('tooling_status')) if h5 and h5.get('tooling_status') else None,
        error=str(preflight.get('error')) if preflight.get('error') else None,
    )


def preflight_gate_reason(
    recording_dir: Path,
) -> Optional[str]:
    preflight = read_recorded_preflight(recording_dir)
    if preflight.status != PRECHECK_FAIL:
        return None

    details: list[str] = []
    if preflight.video_media_status:
        details.append(f'video_media={preflight.video_media_status}')
    if preflight.h5_core_status:
        details.append(f'h5_core={preflight.h5_core_status}')
    if preflight.h5_optional_status:
        details.append(f'h5_optional={preflight.h5_optional_status}')
    if preflight.video_tooling_status:
        details.append(f'video_tooling={preflight.video_tooling_status}')
    if preflight.h5_tooling_status:
        details.append(f'h5_tooling={preflight.h5_tooling_status}')
    if preflight.error:
        details.append(preflight.error)
    suffix = f" ({', '.join(details)})" if details else ''
    return f'preflight failed for {recording_dir}{suffix}'
