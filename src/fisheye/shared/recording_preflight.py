from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

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
    status = str(payload.get("status") or "").strip()
    if status:
        return status
    media_status = str(payload.get("media_status") or "").strip()
    if media_status == PRECHECK_FAIL:
        return PRECHECK_FAIL
    if media_status == PRECHECK_WARN:
        return PRECHECK_WARN
    if media_status == PRECHECK_PASS:
        return PRECHECK_PASS
    return PRECHECK_NOT_RUN


def _section_status_from_h5(payload: Mapping[str, Any]) -> str:
    status = str(payload.get("status") or "").strip()
    if status:
        return status
    core_status = str(payload.get("core_status") or "").strip()
    if core_status == PRECHECK_FAIL:
        return PRECHECK_FAIL
    if core_status == PRECHECK_WARN:
        return PRECHECK_WARN
    if core_status == PRECHECK_PASS:
        return PRECHECK_PASS
    return PRECHECK_NOT_RUN


def _combine_statuses(*statuses: str) -> str:
    effective = [status for status in statuses if status and status != PRECHECK_NOT_RUN]
    if not effective:
        return PRECHECK_NOT_RUN
    if PRECHECK_FAIL in effective:
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
    payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'manifest root must be a JSON object: {manifest_path}')
    return payload


def read_recorded_preflight(recording_dir: Path) -> RecordedPreflight:
    manifest_path = recording_dir / 'recording_manifest.json'
    if not manifest_path.exists():
        return RecordedPreflight(manifest_path=manifest_path, manifest_exists=False, status=PRECHECK_NOT_RUN)

    try:
        payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    except Exception as exc:
        return RecordedPreflight(
            manifest_path=manifest_path,
            manifest_exists=True,
            status=PRECHECK_WARN,
            error=f'failed to read manifest JSON: {exc}',
        )

    if not isinstance(payload, dict):
        return RecordedPreflight(
            manifest_path=manifest_path,
            manifest_exists=True,
            status=PRECHECK_WARN,
            error='manifest root must be a JSON object',
        )

    preflight = payload.get('preflight')
    if not isinstance(preflight, dict):
        return RecordedPreflight(manifest_path=manifest_path, manifest_exists=True, status=PRECHECK_NOT_RUN)

    video = preflight.get('video') if isinstance(preflight.get('video'), dict) else None
    h5 = preflight.get('h5') if isinstance(preflight.get('h5'), dict) else None
    status = str(preflight.get('status') or '').strip() or _combine_statuses(
        _section_status_from_video(video or {}),
        _section_status_from_h5(h5 or {}),
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
    *,
    allow_failures: bool = False,
) -> Optional[str]:
    if allow_failures:
        return None
    preflight = read_recorded_preflight(recording_dir)
    if preflight.status != PRECHECK_FAIL:
        return None

    details: list[str] = []
    if preflight.video_media_status:
        details.append(f'video_media={preflight.video_media_status}')
    if preflight.h5_core_status:
        details.append(f'h5_core={preflight.h5_core_status}')
    suffix = f" ({', '.join(details)})" if details else ''
    return f'preflight failed for {recording_dir}{suffix}'
