from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, TextIO


_SUMMARY_MAX_BYTES = 16 * 1024 * 1024
_JSON_TOKEN_MAX_CHARS = 64 * 1024
_JSON_MAX_DEPTH = 32


class OrangeSyncEvidenceError(ValueError):
    pass


@dataclass(frozen=True)
class OrangeCropSyncEvidence:
    summary_path: Optional[Path]
    keyframe_path: Optional[Path]
    declared_output_kind: Optional[str] = None
    declared_stream_kind: Optional[str] = None
    declared_tuning: Optional[str] = None
    declared_frame_count: Optional[int] = None
    declared_packet_count: Optional[int] = None


@dataclass(frozen=True)
class OrangeSyncAssessment:
    status: str
    summary_path: Optional[str]
    keyframe_path: Optional[str]
    resolved_gop_length: Optional[int] = None
    frames_encoded: Optional[int] = None
    sidecar_total_frames: Optional[int] = None
    keyframe_count: Optional[int] = None
    error: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class _KeyframeSidecarSummary:
    total_frames: int
    keyframe_count: int


def _reject_constant(value: str) -> None:
    raise OrangeSyncEvidenceError(f"non-finite JSON value is not allowed: {value}")


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise OrangeSyncEvidenceError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _load_summary(path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise OrangeSyncEvidenceError(f"summary is unavailable: {exc}") from exc
    if size > _SUMMARY_MAX_BYTES:
        raise OrangeSyncEvidenceError(
            f"summary exceeds {_SUMMARY_MAX_BYTES} byte validation bound"
        )
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OrangeSyncEvidenceError(f"summary is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise OrangeSyncEvidenceError("summary JSON root is not an object")
    return payload


class _StreamingJson:
    """Small bounded-memory JSON reader for producer keyframe sidecars."""

    def __init__(self, fh: TextIO, *, chunk_chars: int = 64 * 1024) -> None:
        self._fh = fh
        self._chunk_chars = chunk_chars
        self._buffer = ""
        self._pos = 0
        self._eof = False

    def _compact(self) -> None:
        if self._pos:
            self._buffer = self._buffer[self._pos :]
            self._pos = 0

    def _fill(self) -> bool:
        if self._eof:
            return False
        self._compact()
        chunk = self._fh.read(self._chunk_chars)
        if chunk:
            self._buffer += chunk
            return True
        self._eof = True
        return False

    def peek(self) -> str:
        while self._pos >= len(self._buffer):
            if not self._fill():
                return ""
        return self._buffer[self._pos]

    def take(self) -> str:
        char = self.peek()
        if char:
            self._pos += 1
        return char

    def skip_ws(self) -> None:
        while self.peek() in {" ", "\t", "\r", "\n"}:
            self._pos += 1

    def expect(self, expected: str) -> None:
        self.skip_ws()
        actual = self.take()
        if actual != expected:
            raise OrangeSyncEvidenceError(
                f"expected JSON token {expected!r}, found {actual or 'EOF'!r}"
            )

    def scalar(self) -> Any:
        self.skip_ws()
        token = ""
        in_string = self.peek() == '"'
        escaped = False
        while True:
            char = self.take()
            if not char:
                if in_string:
                    raise OrangeSyncEvidenceError("unterminated JSON string")
                break
            token += char
            if len(token) > _JSON_TOKEN_MAX_CHARS:
                raise OrangeSyncEvidenceError("JSON scalar exceeds validation bound")
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"' and len(token) > 1:
                    break
            elif self.peek() in {"", " ", "\t", "\r", "\n", ",", "]", "}"}:
                break
        try:
            return json.loads(token, parse_constant=_reject_constant)
        except (json.JSONDecodeError, TypeError) as exc:
            raise OrangeSyncEvidenceError(f"invalid JSON scalar {token!r}") from exc


def _skip_json_value(reader: _StreamingJson, *, depth: int) -> None:
    if depth > _JSON_MAX_DEPTH:
        raise OrangeSyncEvidenceError("JSON nesting exceeds validation bound")
    reader.skip_ws()
    token = reader.peek()
    if token == "{":
        reader.take()
        seen: set[str] = set()
        reader.skip_ws()
        if reader.peek() == "}":
            reader.take()
            return
        while True:
            key = reader.scalar()
            if not isinstance(key, str):
                raise OrangeSyncEvidenceError("JSON object key is not a string")
            if key in seen:
                raise OrangeSyncEvidenceError(f"duplicate JSON key: {key}")
            seen.add(key)
            reader.expect(":")
            _skip_json_value(reader, depth=depth + 1)
            reader.skip_ws()
            delimiter = reader.take()
            if delimiter == "}":
                return
            if delimiter != ",":
                raise OrangeSyncEvidenceError("malformed JSON object delimiter")
    elif token == "[":
        reader.take()
        reader.skip_ws()
        if reader.peek() == "]":
            reader.take()
            return
        while True:
            _skip_json_value(reader, depth=depth + 1)
            reader.skip_ws()
            delimiter = reader.take()
            if delimiter == "]":
                return
            if delimiter != ",":
                raise OrangeSyncEvidenceError("malformed JSON array delimiter")
    else:
        reader.scalar()


def _read_keyframe_array(reader: _StreamingJson) -> int:
    reader.expect("[")
    reader.skip_ws()
    expected = 0
    if reader.peek() == "]":
        reader.take()
        return 0
    while True:
        value = reader.scalar()
        if type(value) is not int:
            raise OrangeSyncEvidenceError(
                f"keyframe index {expected} is not an integer"
            )
        if value != expected:
            raise OrangeSyncEvidenceError(
                f"keyframe indices are not contiguous 0..N-1 at position {expected}: {value}"
            )
        expected += 1
        reader.skip_ws()
        delimiter = reader.take()
        if delimiter == "]":
            return expected
        if delimiter != ",":
            raise OrangeSyncEvidenceError("malformed keyframe_frames array delimiter")


def _read_keyframe_sidecar(path: Path) -> _KeyframeSidecarSummary:
    try:
        fh = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise OrangeSyncEvidenceError(f"keyframe sidecar is unavailable: {exc}") from exc
    try:
        with fh:
            reader = _StreamingJson(fh)
            reader.expect("{")
            seen: set[str] = set()
            total_frames: Optional[int] = None
            keyframe_count: Optional[int] = None
            reader.skip_ws()
            if reader.peek() == "}":
                reader.take()
            else:
                while True:
                    key = reader.scalar()
                    if not isinstance(key, str):
                        raise OrangeSyncEvidenceError(
                            "keyframe sidecar object key is not a string"
                        )
                    if key in seen:
                        raise OrangeSyncEvidenceError(f"duplicate JSON key: {key}")
                    seen.add(key)
                    reader.expect(":")
                    if key == "total_frames":
                        value = reader.scalar()
                        if type(value) is not int or value < 0:
                            raise OrangeSyncEvidenceError(
                                "sidecar total_frames is not a non-negative integer"
                            )
                        total_frames = value
                    elif key == "keyframe_frames":
                        keyframe_count = _read_keyframe_array(reader)
                    else:
                        _skip_json_value(reader, depth=1)
                    reader.skip_ws()
                    delimiter = reader.take()
                    if delimiter == "}":
                        break
                    if delimiter != ",":
                        raise OrangeSyncEvidenceError(
                            "malformed keyframe sidecar object delimiter"
                        )
            reader.skip_ws()
            if reader.peek():
                raise OrangeSyncEvidenceError("trailing data after keyframe sidecar JSON")
    except (UnicodeError, OSError) as exc:
        raise OrangeSyncEvidenceError(f"keyframe sidecar is unreadable: {exc}") from exc

    if total_frames is None:
        raise OrangeSyncEvidenceError("keyframe sidecar is missing total_frames")
    if keyframe_count is None:
        raise OrangeSyncEvidenceError("keyframe sidecar is missing keyframe_frames")
    return _KeyframeSidecarSummary(
        total_frames=total_frames,
        keyframe_count=keyframe_count,
    )


def _optional_text(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    return text or None


def _optional_nonnegative_int(value: object) -> Optional[int]:
    if type(value) is not int or value < 0:
        return None
    return value


def _combined_text(
    summary: dict[str, Any],
    key: str,
    declared: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    observed = _optional_text(summary.get(key))
    declared_normalized = _optional_text(declared)
    if observed and declared_normalized and observed != declared_normalized:
        return None, f"summary {key}={observed!r} contradicts recording output {declared_normalized!r}"
    return observed or declared_normalized, None


def assess_orange_crop_sync_evidence(
    evidence: OrangeCropSyncEvidence,
    *,
    all_samples_sync_declared: bool,
) -> OrangeSyncAssessment:
    summary_path = Path(evidence.summary_path) if evidence.summary_path else None
    keyframe_path = Path(evidence.keyframe_path) if evidence.keyframe_path else None
    base = {
        "summary_path": str(summary_path) if summary_path else None,
        "keyframe_path": str(keyframe_path) if keyframe_path else None,
    }

    def result(
        status: str,
        *,
        error: Optional[str] = None,
        resolved_gop_length: Optional[int] = None,
        frames_encoded: Optional[int] = None,
        sidecar_total_frames: Optional[int] = None,
        keyframe_count: Optional[int] = None,
    ) -> OrangeSyncAssessment:
        return OrangeSyncAssessment(
            status=status,
            error=error,
            resolved_gop_length=resolved_gop_length,
            frames_encoded=frames_encoded,
            sidecar_total_frames=sidecar_total_frames,
            keyframe_count=keyframe_count,
            **base,
        )

    if summary_path is None or keyframe_path is None:
        return result(
            "orange_idr_sidecar_unavailable",
            error="Orange crop summary and keyframe sidecar are not both available",
        )
    if not summary_path.is_file() or not keyframe_path.is_file():
        return result(
            "orange_idr_sidecar_unavailable",
            error="Orange crop summary or keyframe sidecar is missing",
        )

    try:
        summary = _load_summary(summary_path)
    except OrangeSyncEvidenceError as exc:
        return result("orange_idr_sidecar_unavailable", error=str(exc))

    profile: dict[str, Optional[str]] = {}
    for key, declared in (
        ("output_kind", evidence.declared_output_kind),
        ("stream_kind", evidence.declared_stream_kind),
        ("tuning", evidence.declared_tuning),
    ):
        value, error = _combined_text(summary, key, declared)
        if error:
            return result("orange_idr_sidecar_contradiction", error=error)
        profile[key] = value

    if any(profile[key] is None for key in ("output_kind", "stream_kind", "tuning")):
        return result(
            "orange_idr_sidecar_unavailable",
            error="Orange evidence lacks output_kind, stream_kind, or tuning",
        )
    if profile["output_kind"] != "crop" or profile["stream_kind"] != "crop":
        return result(
            "orange_idr_sidecar_contradiction",
            error="Orange evidence does not describe a crop stream",
        )
    if profile["tuning"] != "lossless":
        return result(
            "orange_idr_sidecar_contradiction",
            error=f"Orange crop tuning is not lossless: {profile['tuning']!r}",
        )

    resolved_gop_length = _optional_nonnegative_int(
        summary.get("resolved_gop_length")
    )
    frames_encoded = _optional_nonnegative_int(summary.get("frames_encoded"))
    if resolved_gop_length is None or frames_encoded is None:
        return result(
            "orange_idr_sidecar_unavailable",
            resolved_gop_length=resolved_gop_length,
            frames_encoded=frames_encoded,
            error="Orange summary lacks resolved_gop_length or frames_encoded",
        )

    for label, declared_count in (
        ("frame_count", evidence.declared_frame_count),
        ("packet_count", evidence.declared_packet_count),
    ):
        if declared_count is not None and (
            type(declared_count) is not int or declared_count != frames_encoded
        ):
            return result(
                "orange_idr_sidecar_contradiction",
                resolved_gop_length=resolved_gop_length,
                frames_encoded=frames_encoded,
                error=(
                    f"recording output {label} contradicts summary frames_encoded"
                ),
            )

    if resolved_gop_length != 1:
        if all_samples_sync_declared:
            return result(
                "orange_idr_sidecar_contradiction",
                resolved_gop_length=resolved_gop_length,
                frames_encoded=frames_encoded,
                error=(
                    "MP4 declares every sample sync, but Orange declares an "
                    f"inter-frame GOP length of {resolved_gop_length}"
                ),
            )
        return result(
            "container_declared",
            resolved_gop_length=resolved_gop_length,
            frames_encoded=frames_encoded,
        )

    try:
        sidecar = _read_keyframe_sidecar(keyframe_path)
    except OrangeSyncEvidenceError as exc:
        return result(
            "orange_idr_sidecar_contradiction",
            resolved_gop_length=resolved_gop_length,
            frames_encoded=frames_encoded,
            error=str(exc),
        )
    if sidecar.total_frames != frames_encoded:
        return result(
            "orange_idr_sidecar_contradiction",
            resolved_gop_length=resolved_gop_length,
            frames_encoded=frames_encoded,
            sidecar_total_frames=sidecar.total_frames,
            keyframe_count=sidecar.keyframe_count,
            error="keyframe sidecar total_frames contradicts summary frames_encoded",
        )
    if sidecar.keyframe_count != frames_encoded:
        return result(
            "orange_idr_sidecar_contradiction",
            resolved_gop_length=resolved_gop_length,
            frames_encoded=frames_encoded,
            sidecar_total_frames=sidecar.total_frames,
            keyframe_count=sidecar.keyframe_count,
            error="keyframe sidecar does not cover every encoded frame",
        )
    return result(
        "orange_idr_sidecar_verified",
        resolved_gop_length=resolved_gop_length,
        frames_encoded=frames_encoded,
        sidecar_total_frames=sidecar.total_frames,
        keyframe_count=sidecar.keyframe_count,
    )
