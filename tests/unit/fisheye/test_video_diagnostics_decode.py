from __future__ import annotations

from pathlib import Path

from fisheye.diagnostics.video import decode as mod


class _FakeCap:
    def __init__(self, *, open_ok: bool = True) -> None:
        self._open_ok = open_ok
        self.position = 0

    def isOpened(self) -> bool:
        return self._open_ok

    def get(self, prop: int) -> float:
        if prop == 7:
            return 10.0
        if prop == 1:
            return float(self.position)
        return 0.0

    def read(self):
        self.position += 1
        return True, object()

    def set(self, prop: int, value: int) -> None:
        if prop == 1:
            self.position = int(value)

    def release(self) -> None:
        return None


class _FakeCV2:
    CAP_PROP_FRAME_COUNT = 7
    CAP_PROP_POS_FRAMES = 1

    def __init__(self, *, open_ok: bool = True) -> None:
        self._open_ok = open_ok

    def VideoCapture(self, _: str) -> _FakeCap:
        return _FakeCap(open_ok=self._open_ok)


def test_inspect_opencv_fails_when_video_cannot_open(monkeypatch) -> None:
    monkeypatch.setattr(mod, "_load_cv2", lambda: (_FakeCV2(open_ok=False), None))

    report, findings = mod.inspect_opencv(Path("/tmp/video.mp4"), frames_to_check=5, seek_samples=3)

    assert report.status == "fail"
    assert report.open_ok is False
    assert any(f.code == "video.opencv_open_failed" for f in findings)


def test_inspect_decode_reports_backend_error_without_failing_other_backend(monkeypatch) -> None:
    monkeypatch.setattr(mod, "_load_cv2", lambda: (_FakeCV2(open_ok=True), None))
    monkeypatch.setattr(mod, "_load_decord", lambda: (None, "decord missing"))

    reports, findings = mod.inspect_decode(Path("/tmp/video.mp4"), backend="all", frames_to_check=5, seek_samples=3)

    statuses = {item.backend: item.status for item in reports}
    assert statuses["opencv"] == "pass"
    assert statuses["decord"] == "error"
    assert any(f.code == "video.decord_unavailable" for f in findings)
    assert any("does not by itself mean the video file is broken" in (f.details or "") for f in findings)
