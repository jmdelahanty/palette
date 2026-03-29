from __future__ import annotations

from fisheye.utils import serve_recording_status_page


def test_host_is_loopback_accepts_local_variants() -> None:
    assert serve_recording_status_page._host_is_loopback("127.0.0.1") is True
    assert serve_recording_status_page._host_is_loopback("localhost") is True
    assert serve_recording_status_page._host_is_loopback("::1") is True


def test_host_is_loopback_rejects_network_bind_hosts() -> None:
    assert serve_recording_status_page._host_is_loopback("0.0.0.0") is False
    assert serve_recording_status_page._host_is_loopback("192.168.1.25") is False
    assert serve_recording_status_page._host_is_loopback("status.palette.example.org") is False


def test_print_network_exposure_warning_is_silent_for_loopback(capsys) -> None:
    serve_recording_status_page._print_network_exposure_warning("127.0.0.1", 8765)
    captured = capsys.readouterr()
    assert captured.err == ""


def test_print_network_exposure_warning_emits_for_non_loopback(capsys) -> None:
    serve_recording_status_page._print_network_exposure_warning("0.0.0.0", 8765)
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "reverse proxy with auth/TLS" in captured.err
    assert "recording_status_page_deployment.md" in captured.err
