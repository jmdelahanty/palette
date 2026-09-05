from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from fisheye.utils import materialize_chaser_proxy_relative_frame as cli


def _core_args(tmp_path):
    digest = "a" * 64
    roster = tmp_path / "core-roster.json"
    roster.write_text(json.dumps({"record_sha256": digest}), encoding="utf-8")
    return cli._parser().parse_args(
        [
            str(tmp_path / "recording_analysis.zarr"),
            "--proxy-run-name",
            "proxy-a",
            "--output-run-name",
            "relative-a",
            "--scratch-root",
            str(tmp_path / "scratch"),
            "--expected-recording-id",
            "recording",
            "--core-authority-roster",
            str(roster),
            "--expected-core-authority-roster-sha256",
            digest,
            "--core-track-id",
            "7",
        ]
    )


def test_core_mode_uses_only_core_preparation(monkeypatch, tmp_path) -> None:
    args = _core_args(tmp_path)
    calls: list[dict[str, object]] = []
    prepared = object()
    bound = SimpleNamespace(
        prepared=prepared,
        to_json=lambda: {"mode": "core"},
    )

    def core_prepare(*positional, **keywords):
        calls.append({"positional": positional, "keywords": keywords})
        return bound

    monkeypatch.setattr(cli, "prepare_core_proxy_chaser_relative_frame", core_prepare)
    monkeypatch.setattr(
        cli,
        "prepare_proxy_relative_frame",
        lambda *args, **kwargs: pytest.fail("legacy preparation must not run"),
    )
    monkeypatch.setattr(
        cli,
        "build_chaser_relative_frame_materialization_plan",
        lambda *args, **kwargs: SimpleNamespace(to_json=lambda: {"planned": True}),
    )

    result = cli.run(args)

    assert result["mode"] == "core"
    assert result["status"] == "planned_no_writes"
    assert len(calls) == 1
    assert calls[0]["keywords"]["core_track_id"] == 7
    assert calls[0]["keywords"]["core_authority_roster"] == {"record_sha256": "a" * 64}


def test_core_mode_never_falls_back_after_failure(monkeypatch, tmp_path) -> None:
    args = _core_args(tmp_path)
    monkeypatch.setattr(
        cli,
        "prepare_core_proxy_chaser_relative_frame",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("core blocked")),
    )
    monkeypatch.setattr(
        cli,
        "prepare_proxy_relative_frame",
        lambda *args, **kwargs: pytest.fail("legacy fallback must not run"),
    )

    with pytest.raises(ValueError, match="core blocked"):
        cli.run(args)


def test_core_mode_rejects_legacy_body_argument_before_reading(
    monkeypatch, tmp_path
) -> None:
    args = _core_args(tmp_path)
    args.body_frame_run = "legacy-body"
    monkeypatch.setattr(
        cli,
        "prepare_core_proxy_chaser_relative_frame",
        lambda *args, **kwargs: pytest.fail("core preparation must not run"),
    )
    monkeypatch.setattr(
        cli,
        "prepare_proxy_relative_frame",
        lambda *args, **kwargs: pytest.fail("legacy preparation must not run"),
    )

    with pytest.raises(ValueError, match="cannot accept a legacy body-frame"):
        cli.run(args)
