from __future__ import annotations

import hashlib
from pathlib import Path

from fisheye.registry.model_resolution import Candidate
from fisheye.utils import resolve_latest_registered_model as mod


class _FakeRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _candidate(*, path: Path, run_id: str, created_utc: str, sha256: str) -> Candidate:
    return Candidate(
        run_id=run_id,
        set_id="detect_set",
        model_path=str(path),
        model_sha256=sha256,
        created_utc=created_utc,
        status="success",
        dataset_count=0,
        weighted_score=0.0,
        feature_match_counts={},
        feature_weights_used=0.0,
    )


def test_resolves_and_verifies_first_recency_ordered_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    registry_path.touch()
    newest = tmp_path / "newest.pt"
    older = tmp_path / "older.pt"
    newest.write_bytes(b"newest")
    older.write_bytes(b"older")
    newest_hash = hashlib.sha256(b"newest").hexdigest()
    older_hash = hashlib.sha256(b"older").hexdigest()
    fake = _FakeRegistry(registry_path)
    monkeypatch.setattr(mod, "Registry", lambda _path: fake)

    def load_candidates(_registry, **kwargs):
        assert kwargs["task"] == "detect"
        assert kwargs["include_non_success"] is False
        assert kwargs["target"].recording_id == "registry_latest_model_resolution"
        return [
            _candidate(
                path=newest,
                run_id="newest_run",
                created_utc="2026-07-24T00:00:00+00:00",
                sha256=newest_hash,
            ),
            _candidate(
                path=older,
                run_id="older_run",
                created_utc="2026-07-23T00:00:00+00:00",
                sha256=older_hash,
            ),
        ]

    monkeypatch.setattr(mod, "load_candidates", load_candidates)
    result = mod.resolve_latest_registered_model(
        registry_path,
        task="detect",
    )

    assert fake.closed is True
    assert result["selection_policy"] == mod.SELECTION_POLICY
    assert result["candidate_count"] == 2
    assert result["selected"]["run_id"] == "newest_run"
    assert result["selected"]["model_sha256"] == newest_hash
    assert result["content_verification"]["sha256"] == newest_hash


def test_rejects_registered_digest_mismatch(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "registry.sqlite"
    registry_path.touch()
    model = tmp_path / "model.pt"
    model.write_bytes(b"actual")
    fake = _FakeRegistry(registry_path)
    monkeypatch.setattr(mod, "Registry", lambda _path: fake)
    monkeypatch.setattr(
        mod,
        "load_candidates",
        lambda *_args, **_kwargs: [
            _candidate(
                path=model,
                run_id="bad_run",
                created_utc="2026-07-24T00:00:00+00:00",
                sha256="0" * 64,
            )
        ],
    )

    try:
        mod.resolve_latest_registered_model(registry_path, task="detect")
    except RuntimeError as exc:
        assert "differs from its registry digest" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("digest mismatch was accepted")
