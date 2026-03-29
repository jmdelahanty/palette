from __future__ import annotations

import pytest

from fisheye.utils import backfill_subject_mask_tuning as mod


class _FakeGroup(dict):
    def __init__(self, *args: object, attrs: dict[str, object] | None = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.attrs = attrs or {}

    def get(self, key: str, default: object = None) -> object:
        return super().get(key, default)

    def require_group(self, key: str) -> "_FakeGroup":
        value = self.get(key)
        if isinstance(value, _FakeGroup):
            return value
        value = _FakeGroup()
        self[key] = value
        return value


def _eye_tuning_payload() -> dict[str, object]:
    return {
        "method": "global_threshold_otsu",
        "version": "1.0",
        "tuned_timestamp": "2026-02-12T19:51:24+00:00",
        "tuned_parameters": {
            "roi_padding": 14,
            "pre_threshold": 33,
            "sobel_strength": 0.25,
            "min_area": 18,
        },
        "context": {
            "crop_run": "crop_001",
            "roi_index": 17,
        },
    }


def test_backfill_subject_mask_tuning_migrates_legacy_eye_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    analysis = _FakeGroup(attrs={"eye_mask_tuning": _eye_tuning_payload()})
    root = _FakeGroup({"analysis_metadata": analysis})
    monkeypatch.setattr(mod, "open_zarr_root", lambda *args, **kwargs: root)

    result = mod.backfill_subject_mask_tuning("recording_training.zarr", apply=True)

    assert result["status"] == "updated"
    subject_tuning = analysis.attrs["subject_mask_tuning"]
    entry = subject_tuning["components"]["eyes_union"]
    assert entry["method"] == "global_threshold_otsu"
    assert entry["subject_method_family"] == "eye_mask_threshold_lr_v1"
    assert entry["output_labels"] == ["eye_left", "eye_right"]
    assert entry["storage_component"] == "eyes_union"
    assert entry["tuned_parameters"]["roi_padding"] == 14
    assert entry["context"]["migration_source_key"] == "eye_mask_tuning"
    assert entry["context"]["storage_component_name"] == "eyes_union"
    assert analysis.attrs["eye_mask_tuning"]["method"] == "global_threshold_otsu"


def test_backfill_subject_mask_tuning_preserves_existing_subject_components(monkeypatch: pytest.MonkeyPatch) -> None:
    analysis = _FakeGroup(
        attrs={
            "eye_mask_tuning": _eye_tuning_payload(),
            "subject_mask_tuning": {
                "version": "2.0",
                "components": {
                    "subject_body": {
                        "method": "traditional_subject_mask_seed",
                        "version": "1.0",
                        "tuned_parameters": {"diff_threshold": 51},
                        "context": {"storage_component_name": "subject_body"},
                    }
                },
                "latest_component": "subject_body",
            },
        }
    )
    root = _FakeGroup({"analysis_metadata": analysis})
    monkeypatch.setattr(mod, "open_zarr_root", lambda *args, **kwargs: root)

    result = mod.backfill_subject_mask_tuning("recording_training.zarr", apply=True)

    assert result["status"] == "updated"
    subject_tuning = analysis.attrs["subject_mask_tuning"]
    assert set(subject_tuning["components"]) == {"subject_body", "eyes_union"}
    assert subject_tuning["components"]["subject_body"]["tuned_parameters"] == {"diff_threshold": 51}
    assert subject_tuning["latest_component"] == "subject_body"


def test_backfill_subject_mask_tuning_skips_existing_eyes_union_without_overwrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = _FakeGroup(
        attrs={
            "eye_mask_tuning": _eye_tuning_payload(),
            "subject_mask_tuning": {
                "version": "2.0",
                "components": {
                    "eyes_union": {
                        "method": "global_threshold_otsu",
                        "version": "1.0",
                        "tuned_parameters": {"roi_padding": 99},
                        "context": {},
                    }
                },
            },
        }
    )
    root = _FakeGroup({"analysis_metadata": analysis})
    monkeypatch.setattr(mod, "open_zarr_root", lambda *args, **kwargs: root)

    result = mod.backfill_subject_mask_tuning("recording_training.zarr", apply=True)

    assert result["status"] == "exists"
    assert analysis.attrs["subject_mask_tuning"]["components"]["eyes_union"]["tuned_parameters"]["roi_padding"] == 99


def test_backfill_subject_mask_tuning_overwrites_existing_eyes_union_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = _FakeGroup(
        attrs={
            "eye_mask_tuning": _eye_tuning_payload(),
            "subject_mask_tuning": {
                "version": "2.0",
                "components": {
                    "eyes_union": {
                        "method": "global_threshold_otsu",
                        "version": "1.0",
                        "tuned_parameters": {"roi_padding": 99},
                        "context": {},
                    }
                },
            },
        }
    )
    root = _FakeGroup({"analysis_metadata": analysis})
    monkeypatch.setattr(mod, "open_zarr_root", lambda *args, **kwargs: root)

    result = mod.backfill_subject_mask_tuning("recording_training.zarr", overwrite=True, apply=True)

    assert result["status"] == "updated"
    assert analysis.attrs["subject_mask_tuning"]["components"]["eyes_union"]["tuned_parameters"]["roi_padding"] == 14


def test_backfill_subject_mask_tuning_dry_run_does_not_modify_attrs(monkeypatch: pytest.MonkeyPatch) -> None:
    analysis = _FakeGroup(attrs={"eye_mask_tuning": _eye_tuning_payload()})
    root = _FakeGroup({"analysis_metadata": analysis})
    monkeypatch.setattr(mod, "open_zarr_root", lambda *args, **kwargs: root)

    result = mod.backfill_subject_mask_tuning("recording_training.zarr", apply=False)

    assert result["status"] == "would_update"
    assert "subject_mask_tuning" not in analysis.attrs
