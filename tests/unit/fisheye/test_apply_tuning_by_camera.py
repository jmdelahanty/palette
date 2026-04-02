from __future__ import annotations

from pathlib import Path

import h5py
import zarr

from fisheye.utils import apply_tuning_by_camera as mod


def _make_zarr(
    path: Path,
    *,
    camera_id: str | None = None,
    zarr_use: str | None = None,
    tuning: dict[str, object] | None = None,
) -> None:
    root = zarr.open_group(str(path), mode="w")
    if camera_id is not None:
        root.attrs["camera_id"] = camera_id
    if zarr_use is not None:
        root.attrs["zarr_use"] = zarr_use
    if tuning:
        analysis = root.create_group("analysis_metadata")
        for key, value in tuning.items():
            analysis.attrs[key] = value


def _read_tuning(path: Path, key: str) -> object:
    root = zarr.open_group(str(path), mode="r", use_consolidated=False)
    analysis = root.get("analysis_metadata")
    if analysis is None:
        return None
    return analysis.attrs.get(key)


def test_candidate_h5_stems_adds_suffixless_variants() -> None:
    assert mod._candidate_h5_stems("recording_training") == ["recording_training", "recording"]
    assert mod._candidate_h5_stems("recording_analysis") == ["recording_analysis", "recording"]
    assert mod._candidate_h5_stems("recording") == ["recording"]


def test_camera_id_for_zarr_falls_back_to_suffixless_h5(tmp_path: Path) -> None:
    recording_dir = tmp_path / "rec"
    raw_dir = recording_dir / "raw"
    zarr_dir = recording_dir / "zarr"
    raw_dir.mkdir(parents=True)
    zarr_dir.mkdir(parents=True)

    h5_path = raw_dir / "example.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["camera_id"] = "2010093"

    zarr_path = zarr_dir / "example_training.zarr"
    _make_zarr(zarr_path, zarr_use="training")

    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert mod._camera_id_for_zarr(zarr_path, root) == "2010093"


def test_main_apply_defaults_to_source_use_scope(tmp_path: Path) -> None:
    source = tmp_path / "rec_src" / "zarr" / "rec_src_training.zarr"
    target_training = tmp_path / "rec_a" / "zarr" / "rec_a_training.zarr"
    target_analysis = tmp_path / "rec_b" / "zarr" / "rec_b_analysis.zarr"
    source.parent.mkdir(parents=True)
    target_training.parent.mkdir(parents=True)
    target_analysis.parent.mkdir(parents=True)

    _make_zarr(
        source,
        camera_id="2010093",
        zarr_use="training",
        tuning={"eye_mask_tuning": {"tuned_parameters": {"min_circularity": 0.77}}},
    )
    _make_zarr(target_training, camera_id="2010093", zarr_use="training")
    _make_zarr(target_analysis, camera_id="2010093", zarr_use="analysis")

    rc = mod.main(
        [
            str(tmp_path),
            "--source",
            str(source),
            "--recursive",
            "--apply",
            "--keys",
            "eye_mask_tuning",
        ]
    )
    assert rc == 0
    assert _read_tuning(target_training, "eye_mask_tuning") == {
        "tuned_parameters": {"min_circularity": 0.77}
    }
    assert _read_tuning(target_analysis, "eye_mask_tuning") is None


def test_main_apply_can_target_analysis_use(tmp_path: Path) -> None:
    source = tmp_path / "rec_src" / "zarr" / "rec_src_training.zarr"
    target_training = tmp_path / "rec_a" / "zarr" / "rec_a_training.zarr"
    target_analysis = tmp_path / "rec_b" / "zarr" / "rec_b_analysis.zarr"
    source.parent.mkdir(parents=True)
    target_training.parent.mkdir(parents=True)
    target_analysis.parent.mkdir(parents=True)

    _make_zarr(
        source,
        camera_id="2010093",
        zarr_use="training",
        tuning={"keypoint_tuning": {"threshold": 0.12}},
    )
    _make_zarr(target_training, camera_id="2010093", zarr_use="training")
    _make_zarr(target_analysis, camera_id="2010093", zarr_use="analysis")

    rc = mod.main(
        [
            str(tmp_path),
            "--source",
            str(source),
            "--recursive",
            "--apply",
            "--zarr-use",
            "analysis",
            "--keys",
            "keypoint_tuning",
        ]
    )
    assert rc == 0
    assert _read_tuning(target_analysis, "keypoint_tuning") == {"threshold": 0.12}
    assert _read_tuning(target_training, "keypoint_tuning") is None


def test_main_apply_merge_dicts_preserves_unrelated_subject_mask_components(tmp_path: Path) -> None:
    source = tmp_path / "rec_src" / "zarr" / "rec_src_training.zarr"
    target = tmp_path / "rec_target" / "zarr" / "rec_target_training.zarr"
    source.parent.mkdir(parents=True)
    target.parent.mkdir(parents=True)

    _make_zarr(
        source,
        camera_id="2010093",
        zarr_use="training",
        tuning={
            "subject_mask_tuning": {
                "version": "2.0",
                "components": {
                    "subject_body": {
                        "method": "traditional_subject_mask_seed",
                        "tuned_parameters": {"diff_threshold": 44},
                    }
                },
            }
        },
    )
    _make_zarr(
        target,
        camera_id="2010093",
        zarr_use="training",
        tuning={
            "subject_mask_tuning": {
                "version": "2.0",
                "components": {
                    "eyes_union": {
                        "method": "global_threshold_otsu",
                        "tuned_parameters": {"roi_padding": 12},
                    },
                    "subject_body": {
                        "method": "traditional_subject_mask_seed",
                        "tuned_parameters": {"diff_threshold": 20, "min_area": 5},
                    },
                },
            }
        },
    )

    rc = mod.main(
        [
            str(tmp_path),
            "--source",
            str(source),
            "--recursive",
            "--apply",
            "--keys",
            "subject_mask_tuning",
            "--merge-dicts",
        ]
    )
    assert rc == 0
    assert _read_tuning(target, "subject_mask_tuning") == {
        "version": "2.0",
        "components": {
            "eyes_union": {
                "method": "global_threshold_otsu",
                "tuned_parameters": {"roi_padding": 12},
            },
            "subject_body": {
                "method": "traditional_subject_mask_seed",
                "tuned_parameters": {"diff_threshold": 44, "min_area": 5},
            },
        },
    }


def test_parse_subject_mask_components_normalizes_aliases() -> None:
    assert mod._parse_subject_mask_components(["swim-bladder,body", "left-eye"]) == [  # noqa: SLF001
        "swim_bladder",
        "subject_body",
        "eye_left",
    ]


def test_filter_subject_mask_tuning_payload_selects_requested_components() -> None:
    payload, missing = mod._filter_subject_mask_tuning_payload(  # noqa: SLF001
        {
            "version": "2.0",
            "latest_component": "swim_bladder",
            "components": {
                "subject_body": {"method": "traditional_subject_mask_seed"},
                "swim_bladder": {"method": "polar_boundary_center_seed"},
            },
        },
        ["swim_bladder", "eye_left"],
    )

    assert payload == {
        "version": "2.0",
        "components": {
            "swim_bladder": {"method": "polar_boundary_center_seed"},
        },
    }
    assert missing == ["eye_left"]


def test_apply_subject_mask_component_updates_preserves_unrelated_components() -> None:
    attrs = {
        "subject_mask_tuning": {
            "version": "2.0",
            "components": {
                "subject_body": {
                    "method": "traditional_subject_mask_seed",
                    "tuned_parameters": {"diff_threshold": 20},
                }
            },
        }
    }

    updated, skipped = mod._apply_subject_mask_component_updates(  # noqa: SLF001
        attrs,
        {
            "version": "2.0",
            "components": {
                "swim_bladder": {
                    "method": "polar_boundary_center_seed",
                    "tuned_parameters": {"ray_count": 96},
                }
            },
        },
        overwrite=False,
        merge_dicts=False,
    )

    assert updated == ["subject_mask_tuning.components.swim_bladder"]
    assert skipped == []
    assert attrs["subject_mask_tuning"] == {
        "version": "2.0",
        "components": {
            "subject_body": {
                "method": "traditional_subject_mask_seed",
                "tuned_parameters": {"diff_threshold": 20},
            },
            "swim_bladder": {
                "method": "polar_boundary_center_seed",
                "tuned_parameters": {"ray_count": 96},
            },
        },
    }


def test_apply_subject_mask_component_updates_skips_then_overwrite_merges_selected_component() -> None:
    attrs = {
        "subject_mask_tuning": {
            "version": "2.0",
            "components": {
                "subject_body": {
                    "method": "traditional_subject_mask_seed",
                    "tuned_parameters": {"diff_threshold": 20},
                },
                "swim_bladder": {
                    "method": "swim_bladder_patch_threshold_v1",
                    "tuned_parameters": {"diff_threshold": 7, "min_area": 5},
                },
            },
        }
    }
    value = {
        "version": "2.0",
        "components": {
            "swim_bladder": {
                "method": "polar_boundary_center_seed",
                "tuned_parameters": {"ray_count": 96},
            }
        },
    }

    updated, skipped = mod._apply_subject_mask_component_updates(  # noqa: SLF001
        attrs,
        value,
        overwrite=False,
        merge_dicts=True,
    )
    assert updated == []
    assert skipped == ["subject_mask_tuning.components.swim_bladder"]

    updated, skipped = mod._apply_subject_mask_component_updates(  # noqa: SLF001
        attrs,
        value,
        overwrite=True,
        merge_dicts=True,
    )
    assert updated == ["subject_mask_tuning.components.swim_bladder"]
    assert skipped == []
    assert attrs["subject_mask_tuning"]["components"]["subject_body"] == {
        "method": "traditional_subject_mask_seed",
        "tuned_parameters": {"diff_threshold": 20},
    }
    assert attrs["subject_mask_tuning"]["components"]["swim_bladder"] == {
        "method": "polar_boundary_center_seed",
        "tuned_parameters": {"diff_threshold": 7, "min_area": 5, "ray_count": 96},
    }


def test_main_apply_subject_mask_components_adds_only_requested_component(tmp_path: Path) -> None:
    source = tmp_path / "rec_src" / "zarr" / "rec_src_training.zarr"
    target = tmp_path / "rec_target" / "zarr" / "rec_target_training.zarr"
    source.parent.mkdir(parents=True)
    target.parent.mkdir(parents=True)

    _make_zarr(
        source,
        camera_id="2010093",
        zarr_use="training",
        tuning={
            "subject_mask_tuning": {
                "version": "2.0",
                "latest_component": "swim_bladder",
                "components": {
                    "subject_body": {
                        "method": "traditional_subject_mask_seed",
                        "tuned_parameters": {"diff_threshold": 44},
                    },
                    "swim_bladder": {
                        "method": "polar_boundary_center_seed",
                        "tuned_parameters": {"ray_count": 96},
                    },
                },
            }
        },
    )
    _make_zarr(
        target,
        camera_id="2010093",
        zarr_use="training",
        tuning={
            "subject_mask_tuning": {
                "version": "2.0",
                "components": {
                    "subject_body": {
                        "method": "traditional_subject_mask_seed",
                        "tuned_parameters": {"diff_threshold": 20, "min_area": 5},
                    }
                },
            }
        },
    )

    rc = mod.main(
        [
            str(tmp_path),
            "--source",
            str(source),
            "--recursive",
            "--apply",
            "--keys",
            "subject_mask_tuning",
            "--subject-mask-components",
            "swim_bladder",
        ]
    )

    assert rc == 0
    assert _read_tuning(target, "subject_mask_tuning") == {
        "version": "2.0",
        "components": {
            "subject_body": {
                "method": "traditional_subject_mask_seed",
                "tuned_parameters": {"diff_threshold": 20, "min_area": 5},
            },
            "swim_bladder": {
                "method": "polar_boundary_center_seed",
                "tuned_parameters": {"ray_count": 96},
            },
        },
    }


def test_main_apply_subject_mask_components_skips_existing_component_without_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "rec_src" / "zarr" / "rec_src_training.zarr"
    target = tmp_path / "rec_target" / "zarr" / "rec_target_training.zarr"
    source.parent.mkdir(parents=True)
    target.parent.mkdir(parents=True)

    _make_zarr(
        source,
        camera_id="2010093",
        zarr_use="training",
        tuning={
            "subject_mask_tuning": {
                "version": "2.0",
                "components": {
                    "swim_bladder": {
                        "method": "polar_boundary_center_seed",
                        "tuned_parameters": {"ray_count": 96},
                    }
                },
            }
        },
    )
    _make_zarr(
        target,
        camera_id="2010093",
        zarr_use="training",
        tuning={
            "subject_mask_tuning": {
                "version": "2.0",
                "components": {
                    "subject_body": {
                        "method": "traditional_subject_mask_seed",
                        "tuned_parameters": {"diff_threshold": 20},
                    },
                    "swim_bladder": {
                        "method": "swim_bladder_patch_threshold_v1",
                        "tuned_parameters": {"diff_threshold": 7},
                    },
                },
            }
        },
    )

    rc = mod.main(
        [
            str(tmp_path),
            "--source",
            str(source),
            "--recursive",
            "--apply",
            "--keys",
            "subject_mask_tuning",
            "--subject-mask-components",
            "swim_bladder",
        ]
    )

    assert rc == 0
    assert _read_tuning(target, "subject_mask_tuning") == {
        "version": "2.0",
        "components": {
            "subject_body": {
                "method": "traditional_subject_mask_seed",
                "tuned_parameters": {"diff_threshold": 20},
            },
            "swim_bladder": {
                "method": "swim_bladder_patch_threshold_v1",
                "tuned_parameters": {"diff_threshold": 7},
            },
        },
    }


def test_main_apply_subject_mask_components_overwrite_merges_selected_component(tmp_path: Path) -> None:
    source = tmp_path / "rec_src" / "zarr" / "rec_src_training.zarr"
    target = tmp_path / "rec_target" / "zarr" / "rec_target_training.zarr"
    source.parent.mkdir(parents=True)
    target.parent.mkdir(parents=True)

    _make_zarr(
        source,
        camera_id="2010093",
        zarr_use="training",
        tuning={
            "subject_mask_tuning": {
                "version": "2.0",
                "components": {
                    "swim_bladder": {
                        "method": "polar_boundary_center_seed",
                        "tuned_parameters": {"ray_count": 96},
                    }
                },
            }
        },
    )
    _make_zarr(
        target,
        camera_id="2010093",
        zarr_use="training",
        tuning={
            "subject_mask_tuning": {
                "version": "2.0",
                "components": {
                    "subject_body": {
                        "method": "traditional_subject_mask_seed",
                        "tuned_parameters": {"diff_threshold": 20},
                    },
                    "swim_bladder": {
                        "method": "swim_bladder_patch_threshold_v1",
                        "tuned_parameters": {"diff_threshold": 7, "min_area": 5},
                    },
                },
            }
        },
    )

    rc = mod.main(
        [
            str(tmp_path),
            "--source",
            str(source),
            "--recursive",
            "--apply",
            "--keys",
            "subject_mask_tuning",
            "--subject-mask-components",
            "swim_bladder",
            "--overwrite",
            "--merge-dicts",
        ]
    )

    assert rc == 0
    assert _read_tuning(target, "subject_mask_tuning") == {
        "version": "2.0",
        "components": {
            "subject_body": {
                "method": "traditional_subject_mask_seed",
                "tuned_parameters": {"diff_threshold": 20},
            },
            "swim_bladder": {
                "method": "polar_boundary_center_seed",
                "tuned_parameters": {"diff_threshold": 7, "min_area": 5, "ray_count": 96},
            },
        },
    }
