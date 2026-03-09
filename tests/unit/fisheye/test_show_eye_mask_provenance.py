from __future__ import annotations

from pathlib import Path

import zarr

from fisheye.diagnostics.show_eye_mask_provenance import show_eye_mask_provenance


def test_show_eye_mask_provenance_prints_raw_and_refined_sections(tmp_path: Path, capsys) -> None:
    root = zarr.open_group(str(tmp_path / 'archive.zarr'), mode='w')

    raw_parent = root.require_group('eye_masks_runs')
    raw_parent.attrs['latest'] = 'eye_001'
    raw = raw_parent.create_group('eye_001')
    raw.attrs.update(
        {
            'method': 'unet_eye_mask_segmenter',
            'source_crop_run': 'crop_001',
            'probabilities_dtype': 'uint8',
            'recommended_probability_threshold': 0.55,
            'recommended_probability_threshold_review': {'reviewer': 'tester'},
        }
    )
    raw.create_array('mask_probs_roi', shape=(3, 1, 8, 8), chunks=(1, 1, 8, 8), dtype='uint8', overwrite=True)
    raw.create_array('frame_indices', shape=(3,), chunks=(3,), dtype='int32', overwrite=True)

    refined_parent = root.require_group('refined_eye_masks_runs')
    refined_parent.attrs['latest'] = 'refined_001'
    refined = refined_parent.create_group('refined_001')
    refined.attrs.update(
        {
            'method': 'refine_eye_masks',
            'source_eye_masks_run': 'eye_001',
            'ellipse_fit_backend': 'opencv',
            'ellipse_fit_method': 'cv2.fitEllipse',
            'mask_probability_threshold': 0.45,
            'metrics_summary': {'reason_tag_counts': {'ellipse_fail_pair': 2}},
        }
    )
    refined.create_array('masks_roi', shape=(3, 2, 8, 8), chunks=(1, 2, 8, 8), dtype='uint8', overwrite=True)
    refined.create_array('ellipse_success', shape=(3, 2), chunks=(3, 2), dtype='bool', overwrite=True)

    show_eye_mask_provenance(tmp_path / 'archive.zarr')
    out = capsys.readouterr().out

    assert '=== eye_masks_runs/eye_001 attrs ===' in out
    assert '=== refined_eye_masks_runs/refined_001 attrs ===' in out
    assert 'recommended_probability_threshold' in out
    assert 'reason_tag_counts' in out
    assert 'cv2.fitEllipse' in out
    assert 'mask_probs_roi' in out
    assert 'masks_roi' in out
