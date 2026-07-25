# Pixel Decode Exposure Census - 2026-07-02

<!-- contract-meta
status: generated
generated_utc: 2026-07-02T08:13:09.908051+00:00
scope: read-only registry training datasets and zarr image surfaces
-->

## Scope

- Registry opened read-only: `/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite`
- Datasets selected: `112` (`zarr_use='training'`, `artifact_kind='derived_training_merge'`, or merged dataset id).
- Surfaces sampled: `265`
- Sample policy: up to `4` frames and `262144` pixels per image surface.
- Default raw-video sampling uses cheap/model-facing raw surfaces (`images_ds`, `images`, and RGB/color downsampled variants); full-resolution raw frames are deliberately not sampled by this census.
- No zarr store or registry row was modified.

## Classification Rule

- `range_expanded_like`: sampled uint8 values mostly occupy the value lattice produced by limited-range expansion `(Y - 16) * 255 / 219`, leaving >=30 forbidden bins empty.
- `direct_y_like`: sampled values include many bins that cannot be produced by that limited-range expansion, consistent with direct full-range Y storage.
- `indeterminate`: insufficient dynamic range/sample size or weak/mixed histogram evidence.

## Headline Finding

- `range_expanded_like` surfaces: `0`
- `direct_y_like` surfaces: `195`
- `indeterminate` surfaces: `68`
- No sampled model-facing training surface showed the limited-range expansion lattice.

## Summary

### Dataset Classifications

| classification | datasets |
| --- | ---: |
| direct_y_like | 102 |
| indeterminate | 8 |
| missing_zarr | 1 |
| unreadable_zarr | 1 |

### Surface Classifications

| classification | surfaces |
| --- | ---: |
| direct_y_like | 195 |
| range_expanded_like | 0 |
| indeterminate | 68 |
| missing_zarr | 1 |
| unreadable_zarr | 1 |

### Pixel Contract Names

| pixel contract | surfaces |
| --- | ---: |
| `missing` | 98 |
| `orange_mono_pynvvc_luma_uint8_v1` | 112 |
| `raw_video_images_full_to_uint8_grayscale` | 55 |

### Decode Backends

| decode backend | surfaces |
| --- | ---: |
| `missing` | 149 |
| `pynvvc_luma` | 116 |

## Per-Dataset Summary

| dataset_id | kind | use | summary | surfaces | contracts | backends | path |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| detect_all_available_detect_training_v001_merged | derived_training_merge | training | direct_y_like | 1 | `missing` | `missing` | `/nvme1/training/datasets/detect_all_available_detect_training_v001/detect_all_available_detect_training_v001_merged.zarr` |
| detect_all_available_detect_training_v002_merged | derived_training_merge | training | direct_y_like | 1 | `missing` | `missing` | `/nvme1/training/datasets/detect_all_available_detect_training_v002/detect_all_available_detect_training_v002_merged.zarr` |
| detect_all_available_detect_training_v003_merged | derived_training_merge | training | direct_y_like | 1 | `missing` | `missing` | `/nvme1/training/datasets/detect_all_available_detect_training_v003/detect_all_available_detect_training_v003_merged.zarr` |
| detect_all_available_detect_training_v004_merged | derived_training_merge | training | direct_y_like | 1 | `missing` | `missing` | `/nvme1/training/datasets/detect_all_available_detect_training_v004/zarr/detect_all_available_detect_training_v004_merged.zarr` |
| detect_cedar_shadow_omnifin0_manual_gray_85dfa0ee_v002_merged | derived_training_merge | training | direct_y_like | 1 | `missing` | `missing` | `/nvme1/training/datasets/detect_cedar_shadow_omnifin0_manual_gray_85dfa0ee_v002/zarr/detect_cedar_shadow_omnifin0_manual_gray_85dfa0ee_v002_merged.zarr` |
| detect_cedar_shadow_omnifin0_manual_gray_85dfa0ee_v003_merged | derived_training_merge | training | direct_y_like | 1 | `missing` | `missing` | `/nvme1/training/datasets/detect_cedar_shadow_omnifin0_manual_gray_85dfa0ee_v003/zarr/detect_cedar_shadow_omnifin0_manual_gray_85dfa0ee_v003_merged.zarr` |
| detect_cedar_shadow_v007_merged | derived_training_merge | training | direct_y_like | 1 | `missing` | `missing` | `/nvme1/training/datasets/detect_cedar_shadow_v007/zarr/detect_cedar_shadow_v007_merged.zarr` |
| detect_recording_only_sleepy_sicky_detect_v001_merged | derived_training_merge | training | direct_y_like | 1 | `missing` | `missing` | `/nvme1/training/datasets/detect_recording_only_sleepy_sicky_detect_v001/detect_recording_only_sleepy_sicky_detect_v001_merged.zarr` |
| path-6c23364be96c | derived_training_merge | training | indeterminate | 1 | `missing` | `missing` | `/nvme1/training/datasets/eye_mask_cedar_shadow_omnifin0_auto_gray_lr_b9164009_v002/zarr/eye_mask_cedar_shadow_omnifin0_auto_gray_lr_b9164009_v002_merged.zarr` |
| path-81c4928a9040 | derived_training_merge | training | indeterminate | 1 | `missing` | `missing` | `/nvme1/training/datasets/eye_mask_cedar_shadow_omnifin0_auto_gray_lr_b9164009_v001/zarr/eye_mask_cedar_shadow_omnifin0_auto_gray_lr_b9164009_v001_merged.zarr` |
| path-f1bb73bbd19c | derived_training_merge | training | indeterminate | 1 | `missing` | `missing` | `/nvme1/training/datasets/eye_mask_cedar_shadow_omnifin0_auto_gray_union_b9164009_v001/zarr/eye_mask_cedar_shadow_omnifin0_auto_gray_union_b9164009_v001_merged.zarr` |
| pose_all_registry_reviewed_v2_keypoints_20260520_v001_merged | derived_training_merge | training | direct_y_like | 1 | `missing` | `missing` | `/nvme1/training/datasets/pose_all_registry_reviewed_v2_keypoints_20260520_v001/zarr/pose_all_registry_reviewed_v2_keypoints_20260520_v001_merged.zarr` |
| pose_cedar_shadow_filtered_gray_latest_traditional_a4c30ae1_v001_merged | derived_training_merge | training | indeterminate | 1 | `missing` | `missing` | `/nvme1/training/datasets/pose_cedar_shadow_filtered_gray_latest_traditional_a4c30ae1_v001/zarr/pose_cedar_shadow_filtered_gray_latest_traditional_a4c30ae1_v001_merged.zarr` |
| pose_cedar_shadow_filtered_gray_latest_traditional_refresh_v001_merged | derived_training_merge | training | indeterminate | 1 | `missing` | `missing` | `/nvme1/training/datasets/pose_cedar_shadow_filtered_gray_latest_traditional_refresh_v001/zarr/pose_cedar_shadow_filtered_gray_latest_traditional_refresh_v001_merged.zarr` |
| pose_cedar_shadow_pose_traditional_v2_v002_merged | derived_training_merge | training | indeterminate | 1 | `missing` | `missing` | `/nvme1/training/datasets/pose_cedar_shadow_pose_traditional_v2_v002/zarr/pose_cedar_shadow_pose_traditional_v2_v002_merged.zarr` |
| pose_pose_box_only_smoke_v001_merged | derived_training_merge | training | missing_zarr | 1 | `missing` | `missing` | `/tmp/pose_box_only_smoke.zarr` |
| pose_pose_box_only_v1_v001_merged | derived_training_merge | training | indeterminate | 1 | `missing` | `missing` | `/nvme1/training/datasets/pose_pose_box_only_v1_v001/zarr/pose_pose_box_only_v1_v001_merged.zarr` |
| subject_mask_cedar_shadow_omnifin0_gray_subject_v1_union_c6ff03ae_v001_merged | derived_training_merge | training | indeterminate | 1 | `missing` | `missing` | `/nvme1/training/datasets/subject_mask_cedar_shadow_omnifin0_gray_subject_v1_union_c6ff03ae_v001/zarr/subject_mask_cedar_shadow_omnifin0_gray_subject_v1_union_c6ff03ae_v001_merged.zarr` |
| 2026-01-28T19-22-28Z_arena_1:zc66de17bea1b | source_recording | training | direct_y_like | 2 | `missing` | `missing` | `/nvme1/recordings/2026-01-28T19-22-28Z_arena_1_DefaultScreen/zarr/2026-01-28T19-22-28Z_arena_1_DefaultScreen_training.zarr` |
| 2026-01-28T19-22-28Z_arena_2:ze41894008350 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T19-22-28Z_arena_2_DefaultScreen/zarr/2026-01-28T19-22-28Z_arena_2_DefaultScreen_training.zarr` |
| 2026-01-28T19-22-28Z_arena_3:ze9b35d662a87 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T19-22-28Z_arena_3_DefaultScreen/zarr/2026-01-28T19-22-28Z_arena_3_DefaultScreen_training.zarr` |
| 2026-01-28T19-22-28Z_arena_4:zd9af96ac53af | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T19-22-28Z_arena_4_DefaultScreen/zarr/2026-01-28T19-22-28Z_arena_4_DefaultScreen_training.zarr` |
| 2026-01-28T19-36-18Z_arena_1:zfe208d1b989b | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T19-36-18Z_arena_1_Feeding/zarr/2026-01-28T19-36-18Z_arena_1_Feeding_training.zarr` |
| 2026-01-28T19-36-18Z_arena_2:z386db9c2a8f6 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T19-36-18Z_arena_2_Feeding/zarr/2026-01-28T19-36-18Z_arena_2_Feeding_training.zarr` |
| 2026-01-28T19-36-18Z_arena_3:z87d5cee5f737 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T19-36-18Z_arena_3_Feeding/zarr/2026-01-28T19-36-18Z_arena_3_Feeding_training.zarr` |
| 2026-01-28T19-36-18Z_arena_4:zb8e6ba77d834 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T19-36-18Z_arena_4_Feeding/zarr/2026-01-28T19-36-18Z_arena_4_Feeding_training.zarr` |
| 2026-01-28T20-41-59Z_arena_1:zd3aa1c19985b | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T20-41-59Z_arena_1_DefaultScreen/zarr/2026-01-28T20-41-59Z_arena_1_DefaultScreen_training.zarr` |
| 2026-01-28T20-41-59Z_arena_2:z28d84b75fc9b | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T20-41-59Z_arena_2_DefaultScreen/zarr/2026-01-28T20-41-59Z_arena_2_DefaultScreen_training.zarr` |
| 2026-01-28T20-41-59Z_arena_3:z0c3ae418e51a | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T20-41-59Z_arena_3_DefaultScreen/zarr/2026-01-28T20-41-59Z_arena_3_DefaultScreen_training.zarr` |
| 2026-01-28T20-41-59Z_arena_4:z1bc30a125d74 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T20-41-59Z_arena_4_DefaultScreen/zarr/2026-01-28T20-41-59Z_arena_4_DefaultScreen_training.zarr` |
| 2026-01-28T20-51-00Z_arena_1:z529662d10605 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T20-51-00Z_arena_1_Feeding/zarr/2026-01-28T20-51-00Z_arena_1_Feeding_training.zarr` |
| 2026-01-28T20-51-00Z_arena_2:z5b0accf196a0 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T20-51-00Z_arena_2_Feeding/zarr/2026-01-28T20-51-00Z_arena_2_Feeding_training.zarr` |
| 2026-01-28T20-51-00Z_arena_3:zd682a53e63f3 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T20-51-00Z_arena_3_Feeding/zarr/2026-01-28T20-51-00Z_arena_3_Feeding_training.zarr` |
| 2026-01-28T20-51-00Z_arena_4:z8f14a8de0d1b | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T20-51-00Z_arena_4_Feeding/zarr/2026-01-28T20-51-00Z_arena_4_Feeding_training.zarr` |
| 2026-01-28T21-18-51Z_arena_1:za048ee62747b | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T21-18-51Z_arena_1_DefaultScreen/zarr/2026-01-28T21-18-51Z_arena_1_DefaultScreen_training.zarr` |
| 2026-01-28T21-18-51Z_arena_2:z3df209582329 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T21-18-51Z_arena_2_DefaultScreen/zarr/2026-01-28T21-18-51Z_arena_2_DefaultScreen_training.zarr` |
| 2026-01-28T21-18-51Z_arena_4:zb7c04a40d97b | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T21-18-51Z_arena_4_DefaultScreen/zarr/2026-01-28T21-18-51Z_arena_4_DefaultScreen_training.zarr` |
| 2026-01-28T21-27-20Z_arena_1:z63f2ce27f5a6 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T21-27-20Z_arena_1_Feeding/zarr/2026-01-28T21-27-20Z_arena_1_Feeding_training.zarr` |
| 2026-01-28T21-27-20Z_arena_2:z8bfc50682db2 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T21-27-20Z_arena_2_Feeding/zarr/2026-01-28T21-27-20Z_arena_2_Feeding_training.zarr` |
| 2026-01-28T21-27-20Z_arena_4:ze9175336c2d0 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T21-27-20Z_arena_4_Feeding/zarr/2026-01-28T21-27-20Z_arena_4_Feeding_training.zarr` |
| 2026-01-28T21-47-47Z_arena_1:z36ae7c3bf7e1 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T21-47-47Z_arena_1_DefaultScreen/zarr/2026-01-28T21-47-47Z_arena_1_DefaultScreen_training.zarr` |
| 2026-01-28T21-47-47Z_arena_2:z4e5ca982387e | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T21-47-47Z_arena_2_DefaultScreen/zarr/2026-01-28T21-47-47Z_arena_2_DefaultScreen_training.zarr` |
| 2026-01-28T21-47-47Z_arena_3:zd155d78626cd | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T21-47-47Z_arena_3_DefaultScreen/zarr/2026-01-28T21-47-47Z_arena_3_DefaultScreen_training.zarr` |
| 2026-01-28T21-47-47Z_arena_4:z24994b002e39 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T21-47-47Z_arena_4_DefaultScreen/zarr/2026-01-28T21-47-47Z_arena_4_DefaultScreen_training.zarr` |
| 2026-01-28T21-56-23Z_arena_1:ze5b9b3bd4150 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T21-56-23Z_arena_1_Feeding/zarr/2026-01-28T21-56-23Z_arena_1_Feeding_training.zarr` |
| 2026-01-28T21-56-23Z_arena_2:zaafad399a098 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T21-56-23Z_arena_2_Feeding/zarr/2026-01-28T21-56-23Z_arena_2_Feeding_training.zarr` |
| 2026-01-28T21-56-23Z_arena_3:z8b4897188ebd | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T21-56-23Z_arena_3_Feeding/zarr/2026-01-28T21-56-23Z_arena_3_Feeding_training.zarr` |
| 2026-01-28T21-56-23Z_arena_4:z750697cddc03 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T21-56-23Z_arena_4_Feeding/zarr/2026-01-28T21-56-23Z_arena_4_Feeding_training.zarr` |
| 2026-01-28T22-15-03Z_arena_1:zd69339d8d0e5 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T22-15-03Z_arena_1_DefaultScreen/zarr/2026-01-28T22-15-03Z_arena_1_DefaultScreen_training.zarr` |
| 2026-01-28T22-15-03Z_arena_2:z87f2843620f0 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T22-15-03Z_arena_2_DefaultScreen/zarr/2026-01-28T22-15-03Z_arena_2_DefaultScreen_training.zarr` |
| 2026-01-28T22-15-04Z_arena_3:z3cfcf3829278 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T22-15-04Z_arena_3_DefaultScreen/zarr/2026-01-28T22-15-04Z_arena_3_DefaultScreen_training.zarr` |
| 2026-01-28T22-15-04Z_arena_4:zcd3cb2f2498d | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T22-15-04Z_arena_4_DefaultScreen/zarr/2026-01-28T22-15-04Z_arena_4_DefaultScreen_training.zarr` |
| 2026-01-28T22-22-57Z_arena_1:zab664f68e1f7 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T22-22-57Z_arena_1_Feeding/zarr/2026-01-28T22-22-57Z_arena_1_Feeding_training.zarr` |
| 2026-01-28T22-22-57Z_arena_2:zccd15e12cefd | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T22-22-57Z_arena_2_Feeding/zarr/2026-01-28T22-22-57Z_arena_2_Feeding_training.zarr` |
| 2026-01-28T22-22-57Z_arena_3:z2484b89ac386 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T22-22-57Z_arena_3_Feeding/zarr/2026-01-28T22-22-57Z_arena_3_Feeding_training.zarr` |
| 2026-01-28T22-22-57Z_arena_4:z857fd8fe737c | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T22-22-57Z_arena_4_Feeding/zarr/2026-01-28T22-22-57Z_arena_4_Feeding_training.zarr` |
| 2026-01-28T22-42-59Z_arena_1:zd24b5941af12 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T22-42-59Z_arena_1_DefaultScreen/zarr/2026-01-28T22-42-59Z_arena_1_DefaultScreen_training.zarr` |
| 2026-01-28T22-42-59Z_arena_2:zc2c29e634201 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T22-42-59Z_arena_2_DefaultScreen/zarr/2026-01-28T22-42-59Z_arena_2_DefaultScreen_training.zarr` |
| 2026-01-28T22-42-59Z_arena_3:z05a0f8cd9f5d | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T22-42-59Z_arena_3_DefaultScreen/zarr/2026-01-28T22-42-59Z_arena_3_DefaultScreen_training.zarr` |
| 2026-01-28T22-42-59Z_arena_4:z7063ef4d9e72 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T22-42-59Z_arena_4_DefaultScreen/zarr/2026-01-28T22-42-59Z_arena_4_DefaultScreen_training.zarr` |
| 2026-01-28T22-50-39Z_arena_1:z95dbde59aaff | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T22-50-39Z_arena_1_Feeding/zarr/2026-01-28T22-50-39Z_arena_1_Feeding_training.zarr` |
| 2026-01-28T22-50-39Z_arena_2:z8d154621d65b | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T22-50-39Z_arena_2_Feeding/zarr/2026-01-28T22-50-39Z_arena_2_Feeding_training.zarr` |
| 2026-01-28T22-50-39Z_arena_3:z8763c63de197 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T22-50-39Z_arena_3_Feeding/zarr/2026-01-28T22-50-39Z_arena_3_Feeding_training.zarr` |
| 2026-01-28T22-50-39Z_arena_4:z1f7fe556da59 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T22-50-39Z_arena_4_Feeding/zarr/2026-01-28T22-50-39Z_arena_4_Feeding_training.zarr` |
| 2026-01-28T23-07-24Z_arena_2:zb438a3d9194a | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T23-07-24Z_arena_2_DefaultScreen/zarr/2026-01-28T23-07-24Z_arena_2_DefaultScreen_training.zarr` |
| 2026-01-28T23-07-24Z_arena_3:z619ff1d52502 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T23-07-24Z_arena_3_DefaultScreen/zarr/2026-01-28T23-07-24Z_arena_3_DefaultScreen_training.zarr` |
| 2026-01-28T23-07-24Z_arena_4:z77daad5336a4 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T23-07-24Z_arena_4_DefaultScreen/zarr/2026-01-28T23-07-24Z_arena_4_DefaultScreen_training.zarr` |
| 2026-01-28T23-15-10Z_arena_2:zd4dc9e3d7f85 | source_recording | training | direct_y_like | 4 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T23-15-10Z_arena_2_Feeding/zarr/2026-01-28T23-15-10Z_arena_2_Feeding_training.zarr` |
| 2026-01-28T23-15-10Z_arena_3:z713b658550fa | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T23-15-10Z_arena_3_Feeding/zarr/2026-01-28T23-15-10Z_arena_3_Feeding_training.zarr` |
| 2026-01-28T23-15-10Z_arena_4:z730ba5b8b697 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/2026-01-28T23-15-10Z_arena_4_Feeding/zarr/2026-01-28T23-15-10Z_arena_4_Feeding_training.zarr` |
| 2026-06-23T16-01-09Z_arena_1:z92f469b75d66 | source_recording | training | direct_y_like | 2 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | `/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T16-01-09Z_arena_1_RedScare/zarr/2026-06-23T16-01-09Z_arena_1_RedScare_training.zarr` |
| 2026-06-23T16-01-09Z_arena_2:zd538265d4b4e | source_recording | training | direct_y_like | 2 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | `/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T16-01-09Z_arena_2_RedScare/zarr/2026-06-23T16-01-09Z_arena_2_RedScare_training.zarr` |
| 2026-06-23T16-01-09Z_arena_3:zd51986c19042 | source_recording | training | direct_y_like | 2 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | `/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T16-01-09Z_arena_3_RedScare/zarr/2026-06-23T16-01-09Z_arena_3_RedScare_training.zarr` |
| 2026-06-23T16-01-10Z_arena_4:zfd621a14deb0 | source_recording | training | direct_y_like | 2 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | `/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T16-01-10Z_arena_4_RedScare/zarr/2026-06-23T16-01-10Z_arena_4_RedScare_training.zarr` |
| 2026-06-23T16-43-36Z_arena_1:ze322f7126920 | source_recording | training | direct_y_like | 2 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | `/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T16-43-36Z_arena_1_RedScare/zarr/2026-06-23T16-43-36Z_arena_1_RedScare_training.zarr` |
| 2026-06-23T16-43-36Z_arena_2:z389618509690 | source_recording | training | direct_y_like | 2 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | `/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T16-43-36Z_arena_2_RedScare/zarr/2026-06-23T16-43-36Z_arena_2_RedScare_training.zarr` |
| 2026-06-23T16-43-36Z_arena_3:z729a6c424abe | source_recording | training | direct_y_like | 2 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | `/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T16-43-36Z_arena_3_RedScare/zarr/2026-06-23T16-43-36Z_arena_3_RedScare_training.zarr` |
| 2026-06-23T16-43-36Z_arena_4:za5fe2361bd57 | source_recording | training | direct_y_like | 2 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | `/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T16-43-36Z_arena_4_RedScare/zarr/2026-06-23T16-43-36Z_arena_4_RedScare_training.zarr` |
| 2026-06-23T17-16-51Z_arena_1:z6f1a9db7c1fe | source_recording | training | direct_y_like | 2 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | `/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T17-16-51Z_arena_1_RedScare/zarr/2026-06-23T17-16-51Z_arena_1_RedScare_training.zarr` |
| 2026-06-23T17-16-51Z_arena_2:zf77fd1c3dac7 | source_recording | training | direct_y_like | 2 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | `/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T17-16-51Z_arena_2_RedScare/zarr/2026-06-23T17-16-51Z_arena_2_RedScare_training.zarr` |
| 2026-06-23T17-16-51Z_arena_3:zcb0899750de3 | source_recording | training | direct_y_like | 2 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | `/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T17-16-51Z_arena_3_RedScare/zarr/2026-06-23T17-16-51Z_arena_3_RedScare_training.zarr` |
| 2026-06-23T17-16-51Z_arena_4:ze8beeb83a12a | source_recording | training | direct_y_like | 2 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | `/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T17-16-51Z_arena_4_RedScare/zarr/2026-06-23T17-16-51Z_arena_4_RedScare_training.zarr` |
| 2026-06-23T17-57-12Z_arena_1:z557718a1bed8 | source_recording | training | direct_y_like | 2 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | `/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T17-57-12Z_arena_1_RedScare/zarr/2026-06-23T17-57-12Z_arena_1_RedScare_training.zarr` |
| 2026-06-23T17-57-12Z_arena_2:ze472bcd79dc9 | source_recording | training | direct_y_like | 2 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | `/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T17-57-12Z_arena_2_RedScare/zarr/2026-06-23T17-57-12Z_arena_2_RedScare_training.zarr` |
| 2026-06-23T17-57-12Z_arena_3:zc88814883bdf | source_recording | training | direct_y_like | 2 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | `/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T17-57-12Z_arena_3_RedScare/zarr/2026-06-23T17-57-12Z_arena_3_RedScare_training.zarr` |
| 2026-06-23T17-57-12Z_arena_4:z8fc21ce574e8 | source_recording | training | direct_y_like | 2 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | `/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T17-57-12Z_arena_4_RedScare/zarr/2026-06-23T17-57-12Z_arena_4_RedScare_training.zarr` |
| 2026-06-23T18-40-24Z_arena_1:za6aff3e0ac55 | source_recording | training | direct_y_like | 2 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | `/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T18-40-24Z_arena_1_RedScare/zarr/2026-06-23T18-40-24Z_arena_1_RedScare_training.zarr` |
| 2026-06-23T18-40-24Z_arena_2:z62b434fca011 | source_recording | training | direct_y_like | 2 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | `/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T18-40-24Z_arena_2_RedScare/zarr/2026-06-23T18-40-24Z_arena_2_RedScare_training.zarr` |
| 2026-06-23T18-40-24Z_arena_3:z4b30e9cd9289 | source_recording | training | direct_y_like | 2 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | `/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T18-40-24Z_arena_3_RedScare/zarr/2026-06-23T18-40-24Z_arena_3_RedScare_training.zarr` |
| 2026-06-23T18-40-24Z_arena_4:z3d364d7f8ced | source_recording | training | direct_y_like | 2 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | `/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T18-40-24Z_arena_4_RedScare/zarr/2026-06-23T18-40-24Z_arena_4_RedScare_training.zarr` |
| 2026-06-23T20-56-02Z_arena_1:zb962d763bd65 | source_recording | training | direct_y_like | 2 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | `/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T20-56-02Z_arena_1_RedScare/zarr/2026-06-23T20-56-02Z_arena_1_RedScare_training.zarr` |
| 2026-06-23T20-56-03Z_arena_2:z54fef8898f87 | source_recording | training | direct_y_like | 2 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | `/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T20-56-03Z_arena_2_RedScare/zarr/2026-06-23T20-56-03Z_arena_2_RedScare_training.zarr` |
| 2026-06-23T20-56-03Z_arena_3:z640bfc8c003f | source_recording | training | direct_y_like | 2 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | `/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T20-56-03Z_arena_3_RedScare/zarr/2026-06-23T20-56-03Z_arena_3_RedScare_training.zarr` |
| 2026-06-23T20-56-03Z_arena_4:z92dea09a10f9 | source_recording | training | direct_y_like | 2 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | `/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T20-56-03Z_arena_4_RedScare/zarr/2026-06-23T20-56-03Z_arena_4_RedScare_training.zarr` |
| 2026-06-23T21-45-13Z_arena_1:z3cad00dbf08a | source_recording | training | direct_y_like | 2 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | `/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T21-45-13Z_arena_1_RedScare/zarr/2026-06-23T21-45-13Z_arena_1_RedScare_training.zarr` |
| 2026-06-23T21-45-13Z_arena_2:z22f4e15e0da8 | source_recording | training | direct_y_like | 2 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | `/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T21-45-13Z_arena_2_RedScare/zarr/2026-06-23T21-45-13Z_arena_2_RedScare_training.zarr` |
| 2026-06-23T21-45-13Z_arena_3:z7592c0275767 | source_recording | training | direct_y_like | 2 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | `/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T21-45-13Z_arena_3_RedScare/zarr/2026-06-23T21-45-13Z_arena_3_RedScare_training.zarr` |
| 2026-06-23T21-45-13Z_arena_4:zcbd97376b99f | source_recording | training | direct_y_like | 2 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | `/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T21-45-13Z_arena_4_RedScare/zarr/2026-06-23T21-45-13Z_arena_4_RedScare_training.zarr` |
| path-9a3024d6e0bd | derived_analysis | training | unreadable_zarr | 1 | `missing` | `missing` | `/tmp/palette_subject_mask_lr_smoke/inference_recording_minimal.zarr` |
| path-b595f3d15260 | derived_analysis | training | direct_y_like | 2 | `missing` | `missing` | `/nvme1/dan.zarr` |
| sickyfish_2026_02_23_16_23_35_cam2010093 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/sickyfish_2026_02_23_16_23_35_cam2010093/zarr/sickyfish_2026_02_23_16_23_35_cam2010093_training.zarr` |
| sickyfish_2026_02_23_16_23_35_cam2010094 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/sickyfish_2026_02_23_16_23_35_cam2010094/zarr/sickyfish_2026_02_23_16_23_35_cam2010094_training.zarr` |
| sickyfish_2026_02_23_16_23_35_cam2010095 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/sickyfish_2026_02_23_16_23_35_cam2010095/zarr/sickyfish_2026_02_23_16_23_35_cam2010095_training.zarr` |
| sickyfish_2026_02_23_16_23_35_cam2010096 | source_recording | training | direct_y_like | 3 | `missing`, `orange_mono_pynvvc_luma_uint8_v1`, `raw_video_images_full_to_uint8_grayscale` | `missing`, `pynvvc_luma` | `/nvme1/recordings/sickyfish_2026_02_23_16_23_35_cam2010096/zarr/sickyfish_2026_02_23_16_23_35_cam2010096_training.zarr` |
| sleepyfish_2026_05_05_17_45_30_cam2010093:z34e43f45bd45 | source_recording | training | direct_y_like | 2 | `missing` | `missing` | `/groups/johnson/johnsonlab/jeremy/recordings/sleepyfish_2026_05_05_17_45_30_cam2010093/zarr/sleepyfish_2026_05_05_17_45_30_cam2010093_training.zarr` |
| sleepyfish_2026_05_05_17_45_30_cam2010093:z3fdd176a8abc | source_recording | training | direct_y_like | 3 | `missing` | `missing`, `pynvvc_luma` | `/groups/johnson/johnsonlab/jeremy/recordings/sleepyfish_2026_05_05_17_45_30_cam2010093/zarr/sleepyfish_2026_05_05_17_45_30_cam2010093_clipped_training.zarr` |
| sleepyfish_2026_05_05_17_45_30_cam2010094 | source_recording | training | direct_y_like | 3 | `missing` | `missing`, `pynvvc_luma` | `/groups/johnson/johnsonlab/jeremy/recordings/sleepyfish_2026_05_05_17_45_30_cam2010094/zarr/sleepyfish_2026_05_05_17_45_30_cam2010094_clipped_training.zarr` |
| sleepyfish_2026_05_05_17_45_30_cam2010094:z59db2d8e87d1 | source_recording | training | direct_y_like | 2 | `missing` | `missing` | `/groups/johnson/johnsonlab/jeremy/recordings/sleepyfish_2026_05_05_17_45_30_cam2010094/zarr/sleepyfish_2026_05_05_17_45_30_cam2010094_training.zarr` |
| sleepyfish_2026_05_05_17_45_30_cam2010095 | source_recording | training | direct_y_like | 3 | `missing` | `missing`, `pynvvc_luma` | `/groups/johnson/johnsonlab/jeremy/recordings/sleepyfish_2026_05_05_17_45_30_cam2010095/zarr/sleepyfish_2026_05_05_17_45_30_cam2010095_clipped_training.zarr` |
| sleepyfish_2026_05_05_17_45_30_cam2010095:z04d4ebc55735 | source_recording | training | direct_y_like | 2 | `missing` | `missing` | `/groups/johnson/johnsonlab/jeremy/recordings/sleepyfish_2026_05_05_17_45_30_cam2010095/zarr/sleepyfish_2026_05_05_17_45_30_cam2010095_training.zarr` |
| sleepyfish_2026_05_05_17_45_30_cam2010096 | source_recording | training | direct_y_like | 3 | `missing` | `missing`, `pynvvc_luma` | `/groups/johnson/johnsonlab/jeremy/recordings/sleepyfish_2026_05_05_17_45_30_cam2010096/zarr/sleepyfish_2026_05_05_17_45_30_cam2010096_clipped_training.zarr` |
| sleepyfish_2026_05_05_17_45_30_cam2010096:z7f543daa9341 | source_recording | training | direct_y_like | 2 | `missing` | `missing` | `/groups/johnson/johnsonlab/jeremy/recordings/sleepyfish_2026_05_05_17_45_30_cam2010096/zarr/sleepyfish_2026_05_05_17_45_30_cam2010096_training.zarr` |

## Surface Evidence

| dataset_id | surface | shape | contract | backend | class | confidence | sampled_px | forbidden_bins_present | zero_rate | error |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| detect_all_available_detect_training_v001_merged | raw_video/images_ds | 1744x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.059831 |  |
| detect_all_available_detect_training_v002_merged | raw_video/images_ds | 12686x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.036766 |  |
| detect_all_available_detect_training_v003_merged | raw_video/images_ds | 12686x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.036766 |  |
| detect_all_available_detect_training_v004_merged | raw_video/images_ds | 14064x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.046199 |  |
| detect_cedar_shadow_omnifin0_manual_gray_85dfa0ee_v002_merged | raw_video/images_ds | 10711x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.064187 |  |
| detect_cedar_shadow_omnifin0_manual_gray_85dfa0ee_v003_merged | raw_video/images_ds | 10942x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.062369 |  |
| detect_cedar_shadow_v007_merged | raw_video/images_ds | 10942x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.062369 |  |
| detect_recording_only_sleepy_sicky_detect_v001_merged | raw_video/images_ds | 1744x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.059831 |  |
| path-6c23364be96c | crop_runs/merged_eye_masks/roi_images | 10788x512x512 | `missing` | `missing` | indeterminate | medium | 262144 | 7 | 3.1e-05 |  |
| path-81c4928a9040 | crop_runs/merged_eye_masks/roi_images | 10942x512x512 | `missing` | `missing` | indeterminate | medium | 262144 | 7 | 8e-06 |  |
| path-f1bb73bbd19c | crop_runs/merged_eye_masks/roi_images | 10788x512x512 | `missing` | `missing` | indeterminate | medium | 262144 | 7 | 3.1e-05 |  |
| pose_all_registry_reviewed_v2_keypoints_20260520_v001_merged | crop_runs/merged_export_20260521T021235Z/roi_images | 12292x512x512 | `missing` | `missing` | direct_y_like | high | 262144 | 23 | 0.093842 |  |
| pose_cedar_shadow_filtered_gray_latest_traditional_a4c30ae1_v001_merged | crop_runs/merged_export_20260227T165447Z/roi_images | 10721x512x512 | `missing` | `missing` | indeterminate | medium | 262144 | 7 | 0.10463 |  |
| pose_cedar_shadow_filtered_gray_latest_traditional_refresh_v001_merged | crop_runs/merged_export_20260304T055134Z/roi_images | 10721x512x512 | `missing` | `missing` | indeterminate | medium | 262144 | 7 | 0.10463 |  |
| pose_cedar_shadow_pose_traditional_v2_v002_merged | crop_runs/merged_export_20260330T153310Z/roi_images | 10554x512x512 | `missing` | `missing` | indeterminate | medium | 262144 | 7 | 6.9e-05 |  |
| pose_pose_box_only_smoke_v001_merged | . |  | `missing` | `missing` | missing_zarr | none | 0 |  |  | zarr_path does not exist |
| pose_pose_box_only_v1_v001_merged | crop_runs/merged_export_20260227T223609Z/roi_images | 6131x512x512 | `missing` | `missing` | indeterminate | medium | 262144 | 7 | 3.4e-05 |  |
| subject_mask_cedar_shadow_omnifin0_gray_subject_v1_union_c6ff03ae_v001_merged | crop_runs/merged_subject_masks/roi_images | 3153x512x512 | `missing` | `missing` | indeterminate | medium | 262144 | 8 | 0.021484 |  |
| 2026-01-28T19-22-28Z_arena_1:zc66de17bea1b | raw_video/images_ds | 231x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.074646 |  |
| 2026-01-28T19-22-28Z_arena_1:zc66de17bea1b | crop_runs/crop_2026-02-03_23-33-48/roi_images | 231x512x512 | `missing` | `missing` | indeterminate | medium | 262144 | 7 | 0.055172 |  |
| 2026-01-28T19-22-28Z_arena_2:ze41894008350 | raw_video/images_ds | 231x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.017709 |  |
| 2026-01-28T19-22-28Z_arena_2:ze41894008350 | crop_runs/crop_2026-02-03_23-31-56/roi_images | 231x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 5 | 6.9e-05 |  |
| 2026-01-28T19-22-28Z_arena_2:ze41894008350 | crop_runs/crop_2026-02-03_23-31-56_pynvvc_luma_v1/roi_images | 231x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 28 | 0.0 |  |
| 2026-01-28T19-22-28Z_arena_3:ze9b35d662a87 | raw_video/images_ds | 231x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.036537 |  |
| 2026-01-28T19-22-28Z_arena_3:ze9b35d662a87 | crop_runs/crop_2026-02-03_23-33-07/roi_images | 231x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 8.4e-05 |  |
| 2026-01-28T19-22-28Z_arena_3:ze9b35d662a87 | crop_runs/crop_2026-02-03_23-33-07_pynvvc_luma_v1/roi_images | 231x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 33 | 0.0 |  |
| 2026-01-28T19-22-28Z_arena_4:zd9af96ac53af | raw_video/images_ds | 231x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.060142 |  |
| 2026-01-28T19-22-28Z_arena_4:zd9af96ac53af | crop_runs/crop_2026-02-03_23-32-56/roi_images | 231x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 1.1e-05 |  |
| 2026-01-28T19-22-28Z_arena_4:zd9af96ac53af | crop_runs/crop_2026-02-03_23-32-56_pynvvc_luma_v1/roi_images | 231x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 31 | 0.0 |  |
| 2026-01-28T19-36-18Z_arena_1:zfe208d1b989b | raw_video/images_ds | 185x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 35 | 0.074876 |  |
| 2026-01-28T19-36-18Z_arena_1:zfe208d1b989b | crop_runs/crop_2026-02-03_23-31-11/roi_images | 185x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 6 | 0.019001 |  |
| 2026-01-28T19-36-18Z_arena_1:zfe208d1b989b | crop_runs/crop_2026-02-03_23-31-11_pynvvc_luma_v1/roi_images | 185x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 29 | 0.0 |  |
| 2026-01-28T19-36-18Z_arena_2:z386db9c2a8f6 | raw_video/images_ds | 185x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.01861 |  |
| 2026-01-28T19-36-18Z_arena_2:z386db9c2a8f6 | crop_runs/crop_2026-02-03_23-31-39/roi_images | 185x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 5 | 0.01524 |  |
| 2026-01-28T19-36-18Z_arena_2:z386db9c2a8f6 | crop_runs/crop_2026-02-03_23-31-39_pynvvc_luma_v1/roi_images | 185x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 27 | 0.0 |  |
| 2026-01-28T19-36-18Z_arena_3:z87d5cee5f737 | raw_video/images_ds | 185x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.040113 |  |
| 2026-01-28T19-36-18Z_arena_3:z87d5cee5f737 | crop_runs/crop_2026-02-03_23-35-24/roi_images | 185x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 0.083477 |  |
| 2026-01-28T19-36-18Z_arena_3:z87d5cee5f737 | crop_runs/crop_2026-02-03_23-35-24_pynvvc_luma_v1/roi_images | 185x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 33 | 0.061523 |  |
| 2026-01-28T19-36-18Z_arena_4:zb8e6ba77d834 | raw_video/images_ds | 185x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.059798 |  |
| 2026-01-28T19-36-18Z_arena_4:zb8e6ba77d834 | crop_runs/crop_2026-02-03_23-30-48/roi_images | 185x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 0.0 |  |
| 2026-01-28T19-36-18Z_arena_4:zb8e6ba77d834 | crop_runs/crop_2026-02-03_23-30-48_pynvvc_luma_v1/roi_images | 185x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 31 | 0.000378 |  |
| 2026-01-28T20-41-59Z_arena_1:zd3aa1c19985b | raw_video/images_ds | 229x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 35 | 0.074548 |  |
| 2026-01-28T20-41-59Z_arena_1:zd3aa1c19985b | crop_runs/crop_2026-02-03_23-30-36/roi_images | 229x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 5 | 0.0 |  |
| 2026-01-28T20-41-59Z_arena_1:zd3aa1c19985b | crop_runs/crop_2026-02-03_23-30-36_pynvvc_luma_v1/roi_images | 229x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 26 | 0.0 |  |
| 2026-01-28T20-41-59Z_arena_2:z28d84b75fc9b | raw_video/images_ds | 229x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 35 | 0.035844 |  |
| 2026-01-28T20-41-59Z_arena_2:z28d84b75fc9b | crop_runs/crop_2026-02-03_23-32-02/roi_images | 229x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 7.6e-05 |  |
| 2026-01-28T20-41-59Z_arena_2:z28d84b75fc9b | crop_runs/crop_2026-02-03_23-32-02_pynvvc_luma_v1/roi_images | 229x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 31 | 0.0 |  |
| 2026-01-28T20-41-59Z_arena_3:z0c3ae418e51a | raw_video/images_ds | 229x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.041516 |  |
| 2026-01-28T20-41-59Z_arena_3:z0c3ae418e51a | crop_runs/crop_2026-02-03_23-33-54/roi_images | 229x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 0.030273 |  |
| 2026-01-28T20-41-59Z_arena_3:z0c3ae418e51a | crop_runs/crop_2026-02-03_23-33-54_pynvvc_luma_v1/roi_images | 229x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 32 | 0.0 |  |
| 2026-01-28T20-41-59Z_arena_4:z1bc30a125d74 | raw_video/images_ds | 229x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.048296 |  |
| 2026-01-28T20-41-59Z_arena_4:z1bc30a125d74 | crop_runs/crop_2026-02-03_23-35-31/roi_images | 229x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 0.003357 |  |
| 2026-01-28T20-41-59Z_arena_4:z1bc30a125d74 | crop_runs/crop_2026-02-03_23-35-31_pynvvc_luma_v1/roi_images | 229x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 32 | 0.0 |  |
| 2026-01-28T20-51-00Z_arena_1:z529662d10605 | raw_video/images_ds | 191x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.074313 |  |
| 2026-01-28T20-51-00Z_arena_1:z529662d10605 | crop_runs/crop_2026-02-03_23-35-05/roi_images | 191x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 5 | 0.0 |  |
| 2026-01-28T20-51-00Z_arena_1:z529662d10605 | crop_runs/crop_2026-02-03_23-35-05_pynvvc_luma_v1/roi_images | 191x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 26 | 0.0 |  |
| 2026-01-28T20-51-00Z_arena_2:z5b0accf196a0 | raw_video/images_ds | 191x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 35 | 0.034752 |  |
| 2026-01-28T20-51-00Z_arena_2:z5b0accf196a0 | crop_runs/crop_2026-02-03_23-31-27/roi_images | 191x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 0.007851 |  |
| 2026-01-28T20-51-00Z_arena_2:z5b0accf196a0 | crop_runs/crop_2026-02-03_23-31-27_pynvvc_luma_v1/roi_images | 191x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 32 | 0.007812 |  |
| 2026-01-28T20-51-00Z_arena_3:zd682a53e63f3 | raw_video/images_ds | 191x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.041112 |  |
| 2026-01-28T20-51-00Z_arena_3:zd682a53e63f3 | crop_runs/crop_2026-02-03_23-30-42/roi_images | 191x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 0.093777 |  |
| 2026-01-28T20-51-00Z_arena_3:zd682a53e63f3 | crop_runs/crop_2026-02-03_23-30-42_pynvvc_luma_v1/roi_images | 191x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 33 | 0.041058 |  |
| 2026-01-28T20-51-00Z_arena_4:z8f14a8de0d1b | raw_video/images_ds | 191x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.050539 |  |
| 2026-01-28T20-51-00Z_arena_4:z8f14a8de0d1b | crop_runs/crop_2026-02-03_23-30-53/roi_images | 191x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 0.0 |  |
| 2026-01-28T20-51-00Z_arena_4:z8f14a8de0d1b | crop_runs/crop_2026-02-03_23-30-53_pynvvc_luma_v1/roi_images | 191x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 31 | 0.0 |  |
| 2026-01-28T21-18-51Z_arena_1:za048ee62747b | raw_video/images_ds | 229x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 33 | 0.069013 |  |
| 2026-01-28T21-18-51Z_arena_1:za048ee62747b | crop_runs/crop_2026-02-03_23-34-13/roi_images | 229x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 5 | 0.000172 |  |
| 2026-01-28T21-18-51Z_arena_1:za048ee62747b | crop_runs/crop_2026-02-03_23-34-13_pynvvc_luma_v1/roi_images | 229x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 29 | 0.0 |  |
| 2026-01-28T21-18-51Z_arena_2:z3df209582329 | raw_video/images_ds | 229x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 35 | 0.030439 |  |
| 2026-01-28T21-18-51Z_arena_2:z3df209582329 | crop_runs/crop_2026-02-03_23-32-44/roi_images | 229x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 0.0 |  |
| 2026-01-28T21-18-51Z_arena_2:z3df209582329 | crop_runs/crop_2026-02-03_23-32-44_pynvvc_luma_v1/roi_images | 229x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 31 | 0.0 |  |
| 2026-01-28T21-18-51Z_arena_4:zb7c04a40d97b | raw_video/images_ds | 229x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.048951 |  |
| 2026-01-28T21-18-51Z_arena_4:zb7c04a40d97b | crop_runs/crop_2026-02-03_23-33-34/roi_images | 229x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 0.0 |  |
| 2026-01-28T21-18-51Z_arena_4:zb7c04a40d97b | crop_runs/crop_2026-02-03_23-33-34_pynvvc_luma_v1/roi_images | 229x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 31 | 0.0 |  |
| 2026-01-28T21-27-20Z_arena_1:z63f2ce27f5a6 | raw_video/images_ds | 184x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 34 | 0.069253 |  |
| 2026-01-28T21-27-20Z_arena_1:z63f2ce27f5a6 | crop_runs/crop_2026-02-03_23-32-38/roi_images | 184x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 4 | 0.159904 |  |
| 2026-01-28T21-27-20Z_arena_1:z63f2ce27f5a6 | crop_runs/crop_2026-02-03_23-32-38_pynvvc_luma_v1/roi_images | 184x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 25 | 0.052734 |  |
| 2026-01-28T21-27-20Z_arena_2:z8bfc50682db2 | raw_video/images_ds | 184x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 35 | 0.0317 |  |
| 2026-01-28T21-27-20Z_arena_2:z8bfc50682db2 | crop_runs/crop_2026-02-03_23-31-22/roi_images | 184x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 6 | 0.146023 |  |
| 2026-01-28T21-27-20Z_arena_2:z8bfc50682db2 | crop_runs/crop_2026-02-03_23-31-22_pynvvc_luma_v1/roi_images | 184x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 29 | 0.144531 |  |
| 2026-01-28T21-27-20Z_arena_4:ze9175336c2d0 | raw_video/images_ds | 184x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.050053 |  |
| 2026-01-28T21-27-20Z_arena_4:ze9175336c2d0 | crop_runs/crop_2026-02-03_23-31-16/roi_images | 184x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 0.009933 |  |
| 2026-01-28T21-27-20Z_arena_4:ze9175336c2d0 | crop_runs/crop_2026-02-03_23-31-16_pynvvc_luma_v1/roi_images | 184x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 32 | 0.0 |  |
| 2026-01-28T21-47-47Z_arena_1:z36ae7c3bf7e1 | raw_video/images_ds | 231x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 35 | 0.067686 |  |
| 2026-01-28T21-47-47Z_arena_1:z36ae7c3bf7e1 | crop_runs/crop_2026-02-03_23-30-17/roi_images | 231x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 3.1e-05 |  |
| 2026-01-28T21-47-47Z_arena_1:z36ae7c3bf7e1 | crop_runs/crop_2026-02-03_23-30-17_pynvvc_luma_v1/roi_images | 231x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 31 | 0.005947 |  |
| 2026-01-28T21-47-47Z_arena_2:z4e5ca982387e | raw_video/images_ds | 231x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.03384 |  |
| 2026-01-28T21-47-47Z_arena_2:z4e5ca982387e | crop_runs/crop_2026-02-03_23-30-59/roi_images | 231x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 0.099476 |  |
| 2026-01-28T21-47-47Z_arena_2:z4e5ca982387e | crop_runs/crop_2026-02-03_23-30-59_pynvvc_luma_v1/roi_images | 231x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 31 | 0.044922 |  |
| 2026-01-28T21-47-47Z_arena_3:zd155d78626cd | raw_video/images_ds | 231x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.020367 |  |
| 2026-01-28T21-47-47Z_arena_3:zd155d78626cd | crop_runs/crop_2026-02-03_23-35-11/roi_images | 231x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 1.9e-05 |  |
| 2026-01-28T21-47-47Z_arena_3:zd155d78626cd | crop_runs/crop_2026-02-03_23-35-11_pynvvc_luma_v1/roi_images | 231x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 32 | 0.0 |  |
| 2026-01-28T21-47-47Z_arena_4:z24994b002e39 | raw_video/images_ds | 231x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.057396 |  |
| 2026-01-28T21-47-47Z_arena_4:z24994b002e39 | crop_runs/crop_2026-02-03_23-32-50/roi_images | 231x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 0.0 |  |
| 2026-01-28T21-47-47Z_arena_4:z24994b002e39 | crop_runs/crop_2026-02-03_23-32-50_pynvvc_luma_v1/roi_images | 231x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 30 | 0.0 |  |
| 2026-01-28T21-56-23Z_arena_1:ze5b9b3bd4150 | raw_video/images_ds | 187x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 34 | 0.075192 |  |
| 2026-01-28T21-56-23Z_arena_1:ze5b9b3bd4150 | crop_runs/crop_2026-02-03_23-31-50/roi_images | 187x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 5.7e-05 |  |
| 2026-01-28T21-56-23Z_arena_1:ze5b9b3bd4150 | crop_runs/crop_2026-02-03_23-31-50_pynvvc_luma_v1/roi_images | 187x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 30 | 0.000309 |  |
| 2026-01-28T21-56-23Z_arena_2:zaafad399a098 | raw_video/images_ds | 187x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.040003 |  |
| 2026-01-28T21-56-23Z_arena_2:zaafad399a098 | crop_runs/crop_2026-02-03_23-33-19/roi_images | 187x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 6 | 0.083199 |  |
| 2026-01-28T21-56-23Z_arena_2:zaafad399a098 | crop_runs/crop_2026-02-03_23-33-19_pynvvc_luma_v1/roi_images | 187x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 30 | 0.083008 |  |
| 2026-01-28T21-56-23Z_arena_3:z8b4897188ebd | raw_video/images_ds | 187x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.028474 |  |
| 2026-01-28T21-56-23Z_arena_3:z8b4897188ebd | crop_runs/crop_2026-02-03_23-34-31/roi_images | 187x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 0.0 |  |
| 2026-01-28T21-56-23Z_arena_3:z8b4897188ebd | crop_runs/crop_2026-02-03_23-34-31_pynvvc_luma_v1/roi_images | 187x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 31 | 0.0 |  |
| 2026-01-28T21-56-23Z_arena_4:z750697cddc03 | raw_video/images_ds | 187x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.064231 |  |
| 2026-01-28T21-56-23Z_arena_4:z750697cddc03 | crop_runs/crop_2026-02-03_23-34-52/roi_images | 187x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 0.0 |  |
| 2026-01-28T21-56-23Z_arena_4:z750697cddc03 | crop_runs/crop_2026-02-03_23-34-52_pynvvc_luma_v1/roi_images | 187x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 31 | 0.0 |  |
| 2026-01-28T22-15-03Z_arena_1:zd69339d8d0e5 | raw_video/images_ds | 227x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 34 | 0.076082 |  |
| 2026-01-28T22-15-03Z_arena_1:zd69339d8d0e5 | crop_runs/crop_2026-02-03_23-32-21/roi_images | 227x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 0.037853 |  |
| 2026-01-28T22-15-03Z_arena_1:zd69339d8d0e5 | crop_runs/crop_2026-02-03_23-32-21_pynvvc_luma_v1/roi_images | 227x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 32 | 0.00293 |  |
| 2026-01-28T22-15-03Z_arena_2:z87f2843620f0 | raw_video/images_ds | 227x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 35 | 0.042296 |  |
| 2026-01-28T22-15-03Z_arena_2:z87f2843620f0 | crop_runs/crop_2026-02-03_23-33-02/roi_images | 227x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 0.023571 |  |
| 2026-01-28T22-15-03Z_arena_2:z87f2843620f0 | crop_runs/crop_2026-02-03_23-33-02_pynvvc_luma_v1/roi_images | 227x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 33 | 0.0 |  |
| 2026-01-28T22-15-04Z_arena_3:z3cfcf3829278 | raw_video/images_ds | 227x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.029631 |  |
| 2026-01-28T22-15-04Z_arena_3:z3cfcf3829278 | crop_runs/crop_2026-02-03_23-33-41/roi_images | 227x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 0.030285 |  |
| 2026-01-28T22-15-04Z_arena_3:z3cfcf3829278 | crop_runs/crop_2026-02-03_23-33-41_pynvvc_luma_v1/roi_images | 227x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 34 | 0.030273 |  |
| 2026-01-28T22-15-04Z_arena_4:zcd3cb2f2498d | raw_video/images_ds | 227x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.06179 |  |
| 2026-01-28T22-15-04Z_arena_4:zcd3cb2f2498d | crop_runs/crop_2026-02-03_23-32-14/roi_images | 227x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 0.0 |  |
| 2026-01-28T22-15-04Z_arena_4:zcd3cb2f2498d | crop_runs/crop_2026-02-03_23-32-14_pynvvc_luma_v1/roi_images | 227x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 30 | 0.0 |  |
| 2026-01-28T22-22-57Z_arena_1:zab664f68e1f7 | raw_video/images_ds | 188x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.079996 |  |
| 2026-01-28T22-22-57Z_arena_1:zab664f68e1f7 | crop_runs/crop_2026-02-03_23-30-24/roi_images | 188x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 5 | 0.0 |  |
| 2026-01-28T22-22-57Z_arena_1:zab664f68e1f7 | crop_runs/crop_2026-02-03_23-30-24_pynvvc_luma_v1/roi_images | 188x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 25 | 0.000195 |  |
| 2026-01-28T22-22-57Z_arena_2:zccd15e12cefd | raw_video/images_ds | 188x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 35 | 0.043967 |  |
| 2026-01-28T22-22-57Z_arena_2:zccd15e12cefd | crop_runs/crop_2026-02-03_23-34-19/roi_images | 188x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 0.051758 |  |
| 2026-01-28T22-22-57Z_arena_2:zccd15e12cefd | crop_runs/crop_2026-02-03_23-34-19_pynvvc_luma_v1/roi_images | 188x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 32 | 0.051758 |  |
| 2026-01-28T22-22-57Z_arena_3:z2484b89ac386 | raw_video/images_ds | 188x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.032082 |  |
| 2026-01-28T22-22-57Z_arena_3:z2484b89ac386 | crop_runs/crop_2026-02-03_23-34-25/roi_images | 188x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 3.1e-05 |  |
| 2026-01-28T22-22-57Z_arena_3:z2484b89ac386 | crop_runs/crop_2026-02-03_23-34-25_pynvvc_luma_v1/roi_images | 188x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 32 | 0.0 |  |
| 2026-01-28T22-22-57Z_arena_4:z857fd8fe737c | raw_video/images_ds | 188x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.064121 |  |
| 2026-01-28T22-22-57Z_arena_4:z857fd8fe737c | crop_runs/crop_2026-02-03_23-35-18/roi_images | 188x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 0.0 |  |
| 2026-01-28T22-22-57Z_arena_4:z857fd8fe737c | crop_runs/crop_2026-02-03_23-35-18_pynvvc_luma_v1/roi_images | 188x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 30 | 0.0 |  |
| 2026-01-28T22-42-59Z_arena_1:zd24b5941af12 | raw_video/images_ds | 231x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.074794 |  |
| 2026-01-28T22-42-59Z_arena_1:zd24b5941af12 | crop_runs/crop_2026-02-03_23-34-58/roi_images | 231x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 5 | 0.126766 |  |
| 2026-01-28T22-42-59Z_arena_1:zd24b5941af12 | crop_runs/crop_2026-02-03_23-34-58_pynvvc_luma_v1/roi_images | 231x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 27 | 0.046875 |  |
| 2026-01-28T22-42-59Z_arena_2:zc2c29e634201 | raw_video/images_ds | 231x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.027044 |  |
| 2026-01-28T22-42-59Z_arena_2:zc2c29e634201 | crop_runs/crop_2026-02-03_23-30-30/roi_images | 231x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 8e-06 |  |
| 2026-01-28T22-42-59Z_arena_2:zc2c29e634201 | crop_runs/crop_2026-02-03_23-30-30_pynvvc_luma_v1/roi_images | 231x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 30 | 0.0 |  |
| 2026-01-28T22-42-59Z_arena_3:z05a0f8cd9f5d | raw_video/images_ds | 231x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.038273 |  |
| 2026-01-28T22-42-59Z_arena_3:z05a0f8cd9f5d | crop_runs/crop_2026-02-03_23-34-45/roi_images | 231x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 0.0 |  |
| 2026-01-28T22-42-59Z_arena_3:z05a0f8cd9f5d | crop_runs/crop_2026-02-03_23-34-45_pynvvc_luma_v1/roi_images | 231x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 31 | 0.0 |  |
| 2026-01-28T22-42-59Z_arena_4:z7063ef4d9e72 | raw_video/images_ds | 231x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.061228 |  |
| 2026-01-28T22-42-59Z_arena_4:z7063ef4d9e72 | crop_runs/crop_2026-02-03_23-31-33/roi_images | 231x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 0.200523 |  |
| 2026-01-28T22-42-59Z_arena_4:z7063ef4d9e72 | crop_runs/crop_2026-02-03_23-31-33_pynvvc_luma_v1/roi_images | 231x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 31 | 0.141602 |  |
| 2026-01-28T22-50-39Z_arena_1:z95dbde59aaff | raw_video/images_ds | 210x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.072228 |  |
| 2026-01-28T22-50-39Z_arena_1:z95dbde59aaff | crop_runs/crop_2026-02-03_23-34-00/roi_images | 210x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 5 | 0.232906 |  |
| 2026-01-28T22-50-39Z_arena_1:z95dbde59aaff | crop_runs/crop_2026-02-03_23-34-00_pynvvc_luma_v1/roi_images | 210x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 26 | 0.232422 |  |
| 2026-01-28T22-50-39Z_arena_2:z8d154621d65b | raw_video/images_ds | 210x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.026618 |  |
| 2026-01-28T22-50-39Z_arena_2:z8d154621d65b | crop_runs/crop_2026-02-03_23-33-26/roi_images | 210x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 4e-06 |  |
| 2026-01-28T22-50-39Z_arena_2:z8d154621d65b | crop_runs/crop_2026-02-03_23-33-26_pynvvc_luma_v1/roi_images | 210x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 34 | 0.0 |  |
| 2026-01-28T22-50-39Z_arena_3:z8763c63de197 | raw_video/images_ds | 210x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.038115 |  |
| 2026-01-28T22-50-39Z_arena_3:z8763c63de197 | crop_runs/crop_2026-02-03_23-31-44/roi_images | 210x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 0.012547 |  |
| 2026-01-28T22-50-39Z_arena_3:z8763c63de197 | crop_runs/crop_2026-02-03_23-31-44_pynvvc_luma_v1/roi_images | 210x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 31 | 0.0 |  |
| 2026-01-28T22-50-39Z_arena_4:z1f7fe556da59 | raw_video/images_ds | 210x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.061141 |  |
| 2026-01-28T22-50-39Z_arena_4:z1f7fe556da59 | crop_runs/crop_2026-02-03_23-33-13/roi_images | 210x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 4 | 0.282497 |  |
| 2026-01-28T22-50-39Z_arena_4:z1f7fe556da59 | crop_runs/crop_2026-02-03_23-33-13_pynvvc_luma_v1/roi_images | 210x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 24 | 0.075195 |  |
| 2026-01-28T23-07-24Z_arena_2:zb438a3d9194a | raw_video/images_ds | 230x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.037083 |  |
| 2026-01-28T23-07-24Z_arena_2:zb438a3d9194a | crop_runs/crop_2026-02-03_23-32-33/roi_images | 230x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 0.0 |  |
| 2026-01-28T23-07-24Z_arena_2:zb438a3d9194a | crop_runs/crop_2026-02-03_23-32-33_pynvvc_luma_v1/roi_images | 230x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 33 | 0.0 |  |
| 2026-01-28T23-07-24Z_arena_3:z619ff1d52502 | raw_video/images_ds | 230x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.046025 |  |
| 2026-01-28T23-07-24Z_arena_3:z619ff1d52502 | crop_runs/crop_2026-02-03_23-32-08/roi_images | 230x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 0.0 |  |
| 2026-01-28T23-07-24Z_arena_3:z619ff1d52502 | crop_runs/crop_2026-02-03_23-32-08_pynvvc_luma_v1/roi_images | 230x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 31 | 0.0 |  |
| 2026-01-28T23-07-24Z_arena_4:z77daad5336a4 | raw_video/images_ds | 230x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.067222 |  |
| 2026-01-28T23-07-24Z_arena_4:z77daad5336a4 | crop_runs/crop_2026-02-03_23-32-27/roi_images | 230x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 0.099667 |  |
| 2026-01-28T23-07-24Z_arena_4:z77daad5336a4 | crop_runs/crop_2026-02-03_23-32-27_pynvvc_luma_v1/roi_images | 230x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 32 | 0.095703 |  |
| 2026-01-28T23-15-10Z_arena_2:zd4dc9e3d7f85 | raw_video/images_ds | 191x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.039168 |  |
| 2026-01-28T23-15-10Z_arena_2:zd4dc9e3d7f85 | crop_runs/crop_2026-02-03_23-34-39/roi_images | 191x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 0.001514 |  |
| 2026-01-28T23-15-10Z_arena_2:zd4dc9e3d7f85 | crop_runs/crop_2026-02-03_23-34-39_pynvvc_luma_v1/roi_images | 191x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 33 | 0.0 |  |
| 2026-01-28T23-15-10Z_arena_2:zd4dc9e3d7f85 | crop_runs/crop_2026-02-03_23-34-39_pynvvc_luma_v1_smoke/roi_images | 191x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 33 | 0.0 |  |
| 2026-01-28T23-15-10Z_arena_3:z713b658550fa | raw_video/images_ds | 191x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.043623 |  |
| 2026-01-28T23-15-10Z_arena_3:z713b658550fa | crop_runs/crop_2026-02-03_23-31-05/roi_images | 191x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 0.076324 |  |
| 2026-01-28T23-15-10Z_arena_3:z713b658550fa | crop_runs/crop_2026-02-03_23-31-05_pynvvc_luma_v1/roi_images | 191x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 32 | 0.027344 |  |
| 2026-01-28T23-15-10Z_arena_4:z730ba5b8b697 | raw_video/images_ds | 191x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.066818 |  |
| 2026-01-28T23-15-10Z_arena_4:z730ba5b8b697 | crop_runs/crop_2026-02-03_23-34-07/roi_images | 191x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 0.0 |  |
| 2026-01-28T23-15-10Z_arena_4:z730ba5b8b697 | crop_runs/crop_2026-02-03_23-34-07_pynvvc_luma_v1/roi_images | 191x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 31 | 0.0 |  |
| 2026-06-23T16-01-09Z_arena_1:z92f469b75d66 | raw_video/images_ds | 200x768x1280 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 245760 | 35 | 0.4 |  |
| 2026-06-23T16-01-09Z_arena_1:z92f469b75d66 | crop_runs/crop_red_scare_acquisition_crop_video_training_2026-06-23T16-01-09Z_arena_1_RedScare/roi_images | 200x384x384 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 147456 | 32 | 0.0 |  |
| 2026-06-23T16-01-09Z_arena_2:zd538265d4b4e | raw_video/images_ds | 200x768x1280 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 245760 | 35 | 0.4 |  |
| 2026-06-23T16-01-09Z_arena_2:zd538265d4b4e | crop_runs/crop_red_scare_acquisition_crop_video_training_2026-06-23T16-01-09Z_arena_2_RedScare/roi_images | 198x384x384 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 147456 | 31 | 0.0 |  |
| 2026-06-23T16-01-09Z_arena_3:zd51986c19042 | raw_video/images_ds | 200x768x1280 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 245760 | 35 | 0.4 |  |
| 2026-06-23T16-01-09Z_arena_3:zd51986c19042 | crop_runs/crop_red_scare_acquisition_crop_video_training_2026-06-23T16-01-09Z_arena_3_RedScare/roi_images | 198x384x384 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 147456 | 32 | 0.0 |  |
| 2026-06-23T16-01-10Z_arena_4:zfd621a14deb0 | raw_video/images_ds | 200x768x1280 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 245760 | 35 | 0.4 |  |
| 2026-06-23T16-01-10Z_arena_4:zfd621a14deb0 | crop_runs/crop_red_scare_acquisition_crop_video_training_2026-06-23T16-01-10Z_arena_4_RedScare/roi_images | 199x384x384 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 147456 | 32 | 0.0 |  |
| 2026-06-23T16-43-36Z_arena_1:ze322f7126920 | raw_video/images_ds | 200x768x1280 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 245760 | 35 | 0.4 |  |
| 2026-06-23T16-43-36Z_arena_1:ze322f7126920 | crop_runs/crop_red_scare_acquisition_crop_video_training_2026-06-23T16-43-36Z_arena_1_RedScare/roi_images | 200x384x384 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 147456 | 33 | 0.0 |  |
| 2026-06-23T16-43-36Z_arena_2:z389618509690 | raw_video/images_ds | 200x768x1280 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 245760 | 34 | 0.4 |  |
| 2026-06-23T16-43-36Z_arena_2:z389618509690 | crop_runs/crop_red_scare_acquisition_crop_video_training_2026-06-23T16-43-36Z_arena_2_RedScare/roi_images | 198x384x384 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 147456 | 32 | 0.0 |  |
| 2026-06-23T16-43-36Z_arena_3:z729a6c424abe | raw_video/images_ds | 200x768x1280 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 245760 | 35 | 0.4 |  |
| 2026-06-23T16-43-36Z_arena_3:z729a6c424abe | crop_runs/crop_red_scare_acquisition_crop_video_training_2026-06-23T16-43-36Z_arena_3_RedScare/roi_images | 200x384x384 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 147456 | 32 | 0.0 |  |
| 2026-06-23T16-43-36Z_arena_4:za5fe2361bd57 | raw_video/images_ds | 200x768x1280 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 245760 | 35 | 0.4 |  |
| 2026-06-23T16-43-36Z_arena_4:za5fe2361bd57 | crop_runs/crop_red_scare_acquisition_crop_video_training_2026-06-23T16-43-36Z_arena_4_RedScare/roi_images | 200x384x384 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 147456 | 32 | 0.0 |  |
| 2026-06-23T17-16-51Z_arena_1:z6f1a9db7c1fe | raw_video/images_ds | 200x768x1280 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 245760 | 35 | 0.4 |  |
| 2026-06-23T17-16-51Z_arena_1:z6f1a9db7c1fe | crop_runs/crop_red_scare_acquisition_crop_video_training_2026-06-23T17-16-51Z_arena_1_RedScare/roi_images | 200x384x384 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 147456 | 30 | 0.0 |  |
| 2026-06-23T17-16-51Z_arena_2:zf77fd1c3dac7 | raw_video/images_ds | 200x768x1280 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 245760 | 35 | 0.4 |  |
| 2026-06-23T17-16-51Z_arena_2:zf77fd1c3dac7 | crop_runs/crop_red_scare_acquisition_crop_video_training_2026-06-23T17-16-51Z_arena_2_RedScare/roi_images | 200x384x384 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 147456 | 32 | 0.0 |  |
| 2026-06-23T17-16-51Z_arena_3:zcb0899750de3 | raw_video/images_ds | 200x768x1280 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 245760 | 35 | 0.4 |  |
| 2026-06-23T17-16-51Z_arena_3:zcb0899750de3 | crop_runs/crop_red_scare_acquisition_crop_video_training_2026-06-23T17-16-51Z_arena_3_RedScare/roi_images | 200x384x384 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 147456 | 35 | 0.0 |  |
| 2026-06-23T17-16-51Z_arena_4:ze8beeb83a12a | raw_video/images_ds | 200x768x1280 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 245760 | 35 | 0.4 |  |
| 2026-06-23T17-16-51Z_arena_4:ze8beeb83a12a | crop_runs/crop_red_scare_acquisition_crop_video_training_2026-06-23T17-16-51Z_arena_4_RedScare/roi_images | 194x384x384 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 147456 | 33 | 0.0 |  |
| 2026-06-23T17-57-12Z_arena_1:z557718a1bed8 | raw_video/images_ds | 200x768x1280 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 245760 | 35 | 0.4 |  |
| 2026-06-23T17-57-12Z_arena_1:z557718a1bed8 | crop_runs/crop_red_scare_acquisition_crop_video_training_2026-06-23T17-57-12Z_arena_1_RedScare/roi_images | 195x384x384 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 147456 | 30 | 0.0 |  |
| 2026-06-23T17-57-12Z_arena_2:ze472bcd79dc9 | raw_video/images_ds | 200x768x1280 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 245760 | 34 | 0.4 |  |
| 2026-06-23T17-57-12Z_arena_2:ze472bcd79dc9 | crop_runs/crop_red_scare_acquisition_crop_video_training_2026-06-23T17-57-12Z_arena_2_RedScare/roi_images | 200x384x384 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 147456 | 32 | 0.0 |  |
| 2026-06-23T17-57-12Z_arena_3:zc88814883bdf | raw_video/images_ds | 200x768x1280 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 245760 | 34 | 0.4 |  |
| 2026-06-23T17-57-12Z_arena_3:zc88814883bdf | crop_runs/crop_red_scare_acquisition_crop_video_training_2026-06-23T17-57-12Z_arena_3_RedScare/roi_images | 199x384x384 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 147456 | 32 | 0.0 |  |
| 2026-06-23T17-57-12Z_arena_4:z8fc21ce574e8 | raw_video/images_ds | 200x768x1280 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 245760 | 35 | 0.4 |  |
| 2026-06-23T17-57-12Z_arena_4:z8fc21ce574e8 | crop_runs/crop_red_scare_acquisition_crop_video_training_2026-06-23T17-57-12Z_arena_4_RedScare/roi_images | 200x384x384 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 147456 | 32 | 0.0 |  |
| 2026-06-23T18-40-24Z_arena_1:za6aff3e0ac55 | raw_video/images_ds | 200x768x1280 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 245760 | 35 | 0.4 |  |
| 2026-06-23T18-40-24Z_arena_1:za6aff3e0ac55 | crop_runs/crop_red_scare_acquisition_crop_video_training_2026-06-23T18-40-24Z_arena_1_RedScare/roi_images | 199x384x384 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 147456 | 33 | 0.0 |  |
| 2026-06-23T18-40-24Z_arena_2:z62b434fca011 | raw_video/images_ds | 200x768x1280 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 245760 | 34 | 0.4 |  |
| 2026-06-23T18-40-24Z_arena_2:z62b434fca011 | crop_runs/crop_red_scare_acquisition_crop_video_training_2026-06-23T18-40-24Z_arena_2_RedScare/roi_images | 199x384x384 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 147456 | 32 | 0.0 |  |
| 2026-06-23T18-40-24Z_arena_3:z4b30e9cd9289 | raw_video/images_ds | 200x768x1280 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 245760 | 35 | 0.4 |  |
| 2026-06-23T18-40-24Z_arena_3:z4b30e9cd9289 | crop_runs/crop_red_scare_acquisition_crop_video_training_2026-06-23T18-40-24Z_arena_3_RedScare/roi_images | 199x384x384 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 147456 | 32 | 0.0 |  |
| 2026-06-23T18-40-24Z_arena_4:z3d364d7f8ced | raw_video/images_ds | 200x768x1280 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 245760 | 35 | 0.4 |  |
| 2026-06-23T18-40-24Z_arena_4:z3d364d7f8ced | crop_runs/crop_red_scare_acquisition_crop_video_training_2026-06-23T18-40-24Z_arena_4_RedScare/roi_images | 186x384x384 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 147456 | 33 | 0.0 |  |
| 2026-06-23T20-56-02Z_arena_1:zb962d763bd65 | raw_video/images_ds | 200x768x1280 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 245760 | 35 | 0.4 |  |
| 2026-06-23T20-56-02Z_arena_1:zb962d763bd65 | crop_runs/crop_red_scare_acquisition_crop_video_training_2026-06-23T20-56-02Z_arena_1_RedScare/roi_images | 198x384x384 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 147456 | 33 | 0.0 |  |
| 2026-06-23T20-56-03Z_arena_2:z54fef8898f87 | raw_video/images_ds | 200x768x1280 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 245760 | 34 | 0.4 |  |
| 2026-06-23T20-56-03Z_arena_2:z54fef8898f87 | crop_runs/crop_red_scare_acquisition_crop_video_training_2026-06-23T20-56-03Z_arena_2_RedScare/roi_images | 198x384x384 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 147456 | 32 | 0.0 |  |
| 2026-06-23T20-56-03Z_arena_3:z640bfc8c003f | raw_video/images_ds | 200x768x1280 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 245760 | 34 | 0.4 |  |
| 2026-06-23T20-56-03Z_arena_3:z640bfc8c003f | crop_runs/crop_red_scare_acquisition_crop_video_training_2026-06-23T20-56-03Z_arena_3_RedScare/roi_images | 200x384x384 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 147456 | 32 | 0.0 |  |
| 2026-06-23T20-56-03Z_arena_4:z92dea09a10f9 | raw_video/images_ds | 200x768x1280 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 245760 | 35 | 0.4 |  |
| 2026-06-23T20-56-03Z_arena_4:z92dea09a10f9 | crop_runs/crop_red_scare_acquisition_crop_video_training_2026-06-23T20-56-03Z_arena_4_RedScare/roi_images | 196x384x384 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 147456 | 33 | 0.0 |  |
| 2026-06-23T21-45-13Z_arena_1:z3cad00dbf08a | raw_video/images_ds | 200x768x1280 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 245760 | 35 | 0.4 |  |
| 2026-06-23T21-45-13Z_arena_1:z3cad00dbf08a | crop_runs/crop_red_scare_acquisition_crop_video_training_2026-06-23T21-45-13Z_arena_1_RedScare/roi_images | 199x384x384 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 147456 | 31 | 0.0 |  |
| 2026-06-23T21-45-13Z_arena_2:z22f4e15e0da8 | raw_video/images_ds | 200x768x1280 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 245760 | 34 | 0.4 |  |
| 2026-06-23T21-45-13Z_arena_2:z22f4e15e0da8 | crop_runs/crop_red_scare_acquisition_crop_video_training_2026-06-23T21-45-13Z_arena_2_RedScare/roi_images | 195x384x384 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 147456 | 33 | 0.0 |  |
| 2026-06-23T21-45-13Z_arena_3:z7592c0275767 | raw_video/images_ds | 200x768x1280 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 245760 | 35 | 0.4 |  |
| 2026-06-23T21-45-13Z_arena_3:z7592c0275767 | crop_runs/crop_red_scare_acquisition_crop_video_training_2026-06-23T21-45-13Z_arena_3_RedScare/roi_images | 200x384x384 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 147456 | 32 | 0.0 |  |
| 2026-06-23T21-45-13Z_arena_4:zcbd97376b99f | raw_video/images_ds | 200x768x1280 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 245760 | 35 | 0.4 |  |
| 2026-06-23T21-45-13Z_arena_4:zcbd97376b99f | crop_runs/crop_red_scare_acquisition_crop_video_training_2026-06-23T21-45-13Z_arena_4_RedScare/roi_images | 200x384x384 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 147456 | 30 | 0.0 |  |
| path-9a3024d6e0bd | . |  | `missing` | `missing` | unreadable_zarr | none | 0 |  |  | No group found in store '/tmp/palette_subject_mask_lr_smoke/inference_recording_minimal.zarr' at path '' |
| path-b595f3d15260 | raw_video/images_ds | 102x768x1280 | `missing` | `missing` | direct_y_like | high | 245760 | 21 | 0.838542 |  |
| path-b595f3d15260 | raw_video/images_ds_rgb | 102x768x1280x3 | `missing` | `missing` | direct_y_like | high | 241560 | 23 | 0.836364 |  |
| sickyfish_2026_02_23_16_23_35_cam2010093 | raw_video/images_ds | 198x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 26 | 0.0 |  |
| sickyfish_2026_02_23_16_23_35_cam2010093 | crop_runs/crop_2026-05-16_18-31-54/roi_images | 198x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 0.0 |  |
| sickyfish_2026_02_23_16_23_35_cam2010093 | crop_runs/crop_2026-05-16_18-31-54_pynvvc_luma_v1/roi_images | 198x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 29 | 0.0 |  |
| sickyfish_2026_02_23_16_23_35_cam2010094 | raw_video/images_ds | 198x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 25 | 0.0 |  |
| sickyfish_2026_02_23_16_23_35_cam2010094 | crop_runs/crop_2026-05-16_18-35-21/roi_images | 198x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 0.0 |  |
| sickyfish_2026_02_23_16_23_35_cam2010094 | crop_runs/crop_2026-05-16_18-35-21_pynvvc_luma_v1/roi_images | 198x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 28 | 0.0 |  |
| sickyfish_2026_02_23_16_23_35_cam2010095 | raw_video/images_ds | 198x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 26 | 0.0 |  |
| sickyfish_2026_02_23_16_23_35_cam2010095 | crop_runs/crop_2026-05-16_18-35-26/roi_images | 198x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 0.0 |  |
| sickyfish_2026_02_23_16_23_35_cam2010095 | crop_runs/crop_2026-05-16_18-35-26_pynvvc_luma_v1/roi_images | 198x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 28 | 0.0 |  |
| sickyfish_2026_02_23_16_23_35_cam2010096 | raw_video/images_ds | 198x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 24 | 0.0 |  |
| sickyfish_2026_02_23_16_23_35_cam2010096 | crop_runs/crop_2026-05-16_18-35-31/roi_images | 198x512x512 | `raw_video_images_full_to_uint8_grayscale` | `missing` | indeterminate | medium | 262144 | 8 | 0.0 |  |
| sickyfish_2026_02_23_16_23_35_cam2010096 | crop_runs/crop_2026-05-16_18-35-31_pynvvc_luma_v1/roi_images | 198x512x512 | `orange_mono_pynvvc_luma_uint8_v1` | `pynvvc_luma` | direct_y_like | high | 262144 | 27 | 0.0 |  |
| sleepyfish_2026_05_05_17_45_30_cam2010093:z34e43f45bd45 | raw_video/images_ds | 238x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.119536 |  |
| sleepyfish_2026_05_05_17_45_30_cam2010093:z34e43f45bd45 | crop_runs/crop_2026-05-16_18-37-46/roi_images | 238x512x512 | `missing` | `missing` | indeterminate | medium | 262144 | 8 | 0.0 |  |
| sleepyfish_2026_05_05_17_45_30_cam2010093:z3fdd176a8abc | raw_video/images_ds | 238x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.11882 |  |
| sleepyfish_2026_05_05_17_45_30_cam2010093:z3fdd176a8abc | crop_runs/crop_2026-05-17_21-43-51/roi_images | 238x512x512 | `missing` | `missing` | direct_y_like | high | 262144 | 35 | 0.0 |  |
| sleepyfish_2026_05_05_17_45_30_cam2010093:z3fdd176a8abc | crop_runs/crop_2026-05-17_21-43-51_pynvvc_luma_v1/roi_images | 238x512x512 | `missing` | `pynvvc_luma` | direct_y_like | high | 262144 | 32 | 0.0 |  |
| sleepyfish_2026_05_05_17_45_30_cam2010094 | raw_video/images_ds | 238x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.115971 |  |
| sleepyfish_2026_05_05_17_45_30_cam2010094 | crop_runs/crop_2026-05-17_21-44-03/roi_images | 238x512x512 | `missing` | `missing` | direct_y_like | high | 262144 | 34 | 0.0 |  |
| sleepyfish_2026_05_05_17_45_30_cam2010094 | crop_runs/crop_2026-05-17_21-44-03_pynvvc_luma_v1/roi_images | 238x512x512 | `missing` | `pynvvc_luma` | direct_y_like | high | 262144 | 30 | 0.0 |  |
| sleepyfish_2026_05_05_17_45_30_cam2010094:z59db2d8e87d1 | raw_video/images_ds | 238x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.117106 |  |
| sleepyfish_2026_05_05_17_45_30_cam2010094:z59db2d8e87d1 | crop_runs/crop_2026-05-16_18-37-52/roi_images | 238x512x512 | `missing` | `missing` | indeterminate | medium | 262144 | 8 | 0.0 |  |
| sleepyfish_2026_05_05_17_45_30_cam2010095 | raw_video/images_ds | 238x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.119634 |  |
| sleepyfish_2026_05_05_17_45_30_cam2010095 | crop_runs/crop_2026-05-17_21-44-11/roi_images | 238x512x512 | `missing` | `missing` | direct_y_like | high | 262144 | 36 | 0.041016 |  |
| sleepyfish_2026_05_05_17_45_30_cam2010095 | crop_runs/crop_2026-05-17_21-44-11_pynvvc_luma_v1/roi_images | 238x512x512 | `missing` | `pynvvc_luma` | direct_y_like | high | 262144 | 33 | 0.041016 |  |
| sleepyfish_2026_05_05_17_45_30_cam2010095:z04d4ebc55735 | raw_video/images_ds | 238x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.120715 |  |
| sleepyfish_2026_05_05_17_45_30_cam2010095:z04d4ebc55735 | crop_runs/crop_2026-05-16_18-37-58/roi_images | 238x512x512 | `missing` | `missing` | indeterminate | medium | 262144 | 8 | 0.041016 |  |
| sleepyfish_2026_05_05_17_45_30_cam2010096 | raw_video/images_ds | 238x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.122374 |  |
| sleepyfish_2026_05_05_17_45_30_cam2010096 | crop_runs/crop_2026-05-17_21-44-18/roi_images | 238x512x512 | `missing` | `missing` | direct_y_like | high | 262144 | 36 | 0.067394 |  |
| sleepyfish_2026_05_05_17_45_30_cam2010096 | crop_runs/crop_2026-05-17_21-44-18_pynvvc_luma_v1/roi_images | 238x512x512 | `missing` | `pynvvc_luma` | direct_y_like | high | 262144 | 34 | 0.067383 |  |
| sleepyfish_2026_05_05_17_45_30_cam2010096:z7f543daa9341 | raw_video/images_ds | 238x640x640 | `missing` | `missing` | direct_y_like | high | 183184 | 36 | 0.123089 |  |
| sleepyfish_2026_05_05_17_45_30_cam2010096:z7f543daa9341 | crop_runs/crop_2026-05-16_18-38-03/roi_images | 238x512x512 | `missing` | `missing` | indeterminate | medium | 262144 | 8 | 0.067394 |  |
