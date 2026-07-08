# Camera Metadata Integration

This document describes how camera configuration metadata should be stored in H5
at acquisition time and mirrored into Zarr for downstream training and audits.

## H5 layout (source of truth)

Preferred layout:

- Group: /camera_metadata
- Dataset: config_json (UTF-8 JSON string)

Fallback layouts supported:
- Group attrs under /camera_metadata
- Group name /device_metadata with the same structure

## Recommended schema (non-PII)

Store only configuration and device fields (no credentials, no file paths):

- name
- device_vendor_name
- device_model_name
- device_serial_number
- device_firmware_version
- width, height
- pixel_format, adc, bin
- offset_x, offset_y
- frame_rate, gain, exposure
- lens_name, lens_focal_length
- lens_mount_present, lens_present, lens_busy
- lens_mount_firmware_version
- sens_temp
- color_temp
- gpu_id
- gpu_direct
- color

Optional metadata:
- schema_version
- captured_at_utc
- camera_config_hash (sha256 of canonical JSON)

## Do not store

- device_snmp_comm_read
- device_snmp_comm_write
- yolo (model path)

## Zarr mirror (post-import)

When importing stimulus metadata into Zarr, mirror the camera metadata into:

- analysis_metadata.camera_metadata (JSON string)
- analysis_metadata.camera_config_hash (optional)

This allows training manifests and registry scans to reference capture-time
camera settings without needing the original H5.

## Training manifests

Training manifests should include:
- camera_config_hash
- selected camera fields or the full camera_metadata payload (optional)
- video codec metadata if available (codec, pix_fmt)
