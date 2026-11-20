"""Ingest Orange's empty-dish background into a recording's ``background_runs``.

``compute_background`` builds a background from the recording's own frames.
Orange can instead capture one before the fish go in and seal it into the
recording (``raw/acquisition/empty_background``). This module stores that
image as a ``background_runs/<run>`` entry, so the existing background
consumers find it through the usual ``latest`` selector.

The run carries ``background_full`` only: the image at the camera's native
raster, byte-for-byte the pixels of the sealed PNG. No ``background_ds`` is
written, because an analysis archive has no ``images_ds`` raster to match.

The capture is a prior, not a truth: the dish can move when the fish are
added. Consumers that need the dish position still fit it on the recording.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import zarr

from ..shared.orange_empty_background import (
    EmptyBackground,
    EmptyBackgroundError,
    load_empty_background,
)
from ..shared.run_provenance import build_writer_run_provenance
from ..shared.zarr_run_completion import (
    is_run_complete_in_parent,
    mark_run_complete,
    mark_run_started,
    note_pending_latest,
    require_runs_parent,
)

METHOD = "orange_empty_background"
RUN_PREFIX = "background_orange_empty_"
# Root attr recording what the import found, whether or not a run was written.
RECORD_ATTR = "orange_empty_background_import"

STATUS_IMPORTED = "imported"
STATUS_ALREADY_IMPORTED = "already_imported"
STATUS_REFUSED = "refused"


def run_name_for(capture_id: str) -> str:
    return f"{RUN_PREFIX}{capture_id}"


def _capture_attrs(background: EmptyBackground) -> dict[str, Any]:
    return {
        "capture_id": background.capture_id,
        "captured_at_utc": background.captured_at_utc,
        "camera_serial": background.camera_serial,
        "status": background.status,
        "descriptor_sha256": background.descriptor_sha256,
        "image_sha256": background.image_sha256,
        "block_version": background.block_version,
        "descriptor_version": background.descriptor_version,
        "selection_source": background.selection_source,
        "selection_requester": _plain(background.selection_requester),
        "capture_requester": _plain(background.capture_requester),
        "scene_declaration": _plain(background.scene_declaration),
        "camera_configuration": _plain(background.camera_configuration),
        "statistics": _plain(background.statistics),
        "any_frame_saturated_fraction": background.any_frame_saturated_fraction,
    }


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def import_orange_empty_background(
    root: zarr.Group,
    *,
    acquisition_dir: Path,
    camera_serial: str,
    start_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Store this camera's sealed empty-dish background as a background run.

    ``root`` is the recording's analysis archive, open for writing without
    consolidated metadata. Returns the record also written to the root attr
    ``orange_empty_background_import``:

    - ``imported``: a new run holds the image and is now ``latest``;
    - ``already_imported``: the run for this capture already holds this image;
    - ``refused``: a declared capture breaks its contract (``code`` names the
      rule) or its camera settings changed before the recording; no run;
    - the loader's own status (``undeclared``, ``not_selected``,
      ``missing_in_capture``, ``copy_failed``): the recording has no image.

    A refusal does not raise: the recording is still imported, without a
    background, and the record says why.
    """

    try:
        background = load_empty_background(
            Path(acquisition_dir), str(camera_serial), start_snapshot=start_snapshot
        )
    except EmptyBackgroundError as exc:
        record = {
            "status": STATUS_REFUSED,
            "code": exc.code,
            "detail": str(exc)[:500],
            "camera_serial": str(camera_serial),
        }
        root.attrs[RECORD_ATTR] = record
        return record
    if not background.usable:
        record = {
            "status": background.status,
            "reason": background.reason,
            "capture_id": background.capture_id,
            "camera_serial": background.camera_serial,
            "block_version": background.block_version,
        }
        root.attrs[RECORD_ATTR] = record
        return record

    image = np.asarray(background.image)
    run_name = run_name_for(str(background.capture_id))
    parent = require_runs_parent(root, "background_runs")
    record = {
        "status": STATUS_IMPORTED,
        "run_name": run_name,
        "capture_id": background.capture_id,
        "camera_serial": background.camera_serial,
        "image_sha256": background.image_sha256,
        "descriptor_sha256": background.descriptor_sha256,
    }
    if run_name in parent:
        existing = parent[run_name]
        same = (
            is_run_complete_in_parent(parent, existing)
            and dict(existing.attrs.get("orange_empty_background") or {}).get("image_sha256")
            == background.image_sha256
            and "background_full" in existing
            and np.array_equal(existing["background_full"][:], image)
        )
        if not same:
            raise ValueError(
                f"background_runs/{run_name} exists and is not this capture's complete image"
            )
        record["status"] = STATUS_ALREADY_IMPORTED
        root.attrs[RECORD_ATTR] = record
        return record

    group = parent.create_group(run_name)
    mark_run_started(group, run_name=run_name, stage="background")
    note_pending_latest(parent, run_name)
    # One chunk, like the computed backgrounds (schema.create_background_arrays).
    array = group.create_array(
        "background_full",
        shape=image.shape,
        chunks=image.shape,
        dtype="uint8",
        fill_value=0,
        serializer=zarr.codecs.BytesCodec(),
        compressors=[zarr.codecs.BloscCodec(cname="lz4", clevel=1, shuffle="bitshuffle")],
    )
    array[:] = image
    if not np.array_equal(array[:], image):
        raise ValueError(f"background_runs/{run_name}/background_full does not read back equal")
    parameters = {
        "method": METHOD,
        "compute_full": True,
        "compute_ds": False,
    }
    capture = _capture_attrs(background)
    group.attrs.update(
        {
            "background_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "method": METHOD,
            "background_source": "pre_recording_empty_dish_capture",
            "parameters": parameters,
            "orange_empty_background": capture,
        }
    )
    mark_run_complete(
        group,
        parent_group=parent,
        run_name=run_name,
        run_provenance=build_writer_run_provenance(
            command="import_orange_empty_background",
            params=parameters,
            input_run_ids={
                "orange_empty_background_capture_id": background.capture_id,
                "orange_empty_background_descriptor_sha256": background.descriptor_sha256,
                "orange_empty_background_image_sha256": background.image_sha256,
            },
        ),
    )
    root.attrs[RECORD_ATTR] = record
    return record


__all__ = [
    "METHOD",
    "RECORD_ATTR",
    "RUN_PREFIX",
    "import_orange_empty_background",
    "run_name_for",
]
