"""Orange's empty-dish background: the declaration, the descriptor and the image.

Before the fish go in, Orange can capture a per-pixel median of 17 frames per
camera and, at record start, copy it into the recording as
``empty_background/background.json`` plus one PNG per camera. The start
snapshot's ``session.empty_background`` block says, per camera serial, whether
that copy can be used (agent-contracts ``orange-palette/empty_background_v1``
and ``empty_background_v2``, which adds who requested the capture and the
recording so a remote client can do both with no person at the Orange GUI).

This module reads one camera's background from a recording's acquisition
folder and checks everything the contract asks a reader to check: both
schemas, the consumer rules JSON Schema cannot express (v1 README rule 9, v2
README rules 7 and 9), the bytes against both declared digests, the PNG against
the declared raster, and the statistics recomputed from the pixels. It reads
files and raises; it writes nothing.

The background is a prior, not a truth: the dish can move when the fish are
added, so a consumer still has to compare it with the recording itself.

A block without ``schema_version`` is version 1 and carries a version-1
descriptor. A version-2 block carries a descriptor of either version. Any
other version, and a version-1 block with a version-2 descriptor, is refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from functools import lru_cache
import hashlib
from importlib.resources import files as resource_files
import json
from pathlib import Path
import struct
from typing import Any, Mapping
import zlib

import numpy as np

CAPTURE_SCHEMA_ID = "orange.empty_background_capture"
FOLDER_NAME = "empty_background"
DESCRIPTOR_NAME = "background.json"
DESCRIPTOR_RELATIVE_PATH = f"{FOLDER_NAME}/{DESCRIPTOR_NAME}"
START_SNAPSHOT_NAME = "recording_snapshot_start.json"

# Statuses of the snapshot block (v1 README rule 2), plus "undeclared" for a
# recording older than the contract, whose start snapshot has no block.
UNDECLARED = "undeclared"
STATUSES_WITH_IMAGE = ("attached", "settings_changed")
STATUSES_WITHOUT_IMAGE = ("not_selected", "missing_in_capture", "copy_failed")

# Byte-identical to agent-contracts 7601cef2 (v1) and 603da1fe (v2); see
# contracts/README.md.
_SCHEMAS = {
    "capture_v2": (
        "orange_empty_background_capture_v2.schema.json",
        "c80c9888e7575d908d8fb82b0485dd3fc0cded684fc56109ea27de1ae843de29",
    ),
    "recording_v2": (
        "orange_recording_empty_background_v2.schema.json",
        "88edb64b480048c07f0e08342e9503b00b5f7ee9c95078dbf1423f127a1d0f99",
    ),
    "requester_v1": (
        "orange_requester_v1.schema.json",
        "58819f17c23feb0bb526c7254d088af24434899ced57b3093c3c8a64bdc4a687",
    ),
    "capture_v1": (
        "orange_empty_background_capture_v1.schema.json",
        "b6b147c5c3f04b5db4e79e1e1659453559c263ea5934088e717e88ca2f64bf03",
    ),
    "levels_v1": (
        "orange_image_levels_v1.schema.json",
        "8e0ac8488d3a7a4c86da0545b63639bb005df8f742c65922fae92068f1e94ef2",
    ),
    "recording_v1": (
        "orange_recording_empty_background_v1.schema.json",
        "5ea84ae2052304a3d26c9fae6cbb6f56b15f4798059683e792f1003d69c86933",
    ),
}

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# Colour-management chunks the contract forbids: the PNG holds raw sensor values.
_FORBIDDEN_PNG_CHUNKS = (b"gAMA", b"iCCP", b"sRGB")
_PERCENTILE_PER_MILLE = {"p01": 10, "p05": 50, "p50": 500, "p95": 950, "p99": 990, "p999": 999}
# Decimals each rounded statistic is compared at (v2 README rule 9).
_ROUNDED_STATISTICS = {"saturated_fraction": 8, "zero_fraction": 8, "mean": 3}


class EmptyBackgroundError(ValueError):
    """The recording's empty background breaks its contract; ``code`` names the rule."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def _require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise EmptyBackgroundError(code, detail)


@dataclass(frozen=True)
class EmptyBackground:
    """One camera's empty-dish background, as far as the recording declares one.

    ``image`` is the median as raw sensor Mono8 values (``uint8``, height by
    width, the video's native raster) when ``status`` is ``attached`` or
    ``settings_changed``, else ``None``. Every other capture field is ``None``
    when the recording attached no capture.

    ``block_version`` and ``descriptor_version`` are the contract versions the
    recording's block and the capture descriptor were written under.
    ``selection_requester`` is who asked for this capture to be attached to
    the recording and ``capture_requester`` who asked for the capture, each as
    Orange's transport saw it; both are ``None`` under version 1.
    ``scene_declaration`` carries the claimed ``declared_by`` under version 2.
    """

    camera_serial: str
    status: str
    reason: str | None = None
    image: np.ndarray | None = None
    capture_id: str | None = None
    captured_at_utc: str | None = None
    descriptor_sha256: str | None = None
    image_sha256: str | None = None
    camera_configuration: Mapping[str, Any] | None = None
    settings_differences: tuple[Mapping[str, Any], ...] = ()
    statistics: Mapping[str, Any] | None = None
    any_frame_saturated_fraction: float | None = None
    scene_declaration: Mapping[str, Any] | None = None
    block_version: int | None = None
    descriptor_version: int | None = None
    selection_source: str | None = None
    selection_requester: Mapping[str, Any] | None = None
    capture_requester: Mapping[str, Any] | None = None

    @property
    def usable(self) -> bool:
        return self.image is not None


@lru_cache(maxsize=None)
def _schema(name: str) -> dict[str, Any]:
    file_name, pin = _SCHEMAS[name]
    data = resource_files("fisheye.shared").joinpath("contracts").joinpath(file_name).read_bytes()
    _require(
        hashlib.sha256(data).hexdigest() == pin,
        "empty_background_schema_pin_mismatch",
        file_name,
    )
    return json.loads(data)


@lru_cache(maxsize=None)
def _validator(name: str):
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource

    # The schemas reference the levels and requester schemas by $id, so the set
    # resolves together.
    registry = Registry().with_resources(
        (_schema(key)["$id"], Resource.from_contents(_schema(key))) for key in _SCHEMAS
    )
    return Draft202012Validator(_schema(name), registry=registry)


def _require_schema(name: str, document: Any, code: str) -> None:
    error = next(iter(_validator(name).iter_errors(document)), None)
    if error is not None:
        path = "/".join(str(part) for part in error.absolute_path)
        raise EmptyBackgroundError(code, f"{path or '<root>'}: {error.message[:200]}")


def _strict_json(data: bytes, code: str) -> Any:
    def refuse_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        keys = [key for key, _ in pairs]
        _require(len(set(keys)) == len(keys), code, "duplicate JSON key")
        return dict(pairs)

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=refuse_duplicates)
    except (UnicodeDecodeError, ValueError) as exc:
        raise EmptyBackgroundError(code, str(exc)[:200]) from exc


def _read_declared_file(path: Path, declared: Mapping[str, Any], code: str) -> bytes:
    """The file's bytes, after its size and sha256 equal what ``declared`` says."""

    _require(path.is_file() and not path.is_symlink(), f"{code}_missing", path.name)
    data = path.read_bytes()
    _require(len(data) == declared["size_bytes"], f"{code}_size_mismatch", path.name)
    _require(
        f"sha256:{hashlib.sha256(data).hexdigest()}" == declared["sha256"],
        f"{code}_sha256_mismatch",
        path.name,
    )
    return data


def declaration(start_snapshot: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The validated ``session.empty_background`` block, or ``None`` when undeclared.

    A recording older than the contract has no block. A block without
    ``schema_version`` is version 1 (v2 README rule 4). A block that names a
    grammar this module does not vendor is refused, never read as version 1.
    """

    session = start_snapshot.get("session")
    block = session.get("empty_background") if isinstance(session, Mapping) else None
    if block is None:
        return None
    _require(isinstance(block, Mapping), "empty_background_block_not_object")
    version = block_version(block)
    _require(
        version in (1, 2),
        "empty_background_block_version_unsupported",
        repr(block.get("schema_version")),
    )
    _require_schema(f"recording_v{version}", block, "empty_background_block_schema")
    return block


def block_version(block: Mapping[str, Any]) -> int | None:
    """The block's contract version: 1 when it has no ``schema_version`` key."""

    if "schema_version" not in block:
        return 1
    version = block["schema_version"]
    return version if type(version) is int else None


def _image_name(camera_serial: str) -> str:
    return f"Cam{camera_serial}_empty_background.png"


def _descriptor_camera(descriptor: Mapping[str, Any], camera_serial: str) -> Mapping[str, Any]:
    """The descriptor's entry for one camera, after the v1 rule 9 checks on all of them."""

    serials = [camera["camera_serial"] for camera in descriptor["cameras"]]
    _require(
        len(set(serials)) == len(serials), "empty_background_descriptor_duplicate_camera_serial"
    )
    for camera in descriptor["cameras"]:
        serial = camera["camera_serial"]
        if "image" in camera:
            _require(
                camera["image"]["relative_path"] == _image_name(serial),
                "empty_background_descriptor_image_names_other_camera",
                f"{serial}: {camera['image']['relative_path']}",
            )
        frame_ids = [frame["stream_frame_id"] for frame in camera.get("source_frames", ())]
        _require(
            all(later - earlier == 1 for earlier, later in zip(frame_ids, frame_ids[1:])),
            "empty_background_source_frames_not_consecutive",
            serial,
        )
    matches = [camera for camera in descriptor["cameras"] if camera["camera_serial"] == camera_serial]
    _require(
        len(matches) == 1 and matches[0]["capture_status"] == "captured" and "image" in matches[0],
        "empty_background_descriptor_has_no_image_for_camera",
        camera_serial,
    )
    return matches[0]


def decode_png_mono8(data: bytes) -> np.ndarray:
    """Decode a contract PNG: 8-bit greyscale, no alpha, no colour-management chunks."""

    _require(data[:8] == _PNG_SIGNATURE, "empty_background_png_not_png")
    offset = 8
    header: tuple[int, ...] | None = None
    ended = False
    while offset + 8 <= len(data) and not ended:
        length, kind = struct.unpack(">I4s", data[offset : offset + 8])
        body = data[offset + 8 : offset + 8 + length]
        crc = data[offset + 8 + length : offset + 12 + length]
        _require(
            len(body) == length
            and len(crc) == 4
            and struct.unpack(">I", crc)[0] == zlib.crc32(kind + body),
            "empty_background_png_chunk_corrupt",
            kind.decode("latin-1"),
        )
        _require(
            kind not in _FORBIDDEN_PNG_CHUNKS,
            "empty_background_png_colour_chunk",
            kind.decode("latin-1"),
        )
        if kind == b"IHDR":
            _require(header is None and length == 13, "empty_background_png_header")
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"IEND":
            ended = True
        offset += 12 + length
    _require(header is not None and ended, "empty_background_png_truncated")
    width, height, bit_depth, colour_type, compression, filter_method, interlace = header
    _require(
        (bit_depth, colour_type, compression, filter_method, interlace) == (8, 0, 0, 0, 0),
        "empty_background_png_not_8bit_greyscale",
        f"bit_depth={bit_depth} colour_type={colour_type} interlace={interlace}",
    )
    import cv2

    # The chunk walk above proved the container; the pixels are decoded by OpenCV,
    # which applies no gamma or colour conversion to an 8-bit greyscale PNG.
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    _require(
        image is not None and image.dtype == np.uint8 and image.shape == (height, width),
        "empty_background_png_data_corrupt",
    )
    return image


def image_levels(image: np.ndarray) -> dict[str, Any]:
    """``orange.image_levels`` v1 statistics, exact to the contract's definitions."""

    counts = np.bincount(image.reshape(-1), minlength=256).astype(object)
    pixel_count = int(sum(counts))
    cumulative = np.cumsum(counts)

    def rounded(value: Fraction, decimals: int) -> float:
        quantum = Decimal(1).scaleb(-decimals)
        exact = Decimal(value.numerator) / Decimal(value.denominator)
        return float(exact.quantize(quantum, rounding=ROUND_HALF_UP))

    levels: dict[str, Any] = {
        "saturation_value": 255,
        "pixel_count": pixel_count,
        "saturated_fraction": rounded(Fraction(int(counts[255]), pixel_count), 8),
        "zero_fraction": rounded(Fraction(int(counts[0]), pixel_count), 8),
        "mean": rounded(
            Fraction(sum(level * int(counts[level]) for level in range(256)), pixel_count), 3
        ),
    }
    for name, per_mille in _PERCENTILE_PER_MILLE.items():
        rank = -(-per_mille * pixel_count // 1000)
        levels[name] = int(np.searchsorted(cumulative, rank, side="left"))
    return levels


def _comparable_statistics(statistics: Mapping[str, Any]) -> dict[str, Any]:
    """Statistics with each rounded value re-rounded to its stated decimals.

    Orange builds before 335c863d could serialize a rounded value one ulp from
    its shortest decimal (``180.96200000000002`` for 180.962), so a stored
    fraction or mean is compared after rounding, half away from zero; counts
    and percentiles compare exactly (v2 README rule 9).
    """

    comparable = dict(statistics)
    for name, decimals in _ROUNDED_STATISTICS.items():
        value = comparable.get(name)
        if type(value) in (int, float):
            quantum = Decimal(1).scaleb(-decimals)
            comparable[name] = float(Decimal(repr(value)).quantize(quantum, rounding=ROUND_HALF_UP))
    return comparable


def load_empty_background(
    acquisition_dir: Path,
    camera_serial: str,
    *,
    start_snapshot: Mapping[str, Any] | None = None,
    accept_settings_changed: bool = False,
) -> EmptyBackground:
    """One camera's empty-dish background from a recording's acquisition folder.

    ``acquisition_dir`` holds ``recording_snapshot_start.json`` and, when a
    capture was attached, ``empty_background/``; in the library that is
    ``<recording>/raw/acquisition``. The folder carries every camera's image,
    so the camera is chosen by serial, never by path.

    Returns an ``EmptyBackground`` whose ``status`` says what the recording
    declares. A recording with no usable image (undeclared, not selected,
    missing in the capture, copy failed) is an answer, not an error. A
    declared image that breaks the contract raises ``EmptyBackgroundError``.
    ``settings_changed`` raises too unless ``accept_settings_changed`` is set
    by a consumer that handles ``settings_differences`` itself.
    """

    camera_serial = str(camera_serial)
    if start_snapshot is None:
        snapshot_path = acquisition_dir / START_SNAPSHOT_NAME
        _require(snapshot_path.is_file(), "empty_background_start_snapshot_missing")
        start_snapshot = _strict_json(
            snapshot_path.read_bytes(), "empty_background_start_snapshot_not_json"
        )
        _require(
            isinstance(start_snapshot, Mapping), "empty_background_start_snapshot_not_json"
        )
    block = declaration(start_snapshot)
    if block is None:
        return EmptyBackground(camera_serial, UNDECLARED)
    version = block_version(block)
    entry = block["cameras"].get(camera_serial)
    _require(entry is not None, "empty_background_camera_not_in_block", camera_serial)
    status = entry["status"]
    if status in STATUSES_WITHOUT_IMAGE:
        return EmptyBackground(
            camera_serial,
            status,
            reason=entry.get("failure_reason"),
            capture_id=block.get("capture_id"),
            block_version=version,
            selection_source=block.get("selection_source"),
            selection_requester=block.get("requester"),
        )
    _require(status in STATUSES_WITH_IMAGE, "empty_background_status_unknown", status)
    _require(
        status == "attached" or accept_settings_changed,
        "empty_background_settings_changed",
        json.dumps(entry.get("settings_differences"), sort_keys=True),
    )

    _require(
        block["descriptor"]["relative_path"] == DESCRIPTOR_RELATIVE_PATH,
        "empty_background_descriptor_path",
        block["descriptor"]["relative_path"],
    )
    folder = acquisition_dir / FOLDER_NAME
    descriptor_bytes = _read_declared_file(
        folder / DESCRIPTOR_NAME, block["descriptor"], "empty_background_descriptor"
    )
    descriptor = _strict_json(descriptor_bytes, "empty_background_descriptor_not_json")
    _require(isinstance(descriptor, Mapping), "empty_background_descriptor_not_json")
    _require(
        descriptor.get("schema_id") == CAPTURE_SCHEMA_ID
        and type(descriptor.get("schema_version")) is int,
        "empty_background_descriptor_schema",
        "schema_id or schema_version",
    )
    descriptor_version = descriptor["schema_version"]
    _require(
        descriptor_version in (1, 2),
        "empty_background_descriptor_version_unsupported",
        repr(descriptor_version),
    )
    # A version-2 block may attach either descriptor; a version-1 block only its own.
    _require(
        descriptor_version <= version,
        "empty_background_v1_block_with_v2_descriptor",
    )
    _require_schema(
        f"capture_v{descriptor_version}", descriptor, "empty_background_descriptor_schema"
    )
    _require(
        descriptor["capture_id"] == block["capture_id"],
        "empty_background_capture_id_mismatch",
        f"descriptor {descriptor['capture_id']} block {block['capture_id']}",
    )

    camera = _descriptor_camera(descriptor, camera_serial)
    image_entry = entry["image"]
    _require(
        image_entry["relative_path"] == _image_name(camera_serial),
        "empty_background_block_image_names_other_camera",
        image_entry["relative_path"],
    )
    _require(
        image_entry["size_bytes"] == camera["image"]["size_bytes"]
        and image_entry["sha256"] == camera["image"]["sha256"],
        "empty_background_image_differs_from_descriptor",
        camera_serial,
    )
    image_bytes = _read_declared_file(
        folder / _image_name(camera_serial), image_entry, "empty_background_image"
    )
    image = decode_png_mono8(image_bytes)
    configuration = camera["camera_configuration"]
    _require(
        image.shape == (configuration["height"], configuration["width"]),
        "empty_background_png_raster_mismatch",
        f"png {image.shape[1]}x{image.shape[0]} "
        f"declared {configuration['width']}x{configuration['height']}",
    )
    recomputed = image_levels(image)
    _require(
        recomputed == _comparable_statistics(camera["statistics"]),
        "empty_background_statistics_mismatch",
        ", ".join(
            f"{key}: declared {camera['statistics'].get(key)!r} recomputed {value!r}"
            for key, value in recomputed.items()
            if _comparable_statistics(camera["statistics"]).get(key) != value
        ),
    )
    image.setflags(write=False)
    return EmptyBackground(
        camera_serial,
        status,
        image=image,
        capture_id=descriptor["capture_id"],
        captured_at_utc=descriptor["captured_at_utc"],
        descriptor_sha256=block["descriptor"]["sha256"],
        image_sha256=image_entry["sha256"],
        camera_configuration=configuration,
        settings_differences=tuple(entry.get("settings_differences", ())),
        statistics=camera["statistics"],
        any_frame_saturated_fraction=camera.get("any_frame_saturated_fraction"),
        scene_declaration=descriptor["scene_declaration"],
        block_version=version,
        descriptor_version=descriptor_version,
        selection_source=block.get("selection_source"),
        selection_requester=block.get("requester"),
        capture_requester=descriptor.get("requester"),
    )
