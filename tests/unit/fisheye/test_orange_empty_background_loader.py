"""Reader for Orange's empty-dish background (contracts v1 and v2): what loads, what is refused."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import struct
import zlib

import cv2
import numpy as np
import pytest

from fisheye.shared import orange_empty_background as eb

FIXTURES = Path(__file__).resolve().parents[2] / "contracts" / "orange_empty_background_v1"
FIXTURES_V2 = FIXTURES.parent / "orange_empty_background_v2"
SERIALS = ("2010095", "2010096")
SHAPE = (48, 64)  # height, width


def _sha(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _with_chunk(png: bytes, kind: bytes, body: bytes) -> bytes:
    """Insert one chunk straight after IHDR."""

    chunk = struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
    return png[:33] + chunk + png[33:]


def _image(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(170, 256, size=SHAPE, dtype=np.uint8)
    image[:4, :4] = 0
    return image


def _recording(tmp_path: Path, *, edit=None, images=None, pngs=None) -> Path:
    """An acquisition folder with a two-camera attached capture. ``edit(descriptor, block)``
    changes the documents before they are written and hashed."""

    template = json.loads((FIXTURES / "valid" / "capture_one_captured_one_failed.json").read_text())
    captured = template["cameras"][0]
    images = images or {serial: _image(index) for index, serial in enumerate(SERIALS)}
    pngs = pngs or {serial: _png(image) for serial, image in images.items()}
    descriptor = {key: value for key, value in template.items() if key != "cameras"}
    descriptor["cameras"] = []
    block_cameras = {}
    for serial in SERIALS:
        image_entry = {
            "relative_path": f"Cam{serial}_empty_background.png",
            "size_bytes": len(pngs[serial]),
            "sha256": _sha(pngs[serial]),
        }
        camera = copy.deepcopy(captured)
        camera["camera_serial"] = serial
        camera["camera_configuration"].update(
            width=SHAPE[1], height=SHAPE[0], offset_x=0, offset_y=0
        )
        camera["image"] = image_entry
        camera["statistics"] = eb.image_levels(images[serial])
        descriptor["cameras"].append(camera)
        block_cameras[serial] = {"status": "attached", "image": dict(image_entry)}
    block = {
        "capture_selected": True,
        "selection_source": "gui_operator",
        "scene_unchanged_confirmed": True,
        "capture_id": descriptor["capture_id"],
        "source_descriptor_path": "/home/jeremy/orange_data/backgrounds/x/background.json",
        "descriptor": {"relative_path": "empty_background/background.json"},
        "cameras": block_cameras,
    }
    if edit is not None:
        edit(descriptor, block)
    descriptor_bytes = json.dumps(descriptor, indent=1).encode()
    block["descriptor"].setdefault("size_bytes", len(descriptor_bytes))
    block["descriptor"].setdefault("sha256", _sha(descriptor_bytes))
    folder = tmp_path / "empty_background"
    folder.mkdir()
    (folder / "background.json").write_bytes(descriptor_bytes)
    for serial in SERIALS:
        (folder / f"Cam{serial}_empty_background.png").write_bytes(pngs[serial])
    (tmp_path / "recording_snapshot_start.json").write_text(
        json.dumps({"session": {"empty_background": block}})
    )
    return tmp_path


def _refused(acquisition_dir: Path, code: str, serial: str = SERIALS[0], **kwargs) -> None:
    with pytest.raises(eb.EmptyBackgroundError) as raised:
        eb.load_empty_background(acquisition_dir, serial, **kwargs)
    assert raised.value.code == code


def test_an_attached_background_loads_the_named_cameras_pixels(tmp_path: Path) -> None:
    images = {serial: _image(index) for index, serial in enumerate(SERIALS)}
    acquisition_dir = _recording(tmp_path, images=images)
    for serial in SERIALS:
        background = eb.load_empty_background(acquisition_dir, serial)
        assert background.status == "attached" and background.usable
        assert np.array_equal(background.image, images[serial])
        assert background.image.dtype == np.uint8 and not background.image.flags.writeable
        assert background.capture_id == "2026_11_02_15_04_11"
        assert background.camera_configuration["exposure_us"] == 150
        assert background.statistics == eb.image_levels(images[serial])
    assert not np.array_equal(images[SERIALS[0]], images[SERIALS[1]])


def test_image_levels_follow_the_contracts_exact_definitions() -> None:
    image = np.array([[0, 0, 10, 10], [10, 200, 255, 255]], dtype=np.uint8)
    assert eb.image_levels(image) == {
        "saturation_value": 255,
        "pixel_count": 8,
        "saturated_fraction": 0.25,
        "zero_fraction": 0.25,
        "mean": 92.5,  # 740 / 8
        "p01": 0,  # rank ceil(10 * 8 / 1000) = 1
        "p05": 0,
        "p50": 10,  # rank 4: cumulative count reaches 5 at level 10
        "p95": 255,  # rank 8
        "p99": 255,
        "p999": 255,
    }
    thirds = np.array([[1, 1, 2]], dtype=np.uint8)
    assert eb.image_levels(thirds)["mean"] == 1.333
    assert eb.image_levels(np.array([[0, 5, 5]], dtype=np.uint8))["zero_fraction"] == 0.33333333


def test_a_recording_without_the_block_is_undeclared(tmp_path: Path) -> None:
    (tmp_path / "recording_snapshot_start.json").write_text(json.dumps({"session": {}}))
    background = eb.load_empty_background(tmp_path, SERIALS[0])
    assert background.status == eb.UNDECLARED and not background.usable


def test_a_declared_absence_is_an_answer_not_an_error(tmp_path: Path) -> None:
    block = json.loads((FIXTURES / "valid" / "snapshot_not_selected.json").read_text())
    background = eb.load_empty_background(
        tmp_path, "2010095", start_snapshot={"session": {"empty_background": block}}
    )
    assert background.status == "not_selected" and background.image is None


def test_a_camera_missing_from_the_capture_has_no_image(tmp_path: Path) -> None:
    block = json.loads(
        (FIXTURES / "valid" / "snapshot_settings_changed_and_missing.json").read_text()
    )
    background = eb.load_empty_background(
        tmp_path, "2010096", start_snapshot={"session": {"empty_background": block}}
    )
    assert background.status == "missing_in_capture" and not background.usable
    assert background.capture_id == "2026_11_02_15_04_11"


def test_settings_changed_is_refused_unless_the_consumer_accepts_it(tmp_path: Path) -> None:
    differences = [{"setting": "iris", "capture_value": 19, "recording_value": 20}]

    def edit(descriptor, block):
        block["cameras"][SERIALS[0]].update(
            status="settings_changed", settings_differences=differences
        )

    acquisition_dir = _recording(tmp_path, edit=edit)
    _refused(acquisition_dir, "empty_background_settings_changed")
    background = eb.load_empty_background(
        acquisition_dir, SERIALS[0], accept_settings_changed=True
    )
    assert background.status == "settings_changed" and background.usable
    assert list(background.settings_differences) == differences
    assert eb.load_empty_background(acquisition_dir, SERIALS[1]).status == "attached"


def _as_v2_block(block) -> None:
    template = json.loads((FIXTURES_V2 / "valid" / "snapshot_local_control_attached.json").read_text())
    block.update(
        schema_version=2, selection_source="local_control", requester=template["requester"]
    )


def _as_v2_descriptor(descriptor) -> None:
    template = json.loads((FIXTURES_V2 / "valid" / "capture_local_control.json").read_text())
    descriptor.update(
        schema_version=2,
        requester=dict(template["requester"], request_id="capture-req-0001"),
        scene_declaration=template["scene_declaration"],
    )


def test_a_version_2_block_with_a_version_2_descriptor_loads(tmp_path: Path) -> None:
    def edit(descriptor, block):
        _as_v2_descriptor(descriptor)
        _as_v2_block(block)

    images = {serial: _image(index) for index, serial in enumerate(SERIALS)}
    background = eb.load_empty_background(_recording(tmp_path, edit=edit, images=images), SERIALS[0])
    assert background.usable and np.array_equal(background.image, images[SERIALS[0]])
    assert (background.block_version, background.descriptor_version) == (2, 2)
    assert background.selection_source == "local_control"
    # The recording's requester names the start request, the descriptor's the capture request.
    assert background.selection_requester["request_id"] == "specsail-req-0001"
    assert background.capture_requester["request_id"] == "capture-req-0001"
    assert background.scene_declaration["declared_by"] == "jeremy"


def test_a_version_2_block_may_attach_a_version_1_descriptor(tmp_path: Path) -> None:
    background = eb.load_empty_background(
        _recording(tmp_path, edit=lambda d, block: _as_v2_block(block)), SERIALS[0]
    )
    assert background.usable
    assert (background.block_version, background.descriptor_version) == (2, 1)
    assert background.capture_requester is None


def test_a_version_1_recording_reports_its_versions_and_no_requester(tmp_path: Path) -> None:
    background = eb.load_empty_background(_recording(tmp_path), SERIALS[0])
    assert (background.block_version, background.descriptor_version) == (1, 1)
    assert background.selection_requester is None and background.capture_requester is None


def test_a_version_1_block_with_a_version_2_descriptor_is_refused(tmp_path: Path) -> None:
    acquisition_dir = _recording(tmp_path, edit=lambda d, block: _as_v2_descriptor(d))
    _refused(acquisition_dir, "empty_background_v1_block_with_v2_descriptor")


def test_a_version_2_block_without_its_requester_is_refused(tmp_path: Path) -> None:
    acquisition_dir = _recording(tmp_path, edit=lambda d, block: block.update(schema_version=2))
    _refused(acquisition_dir, "empty_background_block_schema")


@pytest.mark.parametrize("version", [3, "2", True])
def test_an_unknown_block_version_is_refused_by_name(tmp_path: Path, version) -> None:
    acquisition_dir = _recording(tmp_path, edit=lambda d, block: block.update(schema_version=version))
    _refused(acquisition_dir, "empty_background_block_version_unsupported")


def test_an_unknown_descriptor_version_is_refused_by_name(tmp_path: Path) -> None:
    acquisition_dir = _recording(tmp_path, edit=lambda d, block: d.update(schema_version=3))
    _refused(acquisition_dir, "empty_background_descriptor_version_unsupported")


def test_a_declared_absence_under_version_2_is_an_answer(tmp_path: Path) -> None:
    block = json.loads((FIXTURES_V2 / "valid" / "snapshot_not_selected.json").read_text())
    serial = next(iter(block["cameras"]))
    background = eb.load_empty_background(
        tmp_path, serial, start_snapshot={"session": {"empty_background": block}}
    )
    assert background.status == "not_selected" and not background.usable
    assert background.block_version == 2 and background.selection_requester is None


def test_a_stored_statistic_one_ulp_from_its_decimal_still_matches(tmp_path: Path) -> None:
    # Orange builds before 335c863d could write 180.96200000000002 for 180.962.
    def edit(descriptor, block):
        statistics = descriptor["cameras"][0]["statistics"]
        for name in ("mean", "saturated_fraction"):
            nudged = float(np.nextafter(statistics[name], np.inf))
            assert nudged != statistics[name]
            statistics[name] = nudged

    assert eb.load_empty_background(_recording(tmp_path, edit=edit), SERIALS[0]).usable


def test_a_statistic_that_differs_at_its_stated_decimals_is_refused(tmp_path: Path) -> None:
    def edit(descriptor, block):
        descriptor["cameras"][0]["statistics"]["mean"] += 0.001

    _refused(_recording(tmp_path, edit=edit), "empty_background_statistics_mismatch")


def test_a_block_that_breaks_its_schema_is_refused(tmp_path: Path) -> None:
    block = json.loads(
        (FIXTURES / "invalid" / "snapshot_attached_without_image.json").read_text()
    )
    with pytest.raises(eb.EmptyBackgroundError) as raised:
        eb.declaration({"session": {"empty_background": block}})
    assert raised.value.code == "empty_background_block_schema"


def test_a_camera_the_block_does_not_list_is_refused(tmp_path: Path) -> None:
    _refused(_recording(tmp_path), "empty_background_camera_not_in_block", serial="2010093")


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("Cam2010095_empty_background.png", "empty_background_image_sha256_mismatch"),
        ("background.json", "empty_background_descriptor_sha256_mismatch"),
    ],
)
def test_changed_bytes_are_refused(tmp_path: Path, name: str, code: str) -> None:
    acquisition_dir = _recording(tmp_path)
    path = acquisition_dir / "empty_background" / name
    data = bytearray(path.read_bytes())
    data[-20] ^= 1
    path.write_bytes(bytes(data))
    _refused(acquisition_dir, code)


def test_a_missing_image_file_is_refused(tmp_path: Path) -> None:
    acquisition_dir = _recording(tmp_path)
    (acquisition_dir / "empty_background" / "Cam2010095_empty_background.png").unlink()
    _refused(acquisition_dir, "empty_background_image_missing")


def test_a_descriptor_for_another_capture_is_refused(tmp_path: Path) -> None:
    acquisition_dir = _recording(
        tmp_path, edit=lambda d, block: d.update(capture_id="2026_11_03_09_00_00")
    )
    _refused(acquisition_dir, "empty_background_capture_id_mismatch")


def test_a_block_image_that_differs_from_the_descriptors_is_refused(tmp_path: Path) -> None:
    def edit(descriptor, block):
        descriptor["cameras"][0]["image"]["sha256"] = "sha256:" + "ab" * 32

    _refused(_recording(tmp_path, edit=edit), "empty_background_image_differs_from_descriptor")


def test_an_image_named_for_another_camera_is_refused(tmp_path: Path) -> None:
    def edit(descriptor, block):
        descriptor["cameras"][1]["image"]["relative_path"] = "Cam2010095_empty_background.png"

    _refused(
        _recording(tmp_path, edit=edit), "empty_background_descriptor_image_names_other_camera"
    )


def test_a_camera_listed_twice_in_the_descriptor_is_refused(tmp_path: Path) -> None:
    def edit(descriptor, block):
        descriptor["cameras"][1]["camera_serial"] = SERIALS[0]

    _refused(
        _recording(tmp_path, edit=edit), "empty_background_descriptor_duplicate_camera_serial"
    )


def test_source_frames_that_skip_are_refused(tmp_path: Path) -> None:
    def edit(descriptor, block):
        descriptor["cameras"][1]["source_frames"][5]["stream_frame_id"] += 100
        for frame in descriptor["cameras"][1]["source_frames"][6:]:
            frame["stream_frame_id"] += 100
        descriptor["cameras"][1]["source_frames"][9]["stream_frame_id"] += 1

    _refused(_recording(tmp_path, edit=edit), "empty_background_source_frames_not_consecutive")


def test_a_png_whose_raster_differs_from_the_declared_one_is_refused(tmp_path: Path) -> None:
    def edit(descriptor, block):
        descriptor["cameras"][0]["camera_configuration"]["width"] = SHAPE[1] + 2

    _refused(_recording(tmp_path, edit=edit), "empty_background_png_raster_mismatch")


def test_statistics_that_the_pixels_do_not_reproduce_are_refused(tmp_path: Path) -> None:
    def edit(descriptor, block):
        descriptor["cameras"][0]["statistics"]["p50"] -= 1

    _refused(_recording(tmp_path, edit=edit), "empty_background_statistics_mismatch")


def test_a_png_with_a_gamma_chunk_is_refused(tmp_path: Path) -> None:
    images = {serial: _image(index) for index, serial in enumerate(SERIALS)}
    pngs = {serial: _png(image) for serial, image in images.items()}
    pngs[SERIALS[0]] = _with_chunk(pngs[SERIALS[0]], b"gAMA", struct.pack(">I", 45455))
    _refused(
        _recording(tmp_path, images=images, pngs=pngs), "empty_background_png_colour_chunk"
    )


def test_a_16_bit_png_is_refused(tmp_path: Path) -> None:
    images = {serial: _image(index) for index, serial in enumerate(SERIALS)}
    pngs = {serial: _png(image) for serial, image in images.items()}
    pngs[SERIALS[0]] = _png(images[SERIALS[0]].astype(np.uint16) * 257)
    _refused(
        _recording(tmp_path, images=images, pngs=pngs), "empty_background_png_not_8bit_greyscale"
    )


def test_a_colour_png_is_refused(tmp_path: Path) -> None:
    images = {serial: _image(index) for index, serial in enumerate(SERIALS)}
    pngs = {serial: _png(image) for serial, image in images.items()}
    pngs[SERIALS[0]] = _png(np.dstack([images[SERIALS[0]]] * 3))
    _refused(
        _recording(tmp_path, images=images, pngs=pngs), "empty_background_png_not_8bit_greyscale"
    )
