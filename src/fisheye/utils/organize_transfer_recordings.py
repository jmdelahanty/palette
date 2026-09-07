"""Inventory-complete parent organization for the opt-in Citrus transfer v2.

Original acquisition artifacts are immutable evidence, including unfamiliar
session context. Every staged member gets explicit durable destinations; source
retirement is a separate verified transaction step, never an incidental glob.
Legacy organizer/default identity and serialization contracts are unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager
import errno
import fcntl
import os
from pathlib import Path
import shutil
import stat
import tempfile

from fisheye.shared.json_safety import write_json_atomic

from fisheye.shared.recording_geometry import (
    RECORDING_GEOMETRY_BUNDLE_RELATIVE_PATH,
    verify_recording_geometry_bundle,
)
from fisheye.shared.recording_geometry_bundle import (
    iter_recording_geometry_bundle_files,
)
from fisheye.shared.recording_manifest_context import (
    validate_recording_manifest_context,
)
from fisheye.shared.recording_preflight import default_preflight_payload
from fisheye.shared.recording_transfer_snapshot import (
    MARKER_NAME,
    SNAPSHOT_PATH,
    file_ref,
    plan_parent_recordings,
    require,
    strict_json,
    verify_transfer_snapshot,
)
from fisheye.shared.source_recording_identity import (
    SOURCE_RECORDING_ID_MAPPING_PROFILE,
    SourceRecordingIdentity,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.utils.organize_recordings import (
    _read_camera_context,
    _recording_geometry_bundle_source,
)

PLAN_SCHEMA_ID = "palette.transfer_parent_organization_plan.v1"
ARTIFACT_SCHEMA_ID = "orange_transfer_parent_v1"
INDEX_DIRECTORY = "derived/recording_frame_index"


def _separate_destination(source: Path, destination: Path) -> Path:
    candidate = Path(destination).absolute()
    require(
        not any(path.is_symlink() for path in (candidate, *candidate.parents)),
        "organization destination contains a symlink",
    )
    candidate = candidate.resolve()
    require(
        candidate != source
        and source not in candidate.parents
        and candidate not in source.parents,
        "organization destination must be separate from the staging source tree",
    )
    require(
        not candidate.exists() or candidate.is_dir(),
        "organization destination is not a directory",
    )
    return candidate


def build_transfer_organization_plan(
    recording_dir: Path,
    *,
    destination_root: Path,
    recording_type: str,
    recording_subtype: str,
    behavior_mode: str,
) -> dict:
    """Assign all inventory/control bytes to exact camera parents, without writes.

    The three scientific context fields are explicit operator inputs under the
    existing manifest vocabulary, not guessed from H5 presence or video names.
    H5 camera/session mapping uses the existing producer-context reader. Other
    session context is preserved for every parent, never dropped or relabeled as
    camera-specific evidence. Source retention is not implied by this plan.
    """
    context = {
        "recording_type": recording_type,
        "recording_subtype": recording_subtype,
        "behavior_mode": behavior_mode,
        "artifact_schema_id": ARTIFACT_SCHEMA_ID,
    }
    validate_recording_manifest_context(context)
    current = verify_transfer_snapshot(recording_dir)
    parents = plan_parent_recordings(current)
    source = current.root
    destination = _separate_destination(source, destination_root)
    inventory = {item["path"]: dict(item) for item in current.snapshot["inventory"]}
    for control in (MARKER_NAME, SNAPSHOT_PATH):
        require(control not in inventory, "transfer control overlaps payload inventory")
        inventory[control] = file_ref(source, control)
    # Citrus excludes its namespace controls from scientific payload inventory.
    # Preserve every regular control byte through an explicit organization-plan
    # binding as well; the copied lock is historical context, not a live lease.
    source_files, _ = _source_members(source)
    for relative in sorted(set(source_files) - set(inventory)):
        require(relative.startswith("_citrus_transfer/"), "unplanned source artifact")
        inventory[relative] = file_ref(source, relative)
    session = strict_json(source / "recording_session.json")

    # Explicit producer refs, not filenames, own camera artifacts. All other
    # payload members stay durable session context unless the H5 owner binds
    # them to one exact camera. No partial camera snapshot is manufactured.
    camera_owners: dict[str, tuple[str, str]] = {}
    for source_parent in current.snapshot["parents"]:
        camera = source_parent["parent_key"]["camera_serial"]
        for clip in source_parent["clips"]:
            for output in clip["outputs"]:
                refs = [
                    output[field]
                    for field in ("video", "metadata", "container_finalization")
                ]
                refs.extend(sidecar["artifact"] for sidecar in output["sidecars"])
                for artifact in refs:
                    relative = artifact["path"]
                    owner = camera_owners.setdefault(
                        relative, (camera, "camera_output")
                    )
                    require(
                        owner == (camera, "camera_output"),
                        "conflicting camera artifact ownership",
                    )

    by_camera = {parent.camera_id: parent for parent in parents}
    h5_by_camera: dict[str, str] = {}
    producer_context: dict[str, dict] = {}
    for relative in inventory:
        if Path(relative).suffix.lower() not in (".h5", ".hdf5"):
            continue
        camera, metadata = _read_camera_context(source / relative)
        require(
            camera in by_camera and "error" not in metadata,
            f"H5 lacks a readable exact camera binding: {relative}",
        )
        declared_session = metadata.get("session_uuid")
        require(
            declared_session is None
            or declared_session == by_camera[camera].session_uuid,
            f"H5 session identity differs from its transfer: {relative}",
        )
        require(
            camera not in h5_by_camera,
            f"multiple H5 sources for one parent camera: {camera}",
        )
        require(
            relative not in camera_owners,
            "H5 overlaps a declared camera media artifact",
        )
        camera_owners[relative] = (camera, "camera_h5")
        h5_by_camera[camera] = relative
        producer_context[camera] = dict(metadata)

    geometry_source = _recording_geometry_bundle_source(source)
    geometry_files: set[str] = set()
    geometry = None
    if geometry_source is not None:
        checked = verify_recording_geometry_bundle(
            geometry_source, require_snapshot_pointer=False, verify_all_assets=True
        )
        geometry_files = {
            path.relative_to(source).as_posix()
            for path in iter_recording_geometry_bundle_files(geometry_source)
        }
        require(
            geometry_files <= set(inventory),
            "geometry member is not in the transfer inventory",
        )
        require(
            not geometry_files.intersection(camera_owners),
            "geometry and camera ownership conflict",
        )
        geometry = {
            "schema_id": "palette.recording_geometry_bundle.v1",
            "relative_path": RECORDING_GEOMETRY_BUNDLE_RELATIVE_PATH.as_posix(),
            "source_root": str(geometry_source),
            "contract_sha256": checked.contract_sha256,
            "manifest_sha256": checked.manifest_sha256,
            "manifest_file_count": checked.manifest_file_count,
            "materialized_asset_status": checked.materialized_asset_status.value,
            "snapshot_pointer_status": checked.snapshot_pointer_status,
            "verification_status": "verified",
        }

    records = []
    for parent in parents:
        identity = SourceRecordingIdentity(
            parent.recording_id,
            parent.session_uuid,
            parent.camera_id,
            SOURCE_RECORDING_ID_MAPPING_PROFILE,
        )
        record = {
            "identity": identity.manifest_fields(),
            "recording_name": f"{parent.session_uuid}_Cam{parent.camera_id}",
            "destination_dir": str(destination / parent.recording_id),
            "recording_layout": parent.recording_layout,
            "total_frames": parent.total_frames,
            "clip_count": len(parent.clips),
            "output_kinds": sorted(
                output.output_kind for output in parent.clips[0].outputs
            ),
            "context": dict(context),
            "context_source": "explicit_operator_recording_context",
            "producer_context": producer_context.get(parent.camera_id, {}),
            "h5_relative_path": (
                f"raw/acquisition/{h5_by_camera[parent.camera_id]}"
                if parent.camera_id in h5_by_camera
                else None
            ),
            "recording_geometry_bundle": geometry,
        }
        records.append(record)

    files = []
    targets_seen: set[tuple[str, str]] = set()
    for relative, artifact in sorted(inventory.items()):
        camera, role = camera_owners.get(relative, (None, "session_context"))
        selected = parents if camera is None else (by_camera[camera],)
        if relative in geometry_files:
            role = "geometry_bundle"
            prefix = RECORDING_GEOMETRY_BUNDLE_RELATIVE_PATH.as_posix()
        elif role == "camera_output":
            prefix = "cams/acquisition"
        else:
            prefix = "raw/acquisition"
        targets = []
        for parent in selected:
            target = f"{prefix}/{relative}"
            if relative == "ptp_sync_summary.json" and role == "session_context":
                # The clock owner has an established canonical summary locator.
                # Preserve these original bytes, not a new inferred clock claim.
                target = "raw/ptp_sync_summary.json"
            key = (parent.recording_id, target)
            require(key not in targets_seen, "duplicate organized destination")
            targets_seen.add(key)
            targets.append(
                {"recording_id": parent.recording_id, "relative_path": target}
            )
        files.append(
            {
                "source": artifact,
                "role": role,
                "camera_id": camera,
                "destinations": targets,
            }
        )
    require(len(files) == len(inventory), "organization omitted source files")
    plan = {
        "schema_id": PLAN_SCHEMA_ID,
        "source_dir": str(source),
        "destination_root": str(destination),
        "snapshot_id": current.snapshot_id,
        "transfer_attempt_id": current.attempt_id,
        "recording_layout": current.recording_layout,
        "recording_payload_kind": current.recording_payload_kind,
        "acquisition_session_id": current.snapshot["acquisition_session_id"],
        "orange_producer": session.get("producer"),
        "context": context,
        "parents": records,
        "files": files,
        "status": "organization_planned",
        "import_complete": False,
        "staging_finalized": False,
    }
    return {**plan, "plan_sha256": canonical_json_sha256(plan)}


def _directory_identity(path: Path) -> list[int]:
    require(
        not any(p.is_symlink() for p in (path, *path.parents)),
        "directory contains a symlink",
    )
    current = path.lstat()
    require(stat.S_ISDIR(current.st_mode), "expected a real directory")
    return [current.st_dev, current.st_ino]


def _require_directory(path: Path, expected: list[int]) -> None:
    require(_directory_identity(path) == expected, f"directory ownership lost: {path}")


def _state_directory(plan: dict) -> Path:
    return (
        Path(plan["destination_root"])
        / ".transfer_intake"
        / plan["snapshot_id"].removeprefix("sha256:")
    )


def _validate_plan(plan: dict, *, live_source: bool) -> None:
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    require(plan.get("schema_id") == PLAN_SCHEMA_ID, "invalid organization plan schema")
    require(
        canonical_json_sha256(unsigned) == plan.get("plan_sha256"),
        "organization plan digest differs",
    )
    source = Path(plan["source_dir"])
    destination = Path(plan["destination_root"])
    require(
        str(source.resolve()) == str(source)
        and str(destination.resolve()) == str(destination),
        "organization plan paths must be canonical absolute paths",
    )
    _directory_identity(source)
    _separate_destination(source, destination)
    if live_source:
        context = plan["context"]
        rebuilt = build_transfer_organization_plan(
            Path(plan["source_dir"]),
            destination_root=Path(plan["destination_root"]),
            **{
                key: context[key]
                for key in ("recording_type", "recording_subtype", "behavior_mode")
            },
        )
        require(
            rebuilt == plan, "organization plan differs from live source or context"
        )


def _fsync_regular_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        require(
            stat.S_ISREG(os.fstat(descriptor).st_mode),
            "durability target is not regular",
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    identity = _directory_identity(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        require(
            [opened.st_dev, opened.st_ino] == identity, "durability directory changed"
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_directory_chain(directory: Path) -> None:
    # mkdir(parents=True) may have introduced any of these entries.
    for path in (directory, *directory.parents):
        _fsync_directory(path)


@contextmanager
def _coordinator_lock(plan: dict, *, kind: str):
    directory = _state_directory(plan)
    directory.parent.mkdir(parents=True, exist_ok=True)
    lock_directory_identity = _directory_identity(directory.parent)
    lock_path = directory.with_suffix(kind)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_stat = os.fstat(handle.fileno())
        require(stat.S_ISREG(lock_stat.st_mode), "coordinator lock is not regular")

        def verify_lock():
            _require_directory(directory.parent, lock_directory_identity)
            current = lock_path.lstat()
            require(
                (current.st_dev, current.st_ino)
                == (lock_stat.st_dev, lock_stat.st_ino),
                "coordinator lock ownership lost",
            )

        verify_lock()
        yield verify_lock, handle.fileno()
        verify_lock()
        # Close, never explicit LOCK_UN: an inherited writer descriptor must
        # retain the lease if its supervisor exits before the writer does.


@contextmanager
def transfer_parent_workflow_lock(plan: dict):
    """Exclude a second cooperating importer for the entire prepare/import/retire."""
    _validate_plan(plan, live_source=False)
    with _coordinator_lock(plan, kind=".workflow.lock") as verify:
        yield verify


@contextmanager
def _organization_state(plan: dict):
    """Short recovery-state transaction, separate from the encompassing lease."""
    directory = _state_directory(plan)
    with _coordinator_lock(plan, kind=".lock") as (verify_lock, _lock_fd):
        if directory.exists():
            state = strict_json(directory / "organization_state.json")
            require(state.get("plan") == plan, "existing coordinator has another plan")
            _require_directory(directory, state["state_directory_identity"])
        else:
            directory.mkdir(exist_ok=False)
            state = {
                "schema_id": "palette.transfer_organization_state.v1",
                "plan": plan,
                "state_directory_identity": _directory_identity(directory),
                "source_directory_identity": _directory_identity(
                    Path(plan["source_dir"])
                ),
                "parent_directory_identities": {},
                "status": "reserved",
                "import_complete": False,
                "staging_finalized": False,
            }
            write_json_atomic(
                directory / "organization_state.json", state, overwrite=False
            )
            _sync_directory_chain(directory)

        def save():
            _require_directory(directory, state["state_directory_identity"])
            verify_lock()
            write_json_atomic(directory / "organization_state.json", state)
            _sync_directory_chain(directory)

        yield state, save


def _matches_file(root: Path, relative: str, expected: dict) -> None:
    path = root / relative
    require(
        not any(p.is_symlink() for p in (path, *path.parents)),
        "artifact path contains a symlink",
    )
    actual = file_ref(root, relative)
    require(
        (actual["sha256"], actual["size_bytes"])
        == (expected["sha256"], expected["size_bytes"]),
        f"organized/source artifact differs: {path}",
    )


def _materialize_file(source: Path, destination: Path, expected: dict) -> None:
    """Atomic no-clobber link, or bounded copy across filesystems."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    _directory_identity(destination.parent)
    if destination.exists() or destination.is_symlink():
        _matches_file(destination.parent, destination.name, expected)
        return
    _matches_file(source.parent, source.name, expected)
    try:
        os.link(source, destination, follow_symlinks=False)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        # A unique temporary belongs to this invocation only. Never overwrite
        # or remove a pre-existing destination on failure or retry.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, suffix=".intake-partial", delete=False
            ) as output:
                temporary = Path(output.name)
                with source.open("rb") as input_file:
                    shutil.copyfileobj(input_file, output, length=4 * 1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            _matches_file(temporary.parent, temporary.name, expected)
            os.link(temporary, destination, follow_symlinks=False)
        finally:
            if temporary is not None:
                temporary.unlink()
    _matches_file(destination.parent, destination.name, expected)


def _verify_materialized_files(plan: dict, state: dict) -> None:
    destination = Path(plan["destination_root"])
    for parent in plan["parents"]:
        key = parent["identity"]["recording_id"]
        _require_directory(destination / key, state["parent_directory_identities"][key])
    for item in plan["files"]:
        for target in item["destinations"]:
            _matches_file(
                destination / target["recording_id"],
                target["relative_path"],
                item["source"],
            )


def _flush_parent_publications(plan: dict, state: dict) -> None:
    """Make payloads, import products and all directory entries durable first.

    Receipt validation proves content, not persistence across power loss. This
    barrier is required even for hardlinks and on a partial-retirement retry.
    Only the exact owned parent trees are traversed; symlinks are refused.
    """
    directories = set()
    for parent in plan["parents"]:
        root = Path(parent["destination_dir"])
        _require_directory(
            root,
            state["parent_directory_identities"][parent["identity"]["recording_id"]],
        )
        for directory, names, filenames in os.walk(root, followlinks=False):
            current = Path(directory)
            _directory_identity(current)
            directories.add(current)
            for name in names:
                _directory_identity(current / name)
            for name in filenames:
                _fsync_regular_file(current / name)
        directories.update(root.parents)
    for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        _fsync_directory(directory)


def materialize_transfer_organization(plan: dict) -> dict:
    """Preserve the entire source inventory in exact owned parent directories.

    This does not admit a recording or retire staging. Retry verifies the same
    source generation, reservation identities and all already-written bytes.
    Final parents are reserved before index/receipt creation: persisted absolute
    paths and live-file fingerprints must never be invalidated by later rename.
    """
    _validate_plan(plan, live_source=True)
    with _organization_state(plan) as (state, save):
        require(
            state["status"] in {"reserved", "materialized"},
            "organization is already past materialization",
        )
        source = Path(plan["source_dir"])
        _require_directory(source, state["source_directory_identity"])
        for parent in plan["parents"]:
            key = parent["identity"]["recording_id"]
            directory = Path(parent["destination_dir"])
            expected = state["parent_directory_identities"].get(key)
            if expected is None:
                directory.mkdir(exist_ok=False)
                state["parent_directory_identities"][key] = _directory_identity(
                    directory
                )
                save()
            else:
                _require_directory(directory, expected)
        for item in plan["files"]:
            for target in item["destinations"]:
                parent = Path(plan["destination_root"]) / target["recording_id"]
                _require_directory(
                    parent, state["parent_directory_identities"][target["recording_id"]]
                )
                _materialize_file(
                    source / item["source"]["path"],
                    parent / target["relative_path"],
                    item["source"],
                )
        _validate_plan(plan, live_source=True)
        _verify_materialized_files(plan, state)
        state["status"] = "materialized"
        save()
        return state


def resolve_materialized_parent_sources(
    plan: dict, *, camera_id: str
) -> tuple[Path, dict[str, str]]:
    """Read and verify the organizer's exact source-to-parent path projection."""
    _validate_plan(plan, live_source=True)
    directory = _state_directory(plan)
    state = strict_json(directory / "organization_state.json")
    require(state.get("plan") == plan, "organization state has another plan")
    _require_directory(directory, state["state_directory_identity"])
    require(state["status"] == "materialized", "parent sources are not materialized")
    _verify_materialized_files(plan, state)
    parents = [p for p in plan["parents"] if p["identity"]["camera_id"] == camera_id]
    require(len(parents) == 1, "camera does not select one organized parent")
    parent = parents[0]
    mapping = {
        item["source"]["path"]: target["relative_path"]
        for item in plan["files"]
        for target in item["destinations"]
        if target["recording_id"] == parent["identity"]["recording_id"]
    }
    return Path(parent["destination_dir"]), mapping


def _verify_parent_index(plan: dict, parent: dict) -> dict:
    directory = Path(parent["destination_dir"]) / INDEX_DIRECTORY
    manifest = strict_json(directory / "recording_frame_index_manifest.json")
    binding = manifest.get("source_transfer", {})
    require(
        manifest.get("status") == "ok"
        and manifest.get("dry_run") is False
        and manifest.get("row_count") == parent["total_frames"]
        and manifest.get("camera_serials") == [parent["identity"]["camera_id"]]
        and binding.get("organization_plan_sha256") == plan["plan_sha256"]
        and binding.get("snapshot_id") == plan["snapshot_id"]
        and binding.get("recording_id") == parent["identity"]["recording_id"]
        and binding.get("organized_recording_root") == parent["destination_dir"],
        "parent index does not bind its exact organization plan",
    )
    for filename, field in (
        ("recording_frame_index.parquet", "parquet_sha256"),
        ("recording_clip_index.json", "recording_clip_index_sha256"),
    ):
        actual = file_ref(directory, filename)
        require(
            actual["sha256"] == manifest.get(field),
            "parent index payload digest differs",
        )
    projections = manifest.get("projected_clip_manifests")
    require(
        isinstance(projections, list) and len(projections) == parent["clip_count"],
        "parent index projection inventory differs",
    )
    for artifact in projections:
        require(
            file_ref(directory, artifact["path"]) == artifact,
            "parent clip projection digest differs",
        )
    return manifest


def _parent_manifest(plan: dict, parent: dict) -> dict:
    key = parent["identity"]["recording_id"]
    files = {"raw": [], "cams": [], "derived": [INDEX_DIRECTORY + "/"]}
    mapping = []
    for item in plan["files"]:
        for target in item["destinations"]:
            if target["recording_id"] == key:
                relative = target["relative_path"]
                files[relative.split("/", 1)[0]].append(relative)
                mapping.append(
                    {
                        "source": item["source"],
                        "relative_path": relative,
                        "role": item["role"],
                    }
                )
    return {
        **parent["producer_context"],
        **parent["identity"],
        **parent["context"],
        "recording_name": parent["recording_name"],
        "source_dir": plan["source_dir"],
        "source_layout": "rolling_clips",
        "context_source": parent["context_source"],
        "orange_session_id": plan["acquisition_session_id"],
        "orange_producer": plan["orange_producer"],
        **(
            {"h5_relative_path": parent["h5_relative_path"]}
            if parent["h5_relative_path"] is not None
            else {}
        ),
        "files": files,
        "recording_geometry_bundle": parent["recording_geometry_bundle"],
        "preflight": default_preflight_payload(),
        "source_transfer": {
            "schema_id": "palette.organized_recording_transfer.v1",
            "snapshot_id": plan["snapshot_id"],
            "transfer_attempt_id": plan["transfer_attempt_id"],
            "organization_plan_sha256": plan["plan_sha256"],
            "snapshot_path": f"raw/acquisition/{SNAPSHOT_PATH}",
            "marker_path": f"raw/acquisition/{MARKER_NAME}",
            "source_to_parent_files": mapping,
        },
        "rolling_clip_streams": {
            "schema_id": "palette.orange_rolling_clip_streams.v1",
            "frame_clock": "recording_frame_id",
            "recording_clip_index": f"{INDEX_DIRECTORY}/recording_clip_index.json",
            "recording_frame_index": f"{INDEX_DIRECTORY}/recording_frame_index.parquet",
            "recording_frame_index_manifest": f"{INDEX_DIRECTORY}/recording_frame_index_manifest.json",
            "output_kinds": parent["output_kinds"],
        },
    }


def prepare_transfer_parent_recordings(
    plan: dict,
    *,
    batch_rows: int = 65536,
    registry_path: Path | None = None,
    require_stimulus: bool = False,
) -> dict:
    """Materialize and describe exact parents, leaving actual admission pending.

    Completed derived indexes are content-verified on retry. Incomplete index
    directories remain recoverable and are refused, not overwritten or deleted.
    """
    from fisheye.utils.build_transfer_parent_frame_index import (
        build_transfer_parent_frame_index,
    )

    require(
        plan.get("recording_layout") == "rolling_clips",
        "parent collection preparation requires rolling_clips",
    )
    materialize_transfer_organization(plan)
    with _organization_state(plan) as (state, save):
        contract = {
            "registry_path": (
                str(registry_path.resolve()) if registry_path is not None else None
            ),
            "require_stimulus": require_stimulus,
        }
        require(
            state.get("admission_contract") in (None, contract),
            "preparation cannot change its requested admission contract",
        )
        state["admission_contract"] = contract
        save()
        for parent in plan["parents"]:
            directory = Path(parent["destination_dir"])
            index = directory / INDEX_DIRECTORY
            if not index.exists():
                build_transfer_parent_frame_index(
                    Path(plan["source_dir"]),
                    camera_id=parent["identity"]["camera_id"],
                    output_dir=index,
                    batch_rows=batch_rows,
                    organization_plan=plan,
                )
            _verify_parent_index(plan, parent)
            manifest = _parent_manifest(plan, parent)
            path = directory / "recording_manifest.json"
            if path.exists():
                require(
                    strict_json(path) == manifest,
                    "existing parent manifest differs from organization plan",
                )
            else:
                _require_directory(
                    directory,
                    state["parent_directory_identities"][
                        parent["identity"]["recording_id"]
                    ],
                )
                write_json_atomic(path, manifest, overwrite=False)
        _verify_materialized_files(plan, state)
        state["parent_manifests_prepared"] = True
        save()
        return state


def _verify_parent_imports(
    plan: dict, *, registry_path: Path | None, require_stimulus: bool
) -> dict:
    """Use actual receipt/admission owners; organization state is never proof."""
    from fisheye.registry.recording_identity_authority import (
        load_verified_recording_import_receipt,
    )
    from fisheye.utils.import_recording_analysis import stimulus_runs_present

    receipts = {}
    for parent in plan["parents"]:
        directory = Path(parent["destination_dir"])
        require(
            strict_json(directory / "recording_manifest.json")
            == _parent_manifest(plan, parent),
            "parent manifest changed before retirement",
        )
        _verify_parent_index(plan, parent)
        zarr_path = directory / "zarr" / f"{directory.name}_analysis.zarr"
        receipt = load_verified_recording_import_receipt(zarr_path)
        require(
            receipt.identity_claim.identity.manifest_fields() == parent["identity"],
            "parent receipt has another source identity",
        )
        if require_stimulus:
            require(
                stimulus_runs_present(zarr_path),
                "requested stimulus import is incomplete",
            )
        if registry_path is not None:
            from fisheye.registry.recording_identity_authority import (
                load_verified_registry_recording_import,
            )

            verified = load_verified_registry_recording_import(
                registry_path=registry_path, zarr_path=zarr_path
            )
            require(
                verified.receipt == receipt,
                "registry admission differs from parent receipt",
            )
        receipts[parent["identity"]["recording_id"]] = receipt.receipt_sha256
    if registry_path is not None:
        from fisheye.registry.shadow_publish import validate_registry_sqlite

        validate_registry_sqlite(registry_path)
    return receipts


def _retire_source_file(source: Path, expected: dict, signature: list[int]) -> None:
    _matches_file(source.parent, source.name, expected)
    descriptor = os.open(source.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        current = os.stat(source.name, dir_fd=descriptor, follow_symlinks=False)
        require(
            [current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns]
            == signature,
            "source artifact ownership lost before retirement",
        )
        os.unlink(source.name, dir_fd=descriptor)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _source_members(source: Path) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    files, directories = {}, {}
    for directory, names, filenames in os.walk(source, followlinks=False):
        current = Path(directory)
        for name in names:
            path = current / name
            directories[path.relative_to(source).as_posix()] = _directory_identity(path)
        for name in filenames:
            path = current / name
            info = path.lstat()
            require(
                stat.S_ISREG(info.st_mode), "staging contains a non-regular artifact"
            )
            files[path.relative_to(source).as_posix()] = [
                info.st_dev,
                info.st_ino,
                info.st_size,
                info.st_mtime_ns,
            ]
    return files, directories


def finalize_transfer_staging(
    plan: dict, *, registry_path: Path | None = None, require_stimulus: bool = False
) -> dict:
    """Retire only inventoried staged files after every requested admission gate.

    Original producer custody is untouched. The exact external journal supports
    retry even after the snapshot/marker has been retired; missing files are
    tolerated only after verified retirement began. Every durable copy and
    actual parent receipt is checked again on retry, including completed runs.
    Empty source root is retained; no recursive removal is used.
    """
    _validate_plan(plan, live_source=False)
    registry_path = registry_path.resolve() if registry_path is not None else None
    contract = {
        "registry_path": str(registry_path) if registry_path is not None else None,
        "require_stimulus": require_stimulus,
    }
    with _organization_state(plan) as (state, save):
        source = Path(plan["source_dir"])
        _require_directory(source, state["source_directory_identity"])
        prior_contract = state.get("admission_contract")
        require(
            prior_contract is None or prior_contract == contract,
            "retirement cannot change its requested admission contract",
        )
        state["admission_contract"] = contract
        save()
        _verify_materialized_files(plan, state)
        receipts = _verify_parent_imports(
            plan, registry_path=registry_path, require_stimulus=require_stimulus
        )
        if state["status"] == "complete":
            require(
                not list(source.iterdir()),
                "completed staging source is no longer empty",
            )
            require(
                receipts == state["import_receipts"],
                "completed parent receipts changed",
            )
            return state
        require(
            state["status"] in {"materialized", "retiring"},
            "parent materialization is incomplete",
        )
        _flush_parent_publications(plan, state)
        _verify_materialized_files(plan, state)
        if state["status"] == "materialized":
            _validate_plan(plan, live_source=True)
            files, directories = _source_members(source)
            require(
                set(files) == {item["source"]["path"] for item in plan["files"]},
                "source inventory changed before retirement",
            )
            state.update(
                status="retiring",
                import_receipts=receipts,
                import_complete=True,
                retirement_files=files,
                retirement_directories=directories,
                retired_files=[],
            )
            save()
        require(
            receipts == state["import_receipts"], "retirement parent receipts changed"
        )
        current_files, current_directories = _source_members(source)
        pending = set(state["retirement_files"]) - set(state["retired_files"])
        require(
            set(current_files) <= pending,
            "staging contains unplanned or reappeared source artifacts",
        )
        require(
            set(current_directories) <= set(state["retirement_directories"]),
            "staging contains new unplanned directories",
        )
        by_relative = {item["source"]["path"]: item for item in plan["files"]}
        for relative in sorted(pending):
            item = by_relative[relative]
            _require_directory(source, state["source_directory_identity"])
            for target in item["destinations"]:
                parent = Path(plan["destination_root"]) / target["recording_id"]
                _require_directory(
                    parent, state["parent_directory_identities"][target["recording_id"]]
                )
                _matches_file(parent, target["relative_path"], item["source"])
            if relative in current_files:
                _retire_source_file(
                    source / relative,
                    item["source"],
                    state["retirement_files"][relative],
                )
            # Missing pending member covers unlink-success/journal-write-failure.
            # It is acknowledged only after all declared copies were rechecked.
            state["retired_files"].append(relative)
            save()
        for relative in sorted(
            state["retirement_directories"],
            key=lambda p: (p.count("/"), p),
            reverse=True,
        ):
            directory = source / relative
            if directory.exists() or directory.is_symlink():
                _require_directory(directory, state["retirement_directories"][relative])
                directory.rmdir()  # Exact recorded empty directory, never recursive.
                _fsync_directory(directory.parent)
        _require_directory(source, state["source_directory_identity"])
        require(not list(source.iterdir()), "staging did not become empty")
        _verify_materialized_files(plan, state)
        require(
            _verify_parent_imports(
                plan, registry_path=registry_path, require_stimulus=require_stimulus
            )
            == receipts,
            "parent admission changed during retirement",
        )
        state.update(status="complete", staging_finalized=True)
        save()
        return state
