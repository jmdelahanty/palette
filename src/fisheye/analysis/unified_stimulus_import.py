"""Public importer's native-profile branch; never activates selectors."""

from collections.abc import MutableMapping
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import zarr

from fisheye.shared.run_provenance import (
    build_writer_run_provenance,
    validate_run_provenance,
)
from fisheye.shared.unified_h5 import PROFILE, validate_unified_h5_artifact
from fisheye.shared.unified_h5.common import MAX_JSON_BYTES, parse_json, require
from fisheye.shared.unified_h5.integrity import source_file_digest, source_identity
from fisheye.shared.unified_h5.storage import (
    MANIFEST_DIGEST_ATTR,
    inspect_native_inventory,
    load_unified_stimulus_candidate,
    verify_unpublished_native_candidate,
    write_native_candidate,
)
from fisheye.shared.unified_h5.storage_schema import STORAGE_SCHEMA, STORAGE_VERSION
from fisheye.shared.zarr_run_completion import (
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
)

SELECTOR_ATTRS = (
    "latest",
    "latest_complete",
    "latest_pending",
    "authoritative_run",
    "authoritative_run_provenance",
)


def _selectors(parent):
    return {key: parent.attrs[key] for key in SELECTOR_ATTRS if key in parent.attrs}


class _OwnedAttrs(MutableMapping):
    """Route each lifecycle attr through the existing exact-owner guard."""

    def __init__(self, owner, root, path, token, fresh):
        self._owner, self._root, self._path, self._token, self._fresh = (
            owner,
            root,
            path,
            token,
            fresh,
        )

    def __getitem__(self, name):
        return self._fresh().attrs[name]

    def __iter__(self):
        return iter(dict(self._fresh().attrs))

    def __len__(self):
        return len(self._fresh().attrs)

    def __setitem__(self, name, value):
        require(
            self._owner._persist_owned_stimulus_attr(
                self._root,
                run_path=self._path,
                publication_owner_uuid=self._token,
                name=name,
                value=value,
            ),
            "native_candidate_ownership_lost",
        )

    def __delitem__(self, name):
        require(
            self._owner._delete_owned_stimulus_attr(
                self._root,
                run_path=self._path,
                publication_owner_uuid=self._token,
                name=name,
            ),
            "native_candidate_ownership_lost",
        )


def import_unified_from_open_h5(
    h5, *, source_h5, zarr_path, run_name, overwrite, finalization_receipt
):
    # The established importer owns the stimulus publication interface. Import
    # its guard here after public dispatch, avoiding a second failure grammar.
    from fisheye.analysis import import_stimulus_to_zarr as owner

    require(
        finalization_receipt is not None,
        "unified_external_finalization_receipt_required",
    )
    receipt_path = Path(finalization_receipt).expanduser().resolve(strict=True)
    with receipt_path.open("rb") as stream:
        receipt = parse_json(
            stream.read(MAX_JSON_BYTES + 1), label="external_finalization_receipt"
        )
    admission = validate_unified_h5_artifact(
        h5, source_h5=source_h5, finalization_receipt=receipt
    )
    inventory = inspect_native_inventory(h5, admission.node_kinds)
    require(
        source_identity(h5, Path(source_h5)) == admission.source_identity,
        "source_h5_generation_changed",
    )
    if run_name is None:
        run_name = (
            "unified_native_"
            + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_")
            + uuid4().hex[:12]
        )
    require(
        isinstance(run_name, str)
        and run_name not in ("", ".", "..")
        and not any(value in run_name for value in ("/", "\\", "\0", "\n", "\r")),
        "native_run_name_invalid",
    )
    provenance_result = validate_run_provenance(
        build_writer_run_provenance(
            command="fisheye.analysis.import_stimulus_to_zarr",
            params={
                "source_profile": PROFILE,
                "storage_schema": STORAGE_SCHEMA,
                "storage_version": STORAGE_VERSION,
                "repair_chaser_gaps": False,
                "selector_eligible": False,
                "write_ownership": "single_importer_all_physical_chunks",
                "source_sha256": admission.source_sha256,
            },
            input_artifacts=[
                {
                    "path": admission.source_identity["path"],
                    "sha256": admission.source_sha256,
                    "size_bytes": admission.source_identity["size_bytes"],
                }
            ],
        )
    )
    require(provenance_result.valid, "native_writer_provenance_invalid")
    provenance = provenance_result.normalized

    zarr_path = Path(zarr_path).expanduser().resolve()
    # No destination directory or group is created until every source check and
    # metadata-budget check above has succeeded.
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    require(root.metadata.zarr_format == 3, "native_candidate_requires_zarr_v3")
    parent = require_runs_parent(root.require_group("analysis"), "stimulus_runs")
    before = _selectors(parent)
    require(
        run_name not in parent and run_name not in before.values(),
        "native_run_exists_or_reserved_use_new_name",
    )
    token, path = str(uuid4()), f"analysis/stimulus_runs/{run_name}"

    def fresh():
        candidate = owner._fresh_owned_stimulus_candidate(
            root, run_path=path, publication_owner_uuid=token
        )
        require(
            candidate is not None
            and candidate.attrs.get("stage_selector_eligible") is False,
            "native_candidate_ownership_lost",
        )
        return candidate

    with owner._staged_run_failure_guard(
        root, runs_parent=parent, run_name=run_name, publication_owner_uuid=token
    ) as run:
        attrs = _OwnedAttrs(owner, root, path, token, fresh)
        lifecycle = SimpleNamespace(attrs=attrs)
        mark_run_started(lifecycle, run_name=run_name, stage="stimulus")
        attrs.update(
            source_profile=PROFILE,
            native_storage_schema=STORAGE_SCHEMA,
            native_storage_version=STORAGE_VERSION,
        )
        manifest_digest = write_native_candidate(
            h5,
            run,
            admission=admission,
            inventory=inventory,
            run_name=run_name,
            owner=token,
            provenance=provenance,
            assert_owner=fresh,
        )
        repeated = validate_unified_h5_artifact(
            h5, source_h5=source_h5, finalization_receipt=receipt
        )
        require(
            repeated.source_identity == admission.source_identity
            and repeated.manifest_claims() == admission.manifest_claims()
            and inspect_native_inventory(h5, repeated.node_kinds) == inventory,
            "native_source_changed_during_copy",
        )
        attrs[MANIFEST_DIGEST_ATTR] = manifest_digest
        require(
            _selectors(root["analysis/stimulus_runs"]) == before,
            "native_import_parent_selection_changed",
        )
        # With explicit ineligibility and a non-reserved run name this executes
        # normal provenance finalization without publishing any selector.
        mark_run_complete(
            lifecycle, parent_group=parent, run_name=run_name, run_provenance=provenance
        )
        fresh()
        verify_unpublished_native_candidate(root, run_name=run_name)
        require(
            _selectors(root["analysis/stimulus_runs"]) == before,
            "native_import_parent_selection_changed",
        )
        fresh()
        owner.consolidate_metadata_capture_expected_warnings(zarr_path)
        fresh()
        published = zarr.open_group(str(zarr_path), mode="r", use_consolidated=True)
        require(
            _selectors(published["analysis/stimulus_runs"]) == before
            and _selectors(root["analysis/stimulus_runs"]) == before,
            "native_consolidated_selection_changed",
        )
        loaded = load_unified_stimulus_candidate(published, run_name=run_name)
        require(
            loaded.admission == admission.manifest_claims(),
            "native_published_admission_changed",
        )
        require(
            source_file_digest(h5, admission.source_identity)
            == admission.source_sha256,
            "native_source_changed_before_return",
        )
        fresh()
    return run_name
