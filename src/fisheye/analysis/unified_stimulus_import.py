"""Public importer's unified-profile branch: admit, then seal a reference.

The H5 is not copied. Admission streams it once; the run stores a sealed
reference (see ``shared.unified_h5.reference``). Selectors are never activated.
"""

from collections.abc import MutableMapping
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import zarr

from fisheye.shared.run_provenance import (
    build_writer_run_provenance,
    validate_run_provenance,
)
from fisheye.shared.unified_h5 import PROFILE, validate_unified_h5_artifact
from fisheye.shared.unified_h5.common import (
    MAX_JSON_BYTES,
    admission_scan,
    parse_json,
    require,
)
from fisheye.shared.unified_h5.integrity import source_identity
from fisheye.shared.unified_h5.reference import (
    REFERENCE_DIGEST_ATTR,
    REFERENCE_SCHEMA,
    REFERENCE_VERSION,
    new_native_run_name,
    open_unified_source,
    seal_reference,
)
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
            "unified_run_ownership_lost",
        )

    def __delitem__(self, name):
        require(
            self._owner._delete_owned_stimulus_attr(
                self._root,
                run_path=self._path,
                publication_owner_uuid=self._token,
                name=name,
            ),
            "unified_run_ownership_lost",
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
    with admission_scan() as scan:
        admission = validate_unified_h5_artifact(
            h5, source_h5=source_h5, finalization_receipt=receipt
        )
    if run_name is None:
        run_name = new_native_run_name()
    require(
        isinstance(run_name, str)
        and run_name not in ("", ".", "..")
        and not any(value in run_name for value in ("/", "\\", "\0", "\n", "\r")),
        "unified_run_name_invalid",
    )
    provenance_result = validate_run_provenance(
        build_writer_run_provenance(
            command="fisheye.analysis.import_stimulus_to_zarr",
            params={
                "source_profile": PROFILE,
                "reference_schema": REFERENCE_SCHEMA,
                "reference_version": REFERENCE_VERSION,
                "repair_chaser_gaps": False,
                "selector_eligible": False,
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
    require(provenance_result.valid, "unified_writer_provenance_invalid")
    provenance = provenance_result.normalized

    zarr_path = Path(zarr_path).expanduser().resolve()
    # No destination directory or group is created until admission succeeded.
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    require(root.metadata.zarr_format == 3, "unified_run_requires_zarr_v3")
    parent = require_runs_parent(root.require_group("analysis"), "stimulus_runs")
    before = _selectors(parent)
    require(
        run_name not in parent and run_name not in before.values(),
        "unified_run_exists_or_reserved_use_new_name",
    )
    token, path = str(uuid4()), f"analysis/stimulus_runs/{run_name}"

    def fresh():
        candidate = owner._fresh_owned_stimulus_candidate(
            root, run_path=path, publication_owner_uuid=token
        )
        require(
            candidate is not None
            and candidate.attrs.get("stage_selector_eligible") is False,
            "unified_run_ownership_lost",
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
            unified_reference_schema=REFERENCE_SCHEMA,
            unified_reference_version=REFERENCE_VERSION,
        )
        fresh()
        attrs[REFERENCE_DIGEST_ATTR] = seal_reference(
            run, h5, admission=admission, scan=scan,
            source_h5=source_h5, zarr_path=zarr_path,
        )
        # Nothing was copied, so no second admission pass: an unchanged
        # size/mtime/inode on the open handle is what the reference relies on.
        require(
            source_identity(h5, Path(source_h5)) == admission.source_identity,
            "unified_source_changed_during_admission",
        )
        mark_run_complete(
            lifecycle, parent_group=parent, run_name=run_name, run_provenance=provenance
        )
        fresh()
        owner.consolidate_metadata_capture_expected_warnings(zarr_path)
        fresh()
        published = zarr.open_group(str(zarr_path), mode="r", use_consolidated=True)
        require(
            _selectors(published["analysis/stimulus_runs"]) == before
            and open_unified_source(published, run_name=run_name).admission
            == admission.manifest_claims(),
            "unified_published_selection_or_admission_changed",
        )
        fresh()
    return run_name
