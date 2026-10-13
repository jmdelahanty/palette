"""Validate the exact pinned Citrus tables; do not infer them from emitted rows."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
from importlib.resources import files

import h5py
import numpy as np

from fisheye.shared.heading_convention import (
    LEGACY_CANVAS_ANGLE,
    STIMULUS_ANGLE_CONVENTION,
)

from .common import (
    canonical_json,
    current_admission_scan,
    internal_path,
    parse_json,
    require,
    same_json,
    text,
)
from .hdf5_types import check_dataset_budget, dataset_bytes, iter_blocks, iter_payload

APPEARANCE = "/components/visual_appearance/states"
APPEARANCE_MANIFEST = "/components/visual_appearance/replay_dependency_manifest_json"
CONTRACT_HASHES = {
    "experimental_h5_core_v1.json": "febb7516ff96f359b6784bf137682abe387b7922270e9637ebc58b369e07f9c1",
    "experimental_h5_geometry_v1.json": "4c0c707192cf1fa6ceafb2f632672972f41dc7e4c7cc5cdb0c0e3b7920a1474b",
    "experimental_h5_identity_claims_v1.json": "83ca66b336f3e3a40ebaafc869d7febd020509e238243381dfe871cbeb89507e",
    "object_appearance_replay_dependency_manifest_v1.json": "eccfcfcb3c6e66549b2fdfcfcf88fdc4fa9def3d6ce4a375494d011b04f2607c",
    "unified_h5_admission_v2.json": "ad0aab702e6714131db857afe8d96323999df184e0a3b357705c1acf56a90c38",
    "unified_h5_producer_policy_d544b081.json": "d544b0814006360b11b02cd9d171eefbddae7cc9cc41ad57470bbd240bc83cb9",
    "experimental_h5_core_chaser_v2.json": "cb68ab6ecea614030b8f2a3d2096e9122b1efa5156b56ed7a495b3072c70316a",
    "experimental_h5_correspondence_tables_v2.json": "cb3ac574c76750c23de730129c4a32371328ffb34d792b1cfab164b7201ba1ea",
    "experimental_h5_correspondence_input_v2.schema.json": "a33e06819e966b7273015bfffe4aa0d01f24a04fb9ffac69abb29ed5c76329a3",
    "experimental_h5_correspondence_receipt_v2.schema.json": "8d2a414f46f6d6263f2609c55e15add9d7a2b61ad98349cf1ac3aef598b8d4e8",
    "experimental_h5_capacity_preflight_v1.schema.json": "d5d7ebf5249de35ce7234ddfec3c7cb51b57ac3b6fdcb25bf75724f25223d068",
    # Admission revision 3 (bounding_boxes v2, agent-contracts PR 73).
    "unified_h5_admission_v3.json": "c32282b68dc5d0057abe3b4144df27a19afbc69f4a920a64aabd09db6b8647b0",
    "unified_h5_producer_policy_b2af7764.json": "b2af7764c7f27f6295007ba3049dcf8a35d0f84c5801f8a1ff56150c7f9e7fc1",
    "experimental_h5_core_bounding_boxes_v2.json": "55f94f83e6f43824488161b7238846b2233933be871ac8044bef04d3fa6df4f8",
    # Citrus 4d9d04d docs/schemas/experimental_h5_capacity_preflight_v1.schema.json,
    # renamed here because its revision-2 predecessor keeps the original name.
    "experimental_h5_capacity_preflight_v1_ba8333a3.schema.json": "ba8333a35526dc1fc655734d938b14a816fd143d2c5eb395ad17065f79718a54",
    # Admission revision 4 (frame identity v1; agent-contracts PR 88, Citrus PR 24).
    "unified_h5_admission_v4.json": "e702df25e5164548c72fa94a683922836567245ed7b7880316ea8ead2d33a0a4",
    "unified_h5_producer_policy_3c9866ed.json": "3c9866eda2f78c4aa37d1b68a9f4b6245aab8415b3ce8e3412cb3cbf5ab7eb78",
    "experimental_h5_core_frame_identity_v1.json": "4365821df9cdde35d0417fec5b656de12607c211e24232b408e87344883a8264",
    "experimental_h5_capacity_preflight_v2.schema.json": "3f78bff9fd51207d035ed9a2feb32660f5232b91d3674389882a4914fbf6bbc3",
    "experimental_h5_correspondence_receipt_v3.schema.json": "7143f9cff038b4dda1c184e5f0ca7b1b924b541074d24b25b9b0d67dab9d34b4",
    # Admission revision 5 (angle_v1: stimulus angle convention v1; agent-contracts
    # PRs 91/92, Citrus PR 27 at c9d52be). Merged unchanged: 7a2e5d2, f3d97d4, 576481f.
    "unified_h5_admission_v5.json": "7b1448f83b68df898e1022388581ae833151c8a83d375bb51abb82c13c99ce71",
    "unified_h5_producer_policy_9af921ca.json": "9af921cab2aedae8af66a00600f5eee6b076c5d8218b6f0e928f93b4df3fbedd",
    "experimental_h5_core_angle_v1.json": "0c054c7d5d3c1a4f7b39f800b63e272bb43bac30b5b640993a94974e2be24f07",
    "experimental_h5_capacity_preflight_v3.schema.json": "00b201da455c8f1fc021506ceaf291092f0d61b1d2d5a2689f3ba386be5c60ea",
}
# The admission revision a file is judged under is chosen by its capacity
# preflight's schema_version: preflight v1 files (no catalog_id) keep revision
# 3 and its (chaser_state, bounding_boxes) pair key; preflight v2 files carry
# catalog_id and are selected by revision 4's full table-version map; preflight
# v3 files (catalog_id angle_v1) are selected the same way by revision 5. Any
# other preflight version is refused. A later catalog needs only its pins here.
ADMISSIONS_BY_PREFLIGHT_VERSION = {
    1: "unified_h5_admission_v3.json",
    2: "unified_h5_admission_v4.json",
    3: "unified_h5_admission_v5.json",
}
ADMISSION = ADMISSIONS_BY_PREFLIGHT_VERSION[1]
# Every pinned core catalog. The original v1 catalog stays pinned byte-for-byte
# for provenance and for the legacy adapter, never for admission.
CORE_CATALOG_FILES = (
    "experimental_h5_core_v1.json",
    "experimental_h5_core_chaser_v2.json",
    "experimental_h5_core_bounding_boxes_v2.json",
    "experimental_h5_core_frame_identity_v1.json",
    "experimental_h5_core_angle_v1.json",
)
CORE_CATALOGS = {
    1: "experimental_h5_core_v1.json",
    2: "experimental_h5_core_chaser_v2.json",
}
ADMITTED_CHASER_VERSION = 2
# Table-independent catalog sections (component outcomes, attributes) are
# identical across the pinned catalogs; tests assert it.
CORE_CATALOG = CORE_CATALOGS[ADMITTED_CHASER_VERSION]
CHASER_STATES = "/components/chaser/states"
BOUNDING_BOXES = "/observations/bounding_boxes"
# Admission revision 3 names each admitted catalog's pair of table versions.
_PIN_TABLES = {"chaser_state": CHASER_STATES, "bounding_boxes": BOUNDING_BOXES}
# Stimulus angle convention (agent-contracts citrus-palette/angles v1): fixed by
# the admitted catalog binding, never by the file's date or by the session
# attribute alone. Catalog pins of revisions 3 and 4 carry no
# ``stimulus_angle_convention`` field; every catalog they admit is legacy.
SESSION = "/metadata/session"
SESSION_ANGLE_ATTRIBUTE = "stimulus_angle_convention"
PROTOCOL_ANGLE_DECLARATIONS = (
    "/protocol/authored/protocol_definition_json",
    "/protocol/authored/protocol_trial_index_json",
    "/protocol/authored/protocol_semantic_json",
)
CORRESPONDENCE_RECEIPT_SCHEMAS = {
    2: "experimental_h5_correspondence_receipt_v2.schema.json",
    3: "experimental_h5_correspondence_receipt_v3.schema.json",
}


def _template(path: str) -> str | None:
    if path.startswith("/definitions/enums/") and len(path.split("/")) == 4:
        return "/definitions/enums/{component}"
    if (
        path.startswith("/correspondence/")
        and path.endswith("/sources")
        and path != "/correspondence/frames/sources"
    ):
        return "/correspondence/{component}/sources" if len(path.split("/")) == 4 else None
    return path


def table_template(path: str) -> str:
    """The catalog path a concrete core table path is defined under."""
    template = _template(path)
    require(template is not None, f"unknown_table:{path}")
    return template


def catalog_table_versions(catalog_file: str) -> dict[str, int]:
    """A pinned catalog's {catalog table path -> schema_version} map."""
    return {t["path"]: t["schema_version"] for t in _contract(catalog_file)["tables"]}


@lru_cache(maxsize=1)
def _table_definitions() -> dict:
    """(catalog path, schema_version) -> its one definition over every pinned catalog.

    A version names one definition: two catalogs that define the same
    (path, version) differently are a pin error, refused here.
    """
    definitions = {}
    for name in CORE_CATALOG_FILES:
        for table in _contract(name)["tables"]:
            key = (table["path"], table["schema_version"])
            require(
                definitions.setdefault(key, table) == table,
                f"conflicting_table_definition:{key[0]}:{key[1]}",
            )
    return definitions


def _admitted_catalogs(admission: dict) -> dict[str, dict]:
    pins = admission["core_definition_pins"]
    return {
        name: pin
        for name, pin in pins.items()
        if isinstance(pin, dict) and pin.get("admitted_by_this_revision")
    }


def _catalog_file(pin: dict) -> str:
    return pin["catalog_path"].rsplit("/", 1)[-1]


@lru_cache(maxsize=1)
def admitted_table_versions() -> dict[str, frozenset[int]]:
    """Catalog path -> the versions some admitted catalog of some revision pins."""
    versions: dict[str, set[int]] = {}
    for admission_name in ADMISSIONS_BY_PREFLIGHT_VERSION.values():
        for pin in _admitted_catalogs(_contract(admission_name)).values():
            for path, version in catalog_table_versions(_catalog_file(pin)).items():
                versions.setdefault(path, set()).add(version)
    return {path: frozenset(values) for path, values in versions.items()}


def preflight_admission(document: dict) -> str:
    """The admission revision a capacity preflight's schema_version selects."""
    version = document.get("schema_version")
    require(
        type(version) is int and version in ADMISSIONS_BY_PREFLIGHT_VERSION,
        "capacity_preflight_unknown_schema_version",
    )
    return ADMISSIONS_BY_PREFLIGHT_VERSION[version]


def _present_declared_versions(h5, declared_paths) -> dict[str, int]:
    """{declared, present core table path -> its schema_version}."""
    known = admitted_table_versions()
    observed = {}
    for path in declared_paths:
        if path in h5 and _template(path) in known:
            observed[path] = table_version(h5, path)
    return observed


def _map_matches(observed: dict[str, int], catalog_versions: dict[str, int]) -> bool:
    return all(
        catalog_versions.get(table_template(path)) == version
        for path, version in observed.items()
    )


def admitted_binding(h5, declared_policy_sha256: str, preflight: dict | None = None) -> dict:
    """The one admitted catalog binding for this file.

    The capacity preflight's schema_version selects the admission revision.
    Revisions 4 and 5 (preflight v2, v3) select the same way. The result's
    ``stimulus_angle_convention`` is the selected catalog pin's convention
    (``canvas_y_down_clockwise_legacy`` for every catalog of revisions 3-4
    and for the legacy pins of revision 5); readers of stimulus angles use it.
    Revision 3 (preflight v1): present (chaser_state, bounding_boxes) versions
    and the declared producer policy select one pair-keyed catalog. Revision 4
    (preflight v2): the declared, present core tables' {path -> schema_version}
    map must equal an admitted catalog's map, the declared catalog_id must name
    that same catalog (otherwise ``catalog_id_mismatch``), and the declared
    policy must be the one bound to it. Absent and undeclared tables do not
    constrain. Exactly one binding must remain, or the file is refused.
    """
    if preflight is None:
        preflight = {"schema_version": 1}
    admission_name = preflight_admission(preflight)
    admission = contract(admission_name)
    bindings = admission["admitted_catalog_bindings"]
    catalogs = _admitted_catalogs(admission)
    declared = "sha256:" + declared_policy_sha256
    if admission["core_definition_pins"]["selection"]["key"] == "table_version_map":
        declared_paths = [table.get("path") for table in preflight.get("tables", [])]
        observed = _present_declared_versions(
            h5, [path for path in declared_paths if isinstance(path, str)]
        )
        matches = [
            name
            for name, pin in catalogs.items()
            if _map_matches(observed, catalog_table_versions(_catalog_file(pin)))
        ]
        catalog_id = preflight.get("catalog_id")
        chosen = [name for name in matches if catalogs[name].get("catalog_id") == catalog_id]
        require(
            catalog_id is not None and len(chosen) == 1, "catalog_id_mismatch"
        )
        require(
            bindings[chosen[0]]["producer_policy"]["file_sha256"] == declared
            and bindings[chosen[0]].get("catalog_id") == catalog_id,
            "capacity_preflight_producer_policy",
        )
        matches = chosen
    else:
        matches = []
        for name, pin in catalogs.items():
            tables_agree = all(
                path not in h5 or table_version(h5, path) == pin["table_versions"][key]
                for key, path in _PIN_TABLES.items()
            )
            if tables_agree and bindings[name]["producer_policy"]["file_sha256"] == declared:
                matches.append(name)
        require(len(matches) == 1, "capacity_preflight_producer_policy")
    binding = bindings[matches[0]]
    schema_sha = binding["capacity_preflight_schema"]["file_sha256"].removeprefix("sha256:")
    (schema_name,) = [name for name, digest in CONTRACT_HASHES.items() if digest == schema_sha]
    receipt_pin = admission["correspondence_definition_pins"]["receipt_schema"]
    angle = catalogs[matches[0]].get("stimulus_angle_convention", LEGACY_CANVAS_ANGLE)
    require(angle in (LEGACY_CANVAS_ANGLE, STIMULUS_ANGLE_CONVENTION), "stimulus_angle_convention_unknown")
    return {
        "catalog": matches[0],
        "stimulus_angle_convention": angle,
        "catalog_file": _catalog_file(catalogs[matches[0]]),
        "preflight_schema": schema_name,
        "admission": admission_name,
        "correspondence_receipt_version": receipt_pin["schema_version"],
        **binding,
    }


def require_stimulus_angle_convention(h5, binding: dict) -> str:
    """The file's stimulus angle declarations agree with its binding's convention.

    A legacy binding (``canvas_y_down_clockwise_legacy``) never carries the
    ``/metadata/session`` attribute or a protocol declaration: either is
    refused as ``stimulus_angle_convention_unexpected``, and the sign is never
    taken from it. An ``angle_v1`` binding requires the session attribute and
    the authored protocol definition, trial index and semantic identity to
    declare exactly ``atan2_negative_y_x_degrees``; anything else is
    ``stimulus_angle_convention_mismatch``. Returns the binding's convention.
    """
    convention = binding["stimulus_angle_convention"]
    attrs = h5[SESSION].attrs if SESSION in h5 else {}
    session = attrs.get(SESSION_ANGLE_ATTRIBUTE) if SESSION_ANGLE_ATTRIBUTE in attrs else None
    if isinstance(session, bytes):
        session = session.decode("utf-8", errors="replace")
    protocol = []
    for path in PROTOCOL_ANGLE_DECLARATIONS:
        document = read_json(h5, path) if path in h5 else {}
        protocol.append(
            document.get(SESSION_ANGLE_ATTRIBUTE) if isinstance(document, dict) else None
        )
    if convention == LEGACY_CANVAS_ANGLE:
        require(
            SESSION_ANGLE_ATTRIBUTE not in attrs
            and all(value is None for value in protocol),
            "stimulus_angle_convention_unexpected",
        )
        return convention
    require(
        session == convention and all(value == convention for value in protocol),
        "stimulus_angle_convention_mismatch",
    )
    return convention


def require_selected_catalog_versions(h5, catalog_file: str) -> None:
    """Every present core table carries the selected catalog's version for it."""
    versions = catalog_table_versions(catalog_file)
    known = admitted_table_versions()

    def visit(name, node):
        path = "/" + name
        if not isinstance(node, h5py.Dataset):
            return
        template = _template(path)
        if template in versions and template in known:
            require(
                table_version(h5, path) == versions[template],
                f"table_version_not_in_selected_catalog:{path}",
            )

    h5.visititems(visit)


def require_declared_core_tables(h5, declared_paths, catalog_file: str) -> None:
    """The selected catalog's literal tables are present iff the preflight declares them.

    Templated catalog paths (``/correspondence/{component}/sources``, enums) and
    non-core tables follow their own contracts (admission revision 3,
    ``admitted_catalog_bindings.declared_tables``).
    """
    declared = set(declared_paths)
    for table in contract(catalog_file)["tables"]:
        path = table["path"]
        if "{" in path:
            continue
        require((path in declared) == (path in h5), f"declared_core_table_mismatch:{path}")
CORRESPONDENCE_CATALOG = "experimental_h5_correspondence_tables_v2.json"


@lru_cache(maxsize=4)
def _contract(name: str) -> dict:
    data = files(__package__).joinpath("contracts").joinpath(name).read_bytes()
    require(
        sha256(data).hexdigest() == CONTRACT_HASHES[name],
        f"packaged_contract_drift:{name}",
    )
    return parse_json(data, label=name)


def contract(name: str) -> dict:
    return deepcopy(_contract(name))


def read_json(h5, path: str, *, canonical: bool = False) -> dict:
    require(
        path in h5 and isinstance(h5[path], h5py.Dataset),
        f"missing_json_dataset:{path}",
    )
    return parse_json(dataset_bytes(h5[path]), label=path, canonical=canonical)


def correspondence_table_paths() -> frozenset[str]:
    return frozenset(t["path"] for t in _contract(CORRESPONDENCE_CATALOG)["tables"])


def validate_catalog_table(h5, path: str) -> dict:
    """Layout, attributes and value rules of a correspondence-catalog table.

    These tables are digested as closed logical values, so the catalog is
    checked here, separately, and never folded into their digests.
    """
    matches = [
        table
        for table in _contract(CORRESPONDENCE_CATALOG)["tables"]
        if table["path"] == path
    ]
    require(len(matches) == 1, f"unknown_correspondence_table:{path}")
    schema = deepcopy(matches[0])
    require(path in h5 and isinstance(h5[path], h5py.Dataset), f"missing_table:{path}")
    dataset = h5[path]
    check_dataset_budget(dataset)
    require(dataset.ndim == schema["rank"], f"table_rank:{path}")
    _validate_fields(dataset.id.get_type(), schema["fields"], schema["itemsize"], path)
    required = schema["required_attributes"]
    require(set(dataset.attrs) == set(required), f"table_attributes:{path}")
    for name, spec in required.items():
        value = dataset.attrs.get(name)
        if spec["dtype"] == "utf8":
            require(
                text(value, f"{path}@{name}") == spec["value"],
                f"table_attribute_mismatch:{path}:{name}",
            )
        else:
            attr = dataset.attrs.get_id(name)
            require(
                attr.shape == () and attr.dtype == np.dtype(spec["dtype"])
                and int(value) == spec["value"],
                f"table_attribute_mismatch:{path}:{name}",
            )
    for _, block in iter_blocks(dataset):
        _validate_values(block, schema["fields"], path)
    return schema


def table_version(h5, path: str) -> int | None:
    """A core table's declared schema_version, refused unless some admitted catalog pins it."""
    if path not in h5:
        return None
    versions = admitted_table_versions().get(_template(path))
    if versions is None:
        return None
    version = h5[path].attrs.get("schema_version")
    require(
        version is not None and int(version) in versions,
        f"unsupported_table_version:{path}",
    )
    return int(version)


def table_schema(path: str, version: int | None = None) -> dict:
    """The pinned definition of ``path`` at ``version`` (any pinned catalog).

    A (path, version) pair names one definition across every pinned catalog;
    without a version the revision 2 core catalog's definition is returned.
    """
    template = table_template(path)
    if version is None:
        matches = [
            table for table in _contract(CORE_CATALOG)["tables"] if table["path"] == template
        ]
    else:
        definition = _table_definitions().get((template, version))
        matches = [] if definition is None else [definition]
    require(len(matches) == 1, f"unknown_table:{path}")
    result = deepcopy(matches[0])
    result["path"] = path
    return result


def _validate_fields(datatype, fields, itemsize: int, path: str) -> None:
    require(
        datatype.get_class() == h5py.h5t.COMPOUND and datatype.get_size() == itemsize,
        f"table_itemsize_or_type:{path}",
    )
    require(datatype.get_nmembers() == len(fields), f"table_field_count:{path}")
    for index, field in enumerate(fields):
        name = field["name"]
        require(
            datatype.get_member_name(index).decode("utf-8") == name
            and datatype.get_member_offset(index) == field["offset"],
            f"table_field_layout:{path}:{name}",
        )
        member = datatype.get_member_type(index)
        if field["dtype"] == "compound":
            _validate_fields(
                member, field["fields"], field["itemsize"], f"{path}:{name}"
            )
        elif field["dtype"] == "utf8_nullterm":
            require(
                member.get_class() == h5py.h5t.STRING
                and not member.is_variable_str()
                and member.get_cset() == h5py.h5t.CSET_UTF8
                and member.get_strpad() == h5py.h5t.STR_NULLTERM
                and member.get_size() == field["count"],
                f"table_utf8_type:{path}:{name}",
            )
        else:
            expected = np.dtype(field["dtype"])
            if field["count"] != 1:
                require(
                    member.get_class() == h5py.h5t.ARRAY
                    and member.get_array_dims() == (field["count"],),
                    f"table_array_type:{path}:{name}",
                )
                member = member.get_super()
            require(
                member.equal(h5py.h5t.py_create(expected)),
                f"table_field_type:{path}:{name}",
            )


def _validate_values(values, fields, path: str) -> None:
    for field in fields:
        name = field["name"]
        if field["dtype"] == "compound":
            _validate_values(values[name], field["fields"], f"{path}:{name}")
        if field.get("boolean"):
            require(
                bool(np.all((values[name] == 0) | (values[name] == 1))),
                f"invalid_boolean:{path}:{name}",
            )
        validity = field.get("validity_field")
        if validity:
            require(
                bool(np.all(values[name][values[validity] == 0] == 0)),
                f"invalid_value_fill:{path}:{name}",
            )
        if field["dtype"] == "utf8_nullterm":
            for value in values[name].reshape(-1):
                raw = bytes(value)
                require(len(raw) < field["count"], f"unterminated_text:{path}:{name}")
                text(raw.split(b"\0", 1)[0], f"{path}:{name}", empty=True)


def appearance_dependencies(h5) -> dict:
    dataset = h5[APPEARANCE]
    require(
        dataset.attrs.get("replay_dependency_manifest_ref") == APPEARANCE_MANIFEST,
        "appearance_replay_reference_mismatch:manifest",
    )
    manifest = read_json(h5, APPEARANCE_MANIFEST, canonical=True)
    require(
        same_json(
            manifest, _contract("object_appearance_replay_dependency_manifest_v1.json")
        ),
        "appearance_replay_manifest_mismatch",
    )
    bindings = {
        "appearance_profile_index": "profile_definition_ref",
        "protocol_semantic_identity": "protocol_semantic_ref",
        "chaser_runtime_state": "stimulus_state_table_ref",
        "stimulus_frame_timeline": "frame_table_ref",
        "renderer_snapshot": "renderer_snapshot_ref",
        "presentation_timing": "presentation_timing_ref",
        "appearance_enum_definitions": "appearance_enum_group_ref",
        "build_identity": "build_identity_ref",
    }
    values = {}
    for dependency in manifest["adapters"][0]["dependencies"]:
        role, path = dependency["role"], internal_path(dependency["object_ref"])
        require(path in h5, f"appearance_replay_dependency_missing:{role}")
        if role in bindings:
            require(
                dataset.attrs.get(bindings[role]) == path,
                f"appearance_replay_reference_mismatch:{role}",
            )
        for name in dependency["required_attributes"]:
            value = text(h5[path].attrs.get(name), f"{path}@{name}")
            values.setdefault(role, {})[name] = value
    require(
        "/protocol/executed/execution_index_json" in h5,
        "appearance_execution_index_missing",
    )
    return values


def describe_table(h5, path: str) -> dict:
    scan = current_admission_scan()
    if scan is not None and path in scan.tables:
        return deepcopy(scan.tables[path])
    schema = table_schema(path, table_version(h5, path))
    require(path in h5 and isinstance(h5[path], h5py.Dataset), f"missing_table:{path}")
    dataset = h5[path]
    check_dataset_budget(dataset)
    require(dataset.ndim == schema["rank"], f"table_rank:{path}")
    _validate_fields(dataset.id.get_type(), schema["fields"], schema["itemsize"], path)
    for name, expected in (
        ("schema_id", schema["id"]),
        ("schema_version", schema["schema_version"]),
        ("key_fields", schema["key_fields"]),
    ):
        require(
            dataset.attrs.get(name) == expected,
            f"table_attribute_mismatch:{path}:{name}",
        )
    require(
        dataset.attrs.get_id("schema_version").shape == ()
        and dataset.attrs.get_id("schema_version").dtype == np.dtype("<u8"),
        f"table_version_width:{path}",
    )
    references = {}
    for name in schema["required_reference_attributes"]:
        reference = internal_path(dataset.attrs.get(name))
        require(reference in h5, f"table_reference_missing:{path}:{name}")
        references[name] = reference
    expected_refs = {
        "frame_table_ref": "/frames/stimulus",
        "frame_context_ref": "/frames/stimulus",
        "acquisition_binding_ref": "/correspondence/acquisition/binding_json",
        "protocol_snapshot_ref": "/protocol/authored",
        "enum_group_ref": "/definitions/enums",
        "clock_domain_ref": "/timing/clock_domains/citrus_process",
        "coordinate_context_ref": "/geometry/runtime/contract_json",
    }
    for name, expected in expected_refs.items():
        if name in references:
            require(
                references[name] == expected, f"table_reference_mismatch:{path}:{name}"
            )
    if "state_table_ref" in references:
        require(
            references["state_table_ref"] == f'/components/{path.split("/")[2]}/states',
            f"table_reference_mismatch:{path}:state_table_ref",
        )
    descriptor = {
        "schema_id": "citrus.experimental_h5.core_dataset_digest",
        "schema_version": 1,
        "encoding": "core_schema_json_and_packed_le_v1",
        "table": schema,
        "shape": list(dataset.shape),
        "references": references,
    }
    if path == APPEARANCE:
        descriptor["required_dependency_attribute_values"] = appearance_dependencies(h5)
    header = canonical_json(descriptor)
    hasher = sha256(
        b"citrus.experimental_h5.core_dataset_digest.v1\n"
        + len(header).to_bytes(8, "little")
        + header
    )
    for start, block in iter_blocks(dataset):
        _validate_values(block, schema["fields"], path)
        data = block.tobytes(order="C")
        hasher.update(data)
        if scan is not None:
            scan.record_block(dataset.name, start, data)
    descriptor["content_sha256"] = "sha256:" + hasher.hexdigest()
    if scan is not None:
        scan.tables[path] = deepcopy(descriptor)
    return descriptor


def describe_internal_dataset(dataset, component_id: str) -> dict:
    spec = check_dataset_budget(dataset)
    variable = spec["class"] == "string" and spec["variable_length"]
    if variable:
        logical_size = sum(
            len(value.encode("utf-8") if isinstance(value, str) else bytes(value))
            for _, block in iter_blocks(dataset)
            for value in block.reshape(-1)
        )
    else:
        logical_size = dataset.size * dataset.dtype.itemsize
    result = {
        "kind": "internal_dataset",
        "component_id": component_id,
        "path": dataset.name,
        "encoding": "hdf5_closed_logical_values_v1",
        "shape": list(dataset.shape),
        "type": spec,
        "logical_payload_size_bytes": logical_size,
    }
    header = canonical_json(result)
    hasher = sha256(
        b"citrus.experimental_h5.internal_dataset_digest.v1\n"
        + len(header).to_bytes(8, "little")
        + header
    )
    scan = current_admission_scan()
    if variable or scan is None:
        for block in iter_payload(dataset):
            hasher.update(block)
    else:
        for start, block in iter_blocks(dataset):
            data = np.asarray(block, dtype=dataset.dtype).tobytes(order="C")
            hasher.update(data)
            scan.record_block(dataset.name, start, data)
    result["content_sha256"] = "sha256:" + hasher.hexdigest()
    return result
