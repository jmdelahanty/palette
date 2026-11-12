"""Cross-check Citrus's recorded pancakebatter rig-file commit (Citrus #75).

Citrus ``SaveRigSnapshot`` (citrus ``dbf2ee9``) writes
``/metadata/rig/pancakebatter_rig_file``. Its ``source_commit`` attribute is
"the sidecar's ``commit``, recorded only when it is a full sha"
(``RigBinding::PinRigFile``; ``docs/Understanding_H5_Log.md``), and the
installer's sidecar bytes are stored in the ``sidecar`` dataset. The dataset is
covered by the sealed dependency manifest, which the inner integrity gate has
already checked; group attributes are not. So a non-empty ``source_commit``
must equal the sealed sidecar's own ``commit``, and ``status=pinned`` must
carry one.

This is an enforcement check only. It makes no physical-rig claim, adds no
admission claim, and leaves files without the group (or with an empty
``source_commit`` and a status other than ``pinned``) exactly as before.
"""

from __future__ import annotations

import re

import h5py
import yaml

from .common import require
from .metadata import native_attributes, string_attributes

GROUP = "/metadata/rig/pancakebatter_rig_file"
PINNED = "pinned"
FULL_COMMIT = re.compile(r"[0-9a-f]{40}\Z")

CONTRADICTS = "rig_provenance_source_commit_contradicts_sidecar"
UNVERIFIABLE = "rig_provenance_source_commit_unverifiable"
PINNED_WITHOUT = "rig_provenance_pinned_without_source_commit"
SIDECAR_NOT_STRING = "rig_provenance_sidecar_not_scalar_string"


def _sidecar_commit(raw: bytes) -> str | None:
    """The sidecar's top-level ``commit`` scalar text, as yaml-cpp reads it.

    ``BaseLoader`` keeps every scalar as its text, matching ``ScalarText``;
    a value that is not a scalar, a repeated key, or a sidecar that is not a
    single UTF-8 YAML map names no commit.
    """

    try:
        node = yaml.compose(raw.decode("utf-8"), Loader=yaml.BaseLoader)
    except (UnicodeDecodeError, yaml.YAMLError):
        return None
    if not isinstance(node, yaml.MappingNode):
        return None
    values = [
        value
        for key, value in node.value
        if isinstance(key, yaml.ScalarNode) and key.value == "commit"
    ]
    if len(values) != 1 or not isinstance(values[0], yaml.ScalarNode):
        return None
    return values[0].value


def validate_rig_provenance(h5) -> None:
    group = h5.get(GROUP)
    if group is None:
        return
    native = native_attributes(group)
    attrs = string_attributes(
        {name: native[name] for name in ("status", "source_commit") if name in native}
    )
    source_commit = attrs.get("source_commit", "")
    if not source_commit:
        require(attrs.get("status") != PINNED, PINNED_WITHOUT)
        return
    sidecar = group.get("sidecar")
    require(isinstance(sidecar, h5py.Dataset), UNVERIFIABLE)
    require(
        sidecar.shape == () and h5py.check_string_dtype(sidecar.dtype) is not None,
        SIDECAR_NOT_STRING,
    )
    raw = sidecar[()]
    commit = _sidecar_commit(raw if isinstance(raw, bytes) else bytes(raw))
    require(
        commit is not None
        and FULL_COMMIT.match(commit) is not None
        and commit == source_commit,
        CONTRADICTS,
    )
