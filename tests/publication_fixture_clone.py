"""Small helpers for safely cloning sealed in-memory publication fixtures."""

from __future__ import annotations

from typing import Any


def sealed_fixture_copy_memo(template: Any) -> dict[int, Any]:
    """Preserve sealed identities while deepcopy clones mutable archive data."""

    memo: dict[int, Any] = {}
    pending = [template]
    visited: set[int] = set()
    while pending:
        value = pending.pop()
        identity = id(value)
        if identity in visited:
            continue
        visited.add(identity)
        archive_token = getattr(value, "_coordinate_archive_token", None)
        if archive_token is not None:
            # ArchiveIdentity records bind the synthetic store token by object
            # identity. Preserve that authority while copying every mutable
            # group, attribute mapping, and array payload.
            memo[id(archive_token)] = archive_token
        if isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, (list, tuple, set, frozenset)):
            pending.extend(value)
        elif hasattr(value, "__dict__"):
            attributes = vars(value)
            for name, attribute in attributes.items():
                if name.endswith("_seal") and attribute is not None:
                    memo[id(attribute)] = attribute
            pending.extend(attributes.values())
    return memo
