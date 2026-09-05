from __future__ import annotations

import pytest

from fisheye.shared.bounded_identity_cache import BoundedIdentityCache


def test_identity_cache_reuses_only_the_complete_exact_key() -> None:
    cache: BoundedIdentityCache[tuple[str, str], object] = BoundedIdentityCache(
        max_entries=2
    )
    calls: list[str] = []

    def load(label: str) -> object:
        calls.append(label)
        return object()

    first = cache.get_or_load(("receipt-a", "view-a"), lambda: load("first"))
    repeated = cache.get_or_load(
        ("receipt-a", "view-a"), lambda: load("unexpected")
    )
    changed_receipt = cache.get_or_load(
        ("receipt-b", "view-a"), lambda: load("changed-receipt")
    )
    changed_view = cache.get_or_load(
        ("receipt-b", "view-b"), lambda: load("changed-view")
    )

    assert repeated is first
    assert changed_receipt is not first
    assert changed_view is not changed_receipt
    assert calls == ["first", "changed-receipt", "changed-view"]
    assert len(cache) == 2


def test_identity_cache_uses_lru_eviction() -> None:
    cache: BoundedIdentityCache[str, str] = BoundedIdentityCache(max_entries=2)
    cache.get_or_load("a", lambda: "first-a")
    cache.get_or_load("b", lambda: "first-b")
    cache.get_or_load("a", lambda: "unexpected-a")
    cache.get_or_load("c", lambda: "first-c")

    assert cache.get_or_load("a", lambda: "unexpected-a") == "first-a"
    assert cache.get_or_load("b", lambda: "second-b") == "second-b"


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_identity_cache_rejects_invalid_bounds(value: object) -> None:
    with pytest.raises(ValueError, match="positive exact integer"):
        BoundedIdentityCache(max_entries=value)  # type: ignore[arg-type]


def test_identity_cache_rejects_unhashable_keys_and_noncallable_loaders() -> None:
    cache = BoundedIdentityCache(max_entries=1)
    with pytest.raises(TypeError, match="hashable"):
        cache.get_or_load([], lambda: object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="callable"):
        cache.get_or_load("key", None)  # type: ignore[arg-type]
