"""Small process-local caches for immutable, fully qualified identities.

This helper deliberately knows nothing about files, selectors, or scientific
authority. Callers must provide a hashable key that closes the complete
receipt/manifest and display identity for the cached value.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable
from threading import RLock
from typing import Generic, TypeVar


KeyT = TypeVar("KeyT", bound=Hashable)
ValueT = TypeVar("ValueT")


class BoundedIdentityCache(Generic[KeyT, ValueT]):
    """Bounded LRU cache whose misses are populated by one explicit loader."""

    __slots__ = ("_lock", "_max_entries", "_values")

    def __init__(self, *, max_entries: int) -> None:
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("max_entries must be a positive exact integer")
        self._max_entries = max_entries
        self._values: OrderedDict[KeyT, ValueT] = OrderedDict()
        self._lock = RLock()

    @property
    def max_entries(self) -> int:
        return self._max_entries

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)

    def get_or_load(self, key: KeyT, loader: Callable[[], ValueT]) -> ValueT:
        """Return the exact-key value or load, retain, and return one value."""

        if not callable(loader):
            raise TypeError("loader must be callable")
        try:
            hash(key)
        except TypeError as exc:
            raise TypeError("cache key must be hashable") from exc
        with self._lock:
            try:
                value = self._values.pop(key)
            except KeyError:
                value = loader()
            self._values[key] = value
            while len(self._values) > self._max_entries:
                self._values.popitem(last=False)
            return value

    def clear(self) -> None:
        with self._lock:
            self._values.clear()
