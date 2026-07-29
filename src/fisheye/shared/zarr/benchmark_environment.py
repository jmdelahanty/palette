"""Deterministic process environment shared by storage benchmark runners."""

from types import MappingProxyType
from typing import Mapping


STORAGE_BENCHMARK_THREAD_ENVIRONMENT: Mapping[str, str] = MappingProxyType(
    {
        "BLIS_NUM_THREADS": "1",
        "MKL_DYNAMIC": "FALSE",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
)


__all__ = ["STORAGE_BENCHMARK_THREAD_ENVIRONMENT"]
