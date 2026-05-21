"""Zarr-to-registry row extractors.

Extractor modules are intentionally kept independent of ``registry.db`` so
stage-specific Zarr schema readers can evolve without expanding the SQLite
connection layer.
"""
