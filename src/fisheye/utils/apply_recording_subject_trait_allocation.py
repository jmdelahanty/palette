#!/usr/bin/env python3
"""Validate or atomically apply a recording-subject trait allocation."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.registry.db import Registry


TRAIT_COLUMNS = (
    "pigmentation_phenotype",
    "melanophore_status",
    "xanthophore_status",
    "iridophore_status",
    "pigment_pattern_status",
    "optical_transparency",
)


@dataclass(frozen=True)
class TraitAllocation:
    allocation_id: str
    source_path: Path
    rows: tuple[dict[str, str], ...]
    species: str
    source_label: str
    canonical_strain: str
    mapping_method: str
    assignment_method: str
    assigned_at_utc: str
    assigned_by: str | None
    traits: dict[str, str]


def _required_text(mapping: Mapping[str, Any], key: str) -> str:
    value = str(mapping.get(key) or "").strip()
    if not value:
        raise ValueError(f"Missing required non-empty field: {key}.")
    return value


def _load_allocation(path: Path) -> TraitAllocation:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_id") != "palette.keypoint_recording_allocation.v1":
        raise ValueError("Unsupported recording allocation schema_id.")
    source = payload.get("source")
    strain = payload.get("strain_resolution")
    assignment = payload.get("pigmentation_assignment")
    if not isinstance(source, Mapping):
        raise ValueError("Allocation source must be an object.")
    if not isinstance(strain, Mapping):
        raise ValueError("Allocation strain_resolution must be an object.")
    if not isinstance(assignment, Mapping):
        raise ValueError("Allocation pigmentation_assignment must be an object.")
    traits = assignment.get("traits")
    if not isinstance(traits, Mapping) or set(traits) != set(TRAIT_COLUMNS):
        raise ValueError(
            "pigmentation_assignment.traits must contain exactly: "
            + ", ".join(TRAIT_COLUMNS)
        )
    normalized_traits = {
        key: _required_text(traits, key).lower() for key in TRAIT_COLUMNS
    }

    table_name = _required_text(payload, "recording_table")
    table_path = path.parent / table_name
    with table_path.open("r", encoding="utf-8", newline="") as handle:
        rows = tuple(dict(row) for row in csv.DictReader(handle))
    expected_count = int(source.get("recording_count") or 0)
    if expected_count <= 0 or len(rows) != expected_count:
        raise ValueError(
            f"Recording table row count {len(rows)} does not match source count "
            f"{expected_count}."
        )

    species = _required_text(strain, "species")
    source_label = _required_text(strain, "source_husbandry_label")
    canonical_strain = _required_text(strain, "canonical_strain")
    seen_identities: set[tuple[str, str]] = set()
    for row in rows:
        identity = (
            _required_text(row, "recording_id"),
            _required_text(row, "subject_id"),
        )
        if identity in seen_identities:
            raise ValueError(f"Duplicate recording-subject identity: {identity!r}.")
        seen_identities.add(identity)
        if _required_text(row, "species") != species:
            raise ValueError(f"Species mismatch for {identity!r}.")
        if _required_text(row, "line_strain") != source_label:
            raise ValueError(f"Source husbandry label mismatch for {identity!r}.")
        if _required_text(row, "canonical_strain") != canonical_strain:
            raise ValueError(f"Canonical strain mismatch for {identity!r}.")
        if _required_text(row, "pigmentation_value_origin") != "subject_observed":
            raise ValueError(f"Batman trait value origin must be subject_observed: {identity!r}.")
        for trait_name, expected_value in normalized_traits.items():
            if _required_text(row, trait_name).lower() != expected_value:
                raise ValueError(f"{trait_name} mismatch for {identity!r}.")

    return TraitAllocation(
        allocation_id=_required_text(payload, "allocation_id"),
        source_path=path,
        rows=rows,
        species=species,
        source_label=source_label,
        canonical_strain=canonical_strain,
        mapping_method=_required_text(strain, "mapping_method"),
        assignment_method=_required_text(assignment, "assignment_method"),
        assigned_at_utc=_required_text(assignment, "assigned_at_utc"),
        assigned_by=(
            str(assignment["assigned_by"]).strip()
            if assignment.get("assigned_by")
            else None
        ),
        traits=normalized_traits,
    )


def _validate_registry_subjects(
    conn: sqlite3.Connection,
    allocation: TraitAllocation,
) -> None:
    conn.row_factory = sqlite3.Row
    for source_row in allocation.rows:
        recording_id = source_row["recording_id"]
        subject_id = source_row["subject_id"]
        row = conn.execute(
            """
            SELECT dataset_id, species, line_strain
            FROM recording_subject_overview
            WHERE recording_id = ? AND subject_id = ?;
            """,
            (recording_id, subject_id),
        ).fetchone()
        if row is None:
            raise ValueError(
                "Registry is missing allocation identity "
                f"({recording_id!r}, {subject_id!r})."
            )
        if str(row["dataset_id"] or "").strip() != source_row["dataset_id"]:
            raise ValueError(f"Registry dataset_id mismatch for {recording_id!r}.")
        if str(row["species"] or "").strip() != allocation.species:
            raise ValueError(f"Registry species mismatch for {recording_id!r}.")
        if str(row["line_strain"] or "").strip() != allocation.source_label:
            raise ValueError(f"Registry source label mismatch for {recording_id!r}.")


def _apply_allocation(
    *,
    registry_path: Path,
    allocation: TraitAllocation,
    assigned_by_override: str | None,
) -> None:
    assigned_by = assigned_by_override or allocation.assigned_by
    registry = Registry(registry_path)
    try:
        _validate_registry_subjects(registry.conn, allocation)
        evidence = {
            "allocation_id": allocation.allocation_id,
            "allocation_path": str(allocation.source_path),
            "source_husbandry_label": allocation.source_label,
            "source_label_suffix_semantics": "uninterpreted",
        }
        with registry._transaction_context():
            registry.upsert_strain_label_mapping(
                species=allocation.species,
                source_label=allocation.source_label,
                canonical_strain=allocation.canonical_strain,
                assignment_method=allocation.mapping_method,
                assigned_by=assigned_by,
                assigned_at_utc=allocation.assigned_at_utc,
                evidence=evidence,
            )
            for trait_name, trait_value in allocation.traits.items():
                registry.upsert_strain_trait_expectation(
                    species=allocation.species,
                    canonical_strain=allocation.canonical_strain,
                    trait_name=trait_name,
                    trait_value=trait_value,
                    assignment_method="curated_strain_reference",
                    assigned_by=assigned_by,
                    assigned_at_utc=allocation.assigned_at_utc,
                    evidence={**evidence, "zfin_id": "ZDB-GENO-960809-7"},
                )
            for row in allocation.rows:
                for trait_name, trait_value in allocation.traits.items():
                    registry.upsert_recording_subject_trait(
                        recording_id=row["recording_id"],
                        subject_id=row["subject_id"],
                        trait_name=trait_name,
                        trait_value=trait_value,
                        assignment_method=allocation.assignment_method,
                        assigned_by=assigned_by,
                        assigned_at_utc=allocation.assigned_at_utc,
                        evidence=evidence,
                    )

            placeholders = ", ".join("?" for _ in allocation.rows)
            recording_ids = tuple(row["recording_id"] for row in allocation.rows)
            resolved = registry.conn.execute(
                f"""
                SELECT COUNT(*) AS n
                FROM recording_subject_trait_resolved
                WHERE recording_id IN ({placeholders})
                  AND trait_name IN (
                    'pigmentation_phenotype', 'melanophore_status',
                    'xanthophore_status', 'iridophore_status',
                    'pigment_pattern_status', 'optical_transparency'
                  )
                  AND value_origin = 'subject_observed';
                """,
                recording_ids,
            ).fetchone()
            expected = len(allocation.rows) * len(TRAIT_COLUMNS)
            if resolved is None or int(resolved["n"]) != expected:
                raise RuntimeError(
                    "Resolved trait validation failed: expected "
                    f"{expected} subject observations."
                )
    finally:
        registry.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--assigned-by")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    allocation = _load_allocation(args.allocation.resolve())
    if args.apply:
        _apply_allocation(
            registry_path=args.registry.resolve(),
            allocation=allocation,
            assigned_by_override=(str(args.assigned_by).strip() if args.assigned_by else None),
        )
        mode = "applied"
    else:
        uri = f"file:{args.registry.resolve()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            _validate_registry_subjects(conn, allocation)
        finally:
            conn.close()
        mode = "validated_read_only"
    print(
        json.dumps(
            {
                "status": "ok",
                "mode": mode,
                "allocation_id": allocation.allocation_id,
                "recording_subject_count": len(allocation.rows),
                "trait_count_per_subject": len(TRAIT_COLUMNS),
                "canonical_strain": allocation.canonical_strain,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
