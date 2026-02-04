#!/usr/bin/env python3
"""
List versioned training configs/manifests created by prepare_detect_training.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class VersionEntry:
    versions: Set[int]
    config_versions: Set[int]
    manifest_versions: Set[int]


def _parse_versions(directory: Path, pattern: re.Pattern) -> List[Tuple[str, int]]:
    entries: List[Tuple[str, int]] = []
    if not directory.exists():
        return entries
    for path in directory.iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if not match:
            continue
        name = match.group("name")
        version = int(match.group("version"))
        entries.append((name, version))
    return entries


def _collect_versions(config_dir: Path, manifest_dir: Path) -> Dict[str, VersionEntry]:
    config_pattern = re.compile(r"^(?P<name>.+)_v(?P<version>\d+)\.ya?ml$")
    manifest_pattern = re.compile(r"^(?P<name>.+)_v(?P<version>\d+)\.manifest\.json$")

    entries: Dict[str, VersionEntry] = {}

    for name, version in _parse_versions(config_dir, config_pattern):
        entry = entries.setdefault(name, VersionEntry(set(), set(), set()))
        entry.versions.add(version)
        entry.config_versions.add(version)

    for name, version in _parse_versions(manifest_dir, manifest_pattern):
        entry = entries.setdefault(name, VersionEntry(set(), set(), set()))
        entry.versions.add(version)
        entry.manifest_versions.add(version)

    return entries


def _format_versions(versions: Set[int]) -> str:
    if not versions:
        return "-"
    return ", ".join(f"{v:03d}" for v in sorted(versions))


def _print_entries(entries: Dict[str, VersionEntry], name_filter: Optional[str]) -> None:
    if not entries:
        print("No versioned training sets found.")
        return

    names = sorted(entries.keys())
    if name_filter:
        names = [name for name in names if name == name_filter]

    if not names:
        print("No matching sets found.")
        return

    for name in names:
        entry = entries[name]
        versions_sorted = sorted(entry.versions)
        latest = versions_sorted[-1] if versions_sorted else None
        next_version = (latest + 1) if latest else 1

        print(f"Set: {name}")
        print(f"  versions: {_format_versions(entry.versions)}")
        print(f"  configs: {_format_versions(entry.config_versions)}")
        print(f"  manifests: {_format_versions(entry.manifest_versions)}")
        if latest is not None:
            print(f"  latest: v{latest:03d}")
        print(f"  next: v{next_version:03d}")

        missing_config = entry.versions - entry.config_versions
        missing_manifest = entry.versions - entry.manifest_versions
        if missing_config:
            print(f"  warning: missing configs for { _format_versions(missing_config) }")
        if missing_manifest:
            print(f"  warning: missing manifests for { _format_versions(missing_manifest) }")
        print("")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="List versioned training config/manifest files.",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Optional set name to filter (e.g. detect_base).",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("runs") / "configs" / "detect",
        help="Directory containing versioned config YAML files.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("runs") / "manifests" / "detect",
        help="Directory containing versioned manifest JSON files.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable output.",
    )
    args = parser.parse_args(argv)

    entries = _collect_versions(args.config_dir, args.manifest_dir)
    if args.json:
        payload = {}
        for name, entry in entries.items():
            if args.name and name != args.name:
                continue
            versions_sorted = sorted(entry.versions)
            latest = versions_sorted[-1] if versions_sorted else None
            payload[name] = {
                "versions": versions_sorted,
                "config_versions": sorted(entry.config_versions),
                "manifest_versions": sorted(entry.manifest_versions),
                "latest": latest,
                "next": (latest + 1) if latest else 1,
            }
        print(json.dumps(payload, indent=2))
        return 0

    _print_entries(entries, args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
