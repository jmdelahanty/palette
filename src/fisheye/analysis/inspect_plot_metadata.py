#!/usr/bin/env python3
"""
Inspect provenance metadata stored in Palette analysis plots.

Phase-analysis PNGs saved by ``chaser_phase_analysis`` embed an XMP packet
containing a JSON document under the ``palette:provenance`` tag. This utility
extracts that payload and prints it in a readable form.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover
    Image = None


PALETTE_NS = "https://palette.hhmi.org/ns/analysis/"
NAMESPACES = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "palette": PALETTE_NS,
}


def _extract_xmp(image_path: Path) -> Optional[str]:
    """Return the raw XMP packet embedded in a PNG (if present)."""
    if Image is None:
        raise RuntimeError(
            "Pillow is required to inspect plot metadata. Install with 'pip install pillow'."
        )

    with Image.open(image_path) as img:
        xmp_payload = None
        for key, value in img.info.items():
            if key.startswith("XML:"):
                xmp_payload = value
                break

    return xmp_payload


def _parse_provenance(xmp_packet: str) -> Dict[str, Any]:
    """Parse Palette provenance JSON from the XMP packet."""
    root = ET.fromstring(xmp_packet)
    provenance_node = root.find(".//palette:provenance", NAMESPACES)
    if provenance_node is None or provenance_node.text is None:
        raise ValueError("No palette:provenance element found in XMP metadata.")
    return json.loads(provenance_node.text)


def inspect_plot(image_path: Path) -> Dict[str, Any]:
    """Extract provenance metadata from the given plot image."""
    xmp_packet = _extract_xmp(image_path)
    if not xmp_packet:
        raise ValueError(f"No XMP metadata found in {image_path}.")
    return _parse_provenance(xmp_packet)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect provenance metadata embedded in Palette analysis plots."
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to a PNG saved by chaser_phase_analysis (contains embedded provenance).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON with indentation (default: compact JSON).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        provenance = inspect_plot(args.image)
    except Exception as exc:  # pragma: no cover
        parser.error(str(exc))
        return 2

    if args.pretty:
        print(json.dumps(provenance, indent=2, sort_keys=True))
    else:
        print(json.dumps(provenance, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
