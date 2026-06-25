#!/usr/bin/env python3
"""Serve the Palette group analytics web viewer."""

from __future__ import annotations

import argparse
import ipaddress
from pathlib import Path
import sys
from typing import Optional, Sequence

from ..group_analytics_viewer import build_config, run_server


DEFAULT_EXPORT_ROOT = Path("/nvme1/exports/palette_analytics")


def _host_is_loopback(host: str) -> bool:
    value = str(host).strip().lower()
    if value == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _print_network_exposure_warning(host: str, port: int) -> None:
    if _host_is_loopback(host):
        return
    print(
        "WARNING: binding the group analytics viewer to a non-loopback host exposes it "
        "to other reachable machines.",
        file=sys.stderr,
    )
    print(f"  bind: http://{host}:{int(port)}", file=sys.stderr)
    print(
        "  recommendation: keep the app on 127.0.0.1 behind SSH port forwarding or a "
        "reverse proxy with auth/TLS.",
        file=sys.stderr,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve a read-only web viewer over Palette group analytics Parquet exports.",
    )
    parser.add_argument(
        "--export-root",
        type=Path,
        default=DEFAULT_EXPORT_ROOT,
        help="Analytics export root (default: /nvme1/exports/palette_analytics).",
    )
    parser.add_argument(
        "--export-run-id",
        default="latest",
        help="Export run id to view, or latest (default: latest).",
    )
    parser.add_argument(
        "--stats-run-id",
        default="auto",
        help=(
            "Optional statistics export run id to overlay. Use auto/latest to discover "
            "the latest stats run whose source_export_run_id matches the viewed export."
        ),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="Accepted for command symmetry; the MVP reads the export manifest and Parquet files directly.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8770, help="Bind port (default: 8770).")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        config = build_config(
            export_root=args.export_root,
            export_run_id=str(args.export_run_id),
            stats_run_id=str(args.stats_run_id),
            host=str(args.host),
            port=int(args.port),
        )
    except Exception as exc:
        print(f"Group analytics viewer failed to start: {exc}", file=sys.stderr)
        return 2
    if args.registry is not None:
        registry_path = args.registry.expanduser().resolve()
        if not registry_path.exists():
            print(f"WARNING: registry path was provided but does not exist: {registry_path}", file=sys.stderr)
    _print_network_exposure_warning(str(config.host), int(config.port))
    return run_server(config)


if __name__ == "__main__":
    raise SystemExit(main())
