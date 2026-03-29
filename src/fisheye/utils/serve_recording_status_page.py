#!/usr/bin/env python3
"""Serve the recording status web page."""

from __future__ import annotations

import argparse
import ipaddress
from pathlib import Path
import sys
from typing import Optional, Sequence

from ..status_page import build_config, run_server


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
        "WARNING: binding the recording status page to a non-loopback host exposes it "
        "to other reachable machines.",
        file=sys.stderr,
    )
    print(
        f"  bind: http://{host}:{int(port)}",
        file=sys.stderr,
    )
    print(
        "  recommendation: keep the app on 127.0.0.1 behind SSH port forwarding or a "
        "reverse proxy with auth/TLS.",
        file=sys.stderr,
    )
    print(
        "  runbook: docs/recording_status_page_deployment.md",
        file=sys.stderr,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Serve a live read-only web status page backed by "
            "recording_step_status registry views."
        )
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help=(
            "Registry sqlite path. Defaults to PALETTE_REGISTRY_PATH/config "
            "resolution via RegistryPaths.from_env()."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Bind port (default: 8765).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        config = build_config(
            registry=args.registry,
            host=str(args.host),
            port=int(args.port),
            cwd=Path.cwd(),
        )
    except Exception as exc:
        print(f"Recording status page failed to start: {exc}", file=sys.stderr)
        return 2
    _print_network_exposure_warning(str(config.host), int(config.port))
    return run_server(config)


if __name__ == "__main__":
    raise SystemExit(main())
