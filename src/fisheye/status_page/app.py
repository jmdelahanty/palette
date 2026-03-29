"""Server bootstrap for the recording status page."""

from __future__ import annotations

from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
from typing import Optional

from .api import build_handler
from .query import assert_registry_ready, resolve_registry_path


@dataclass(frozen=True)
class StatusPageConfig:
    registry_path: Path
    host: str
    port: int
    static_dir: Path


def build_config(
    *,
    registry: Optional[Path],
    host: str,
    port: int,
    cwd: Optional[Path] = None,
) -> StatusPageConfig:
    resolved_registry = resolve_registry_path(registry, cwd=cwd)
    ready_registry = assert_registry_ready(resolved_registry)
    static_dir = Path(__file__).resolve().parent / "static"
    if not static_dir.is_dir():
        raise RuntimeError(f"Status page static directory missing: {static_dir}")
    return StatusPageConfig(
        registry_path=ready_registry,
        host=host,
        port=port,
        static_dir=static_dir,
    )


def run_server(config: StatusPageConfig) -> int:
    handler_cls = build_handler(
        registry_path=config.registry_path,
        static_dir=config.static_dir,
    )
    server = ThreadingHTTPServer((config.host, int(config.port)), handler_cls)
    local_url = f"http://{config.host}:{int(config.port)}"
    print(f"Recording status page")
    print(f"  registry: {config.registry_path}")
    print(f"  static:   {config.static_dir}")
    print(f"  url:      {local_url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down status page server...", file=sys.stderr)
    finally:
        server.server_close()
    return 0
