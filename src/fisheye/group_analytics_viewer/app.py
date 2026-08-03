"""Server bootstrap for the group analytics viewer."""

from __future__ import annotations

from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys

from .api import build_handler
from .query import ViewerContext, build_context


@dataclass(frozen=True)
class GroupAnalyticsViewerConfig:
    context: ViewerContext
    host: str
    port: int
    static_dir: Path


def build_config(
    *,
    export_root: Path,
    export_run_id: str,
    stats_run_id: str | None = None,
    allow_legacy_statistics: bool = False,
    host: str,
    port: int,
) -> GroupAnalyticsViewerConfig:
    context = build_context(
        export_root=export_root,
        export_run_id=export_run_id,
        stats_run_id=stats_run_id,
        allow_legacy_statistics=allow_legacy_statistics,
    )
    static_dir = Path(__file__).resolve().parent / "static"
    if not static_dir.is_dir():
        raise RuntimeError(f"Group analytics viewer static directory missing: {static_dir}")
    return GroupAnalyticsViewerConfig(
        context=context,
        host=host,
        port=int(port),
        static_dir=static_dir,
    )


def run_server(config: GroupAnalyticsViewerConfig) -> int:
    handler_cls = build_handler(
        context=config.context,
        static_dir=config.static_dir,
    )
    server = ThreadingHTTPServer((config.host, int(config.port)), handler_cls)
    local_url = f"http://{config.host}:{int(config.port)}"
    print("Palette group analytics viewer")
    print(f"  export_root:   {config.context.export_root}")
    print(f"  export_run_id: {config.context.export_run_id}")
    print(f"  stats_run_id:  {config.context.stats_run_id or 'auto'}")
    print(
        "  legacy_stats: "
        f"{'enabled' if config.context.allow_legacy_statistics else 'disabled'}"
    )
    print(f"  static:        {config.static_dir}")
    print(f"  url:           {local_url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down group analytics viewer...", file=sys.stderr)
    finally:
        server.server_close()
    return 0
