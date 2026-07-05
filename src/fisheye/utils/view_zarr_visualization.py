#!/usr/bin/env python3
"""View or export PNG visualization artifacts directly from a Palette Zarr store."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import zarr

from fisheye.shared.zarr_io import open_zarr_root


@dataclass(frozen=True)
class VisualizationArtifact:
    path: str
    node_type: str
    artifact_role: Optional[str]
    media_type: Optional[str]
    description: Optional[str]
    snapshot_artifact: Optional[str]


def _decode_text(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", "ignore")
    text = str(value).strip()
    return text or None


def _normalize_path(path: str) -> str:
    return "/".join(part for part in str(path).strip("/").split("/") if part)


def _join_path(*parts: str) -> str:
    return "/".join(_normalize_path(part) for part in parts if _normalize_path(part))


def _node_kind(node: object) -> str:
    if isinstance(node, zarr.Array):
        return "array"
    if isinstance(node, zarr.Group):
        return "group"
    if hasattr(node, "shape") and hasattr(node, "dtype"):
        return "array"
    if hasattr(node, "group_keys") or hasattr(node, "array_keys"):
        return "group"
    return type(node).__name__


def _group_names(group: zarr.Group) -> list[str]:
    try:
        return sorted(str(name) for name in group.group_keys())
    except Exception:  # pragma: no cover - compatibility fallback
        return sorted(str(name) for name, node in group.items() if isinstance(node, zarr.Group))


def _array_names(group: zarr.Group) -> list[str]:
    try:
        return sorted(str(name) for name in group.array_keys())
    except Exception:  # pragma: no cover - compatibility fallback
        return sorted(str(name) for name, node in group.items() if isinstance(node, zarr.Array))


def _child_names(group: zarr.Group) -> list[str]:
    return sorted(set(_group_names(group)) | set(_array_names(group)))


def _is_visualizations_group(path: str, group: zarr.Group) -> bool:
    return path.endswith("/visualizations") or path == "visualizations" or group.attrs.get("visualization_group") is True


def _artifact_from_node(path: str, node: object) -> VisualizationArtifact:
    attrs = getattr(node, "attrs", {})
    return VisualizationArtifact(
        path=path,
        node_type=_node_kind(node),
        artifact_role=_decode_text(attrs.get("artifact_role")),
        media_type=_decode_text(attrs.get("media_type") or attrs.get("mime")),
        description=_decode_text(attrs.get("description")),
        snapshot_artifact=_decode_text(attrs.get("snapshot_artifact")),
    )


def iter_visualization_artifacts(root: zarr.Group) -> Iterable[VisualizationArtifact]:
    """Yield immediate children of every ``visualizations`` group in the store."""

    def walk(group: zarr.Group, path: str) -> Iterable[VisualizationArtifact]:
        if _is_visualizations_group(path, group):
            for child_name in _child_names(group):
                child_path = _join_path(path, child_name)
                try:
                    yield _artifact_from_node(child_path, group[child_name])
                except Exception:
                    continue
            return
        for child_name in _group_names(group):
            child_path = _join_path(path, child_name)
            try:
                child = group[child_name]
            except Exception:
                continue
            if isinstance(child, zarr.Group) or hasattr(child, "group_keys"):
                yield from walk(child, child_path)

    yield from walk(root, "")


def _resolve_artifact_path(
    *,
    artifact_path: Optional[str],
    run_path: Optional[str],
    artifact_name: Optional[str],
) -> str:
    if artifact_path and (run_path or artifact_name):
        raise ValueError("Use either artifact_path or --run-path/--artifact, not both.")
    if artifact_path:
        return _normalize_path(artifact_path)
    if not run_path or not artifact_name:
        raise ValueError("Provide artifact_path or both --run-path and --artifact.")
    return _join_path(run_path, "visualizations", artifact_name)


def _resolve_node(root: zarr.Group, path: str) -> object:
    normalized = _normalize_path(path)
    if not normalized:
        raise ValueError("artifact path must not be empty")
    try:
        return root[normalized]
    except Exception as exc:
        raise ValueError(f"Artifact path not found: {normalized}") from exc


def _parent_path(path: str) -> str:
    parts = _normalize_path(path).split("/")
    return "/".join(parts[:-1])


def _resolve_png_array(root: zarr.Group, artifact_path: str) -> tuple[str, zarr.Array]:
    node = _resolve_node(root, artifact_path)
    if isinstance(node, zarr.Array) or (hasattr(node, "shape") and hasattr(node, "dtype")):
        return _normalize_path(artifact_path), node  # type: ignore[return-value]
    if isinstance(node, zarr.Group) or hasattr(node, "attrs"):
        snapshot = _decode_text(getattr(node, "attrs", {}).get("snapshot_artifact"))
        if snapshot:
            candidate = _join_path(_parent_path(artifact_path), snapshot)
            candidate_node = _resolve_node(root, candidate)
            if isinstance(candidate_node, zarr.Array) or (
                hasattr(candidate_node, "shape") and hasattr(candidate_node, "dtype")
            ):
                return candidate, candidate_node  # type: ignore[return-value]
        raise ValueError(
            f"Artifact '{artifact_path}' is not a PNG byte array. "
            "If it is an interactive spec, set snapshot_artifact or choose its PNG snapshot."
        )
    raise ValueError(f"Unsupported artifact node type at {artifact_path}: {_node_kind(node)}")


def load_png_artifact_bytes(root: zarr.Group, artifact_path: str) -> tuple[str, bytes]:
    resolved_path, node = _resolve_png_array(root, artifact_path)
    attrs = getattr(node, "attrs", {})
    media_type = _decode_text(attrs.get("media_type") or attrs.get("mime"))
    if media_type and media_type != "image/png":
        raise ValueError(f"Artifact '{resolved_path}' is not image/png (media_type={media_type}).")
    data = np.asarray(node[:], dtype=np.uint8).tobytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"Artifact '{resolved_path}' does not start with a PNG signature.")
    return resolved_path, data


def export_png_artifact(
    root: zarr.Group,
    artifact_path: str,
    output_path: Path,
) -> tuple[str, Path]:
    """Write a PNG visualization artifact to a filesystem path."""

    resolved_path, png_bytes = load_png_artifact_bytes(root, artifact_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(png_bytes)
    return resolved_path, output_path


def _view_png_bytes(png_bytes: bytes, *, title: str, figsize: tuple[float, float]) -> None:
    import matplotlib.pyplot as plt

    image = plt.imread(BytesIO(png_bytes), format="png")
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(title)
    plt.show()
    plt.close(fig)


def _print_artifact_list(artifacts: Iterable[VisualizationArtifact]) -> int:
    count = 0
    for artifact in artifacts:
        count += 1
        details = []
        if artifact.media_type:
            details.append(artifact.media_type)
        if artifact.artifact_role:
            details.append(artifact.artifact_role)
        if artifact.snapshot_artifact:
            details.append(f"snapshot={artifact.snapshot_artifact}")
        suffix = f" ({', '.join(details)})" if details else ""
        description = f" - {artifact.description}" if artifact.description else ""
        print(f"{artifact.path}{suffix}{description}")
    return count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "View or export PNG visualization artifacts directly from a Palette Zarr archive."
        )
    )
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive.")
    parser.add_argument(
        "artifact_path",
        nargs="?",
        help="Path inside the zarr store to a PNG visualization artifact.",
    )
    parser.add_argument(
        "--run-path",
        help="Run path containing a visualizations group, e.g. analysis/track_kinematics_runs/offline/<run>.",
    )
    parser.add_argument("--artifact", help="Artifact name under <run-path>/visualizations.")
    parser.add_argument("--list", action="store_true", help="List visualization artifacts and exit.")
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Write the resolved PNG bytes to this filesystem path. When set, the "
            "Matplotlib viewer is skipped unless --show is also passed."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="With --output, also open the Matplotlib viewer after writing the file.",
    )
    parser.add_argument("--title", help="Optional Matplotlib window title.")
    parser.add_argument("--figsize", default="16,10", help="Figure size as WIDTH,HEIGHT inches (default: 16,10).")
    return parser


def _parse_figsize(value: str) -> tuple[float, float]:
    try:
        width_raw, height_raw = value.split(",", 1)
        width = float(width_raw)
        height = float(height_raw)
    except Exception as exc:
        raise ValueError("--figsize must be WIDTH,HEIGHT") from exc
    if width <= 0 or height <= 0:
        raise ValueError("--figsize values must be positive")
    return width, height


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        root = open_zarr_root(args.zarr_path, mode="r")
        if args.list:
            count = _print_artifact_list(iter_visualization_artifacts(root))
            if count == 0:
                print("No visualization artifacts found.")
            return 0

        artifact_path = _resolve_artifact_path(
            artifact_path=args.artifact_path,
            run_path=args.run_path,
            artifact_name=args.artifact,
        )
        resolved_path, png_bytes = load_png_artifact_bytes(root, artifact_path)
        if args.output is not None:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(png_bytes)
            print(output_path)
            if not args.show:
                return 0

        title = args.title or resolved_path
        _view_png_bytes(png_bytes, title=title, figsize=_parse_figsize(args.figsize))
        return 0
    except Exception as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
