#!/usr/bin/env python3
"""Compose detection occupancy and chaser-distance PNG artifacts for one zarr."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import zarr  # noqa: E402

from fisheye.analysis.chaser_distance_runs import PNG_ARTIFACT_NAME as CHASER_DISTANCE_ARTIFACT
from fisheye.analysis.detection_occupancy_runs import PNG_ARTIFACT_NAME as DETECTION_OCCUPANCY_ARTIFACT
from fisheye.utils.view_zarr_visualization import load_png_artifact_bytes
from fisheye.utils.zarr_io import open_zarr_root


DETECTION_OCCUPANCY_PARENT = "analysis/detection_occupancy_runs"
CHASER_DISTANCE_PARENT = "analysis/chaser_distance_runs"


@dataclass(frozen=True)
class ResolvedArtifact:
    title: str
    run_path: str
    artifact_name: str
    resolved_artifact_path: str
    png_bytes: bytes


def _normalize_path(path: str) -> str:
    return "/".join(part for part in str(path).strip("/").split("/") if part)


def _join_path(*parts: str) -> str:
    return "/".join(_normalize_path(part) for part in parts if _normalize_path(part))


def _run_names(parent: zarr.Group) -> set[str]:
    try:
        return {str(name) for name in parent.group_keys()}
    except Exception:  # pragma: no cover - compatibility fallback
        return {str(name) for name, node in parent.items() if isinstance(node, zarr.Group)}


def _resolve_run_path(root: zarr.Group, parent_path: str, run_name_or_path: Optional[str]) -> str:
    parent_path = _normalize_path(parent_path)
    if parent_path not in root:
        raise ValueError(f"Missing run parent: {parent_path}")
    parent = root[parent_path]

    if run_name_or_path:
        requested = _normalize_path(run_name_or_path)
        if requested.startswith(parent_path + "/"):
            if requested not in root:
                raise ValueError(f"Requested run path not found: {requested}")
            return requested
        if requested not in _run_names(parent):
            raise ValueError(f"Requested run {requested!r} not found under {parent_path}")
        return _join_path(parent_path, requested)

    latest_complete = parent.attrs.get("latest_complete")
    if latest_complete and str(latest_complete).strip() in _run_names(parent):
        return _join_path(parent_path, str(latest_complete).strip())

    latest = parent.attrs.get("latest")
    if latest and str(latest).strip() in _run_names(parent):
        return _join_path(parent_path, str(latest).strip())

    names = sorted(_run_names(parent))
    if not names:
        raise ValueError(f"No runs found under {parent_path}")
    return _join_path(parent_path, names[-1])


def _load_artifact(
    root: zarr.Group,
    *,
    title: str,
    parent_path: str,
    run_name_or_path: Optional[str],
    artifact_name: str,
) -> ResolvedArtifact:
    run_path = _resolve_run_path(root, parent_path, run_name_or_path)
    artifact_path = _join_path(run_path, "visualizations", artifact_name)
    resolved_path, png_bytes = load_png_artifact_bytes(root, artifact_path)
    return ResolvedArtifact(
        title=title,
        run_path=run_path,
        artifact_name=artifact_name,
        resolved_artifact_path=resolved_path,
        png_bytes=png_bytes,
    )


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


def render_combined_png(
    artifacts: Sequence[ResolvedArtifact],
    *,
    title: str,
    layout: str,
    figsize: tuple[float, float],
    dpi: int,
) -> bytes:
    if len(artifacts) != 2:
        raise ValueError("Exactly two artifacts are required.")
    if layout == "horizontal":
        nrows, ncols = 1, 2
    elif layout == "vertical":
        nrows, ncols = 2, 1
    else:
        raise ValueError("layout must be 'vertical' or 'horizontal'")

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes_list = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]
    for ax, artifact in zip(axes_list, artifacts):
        image = plt.imread(BytesIO(artifact.png_bytes), format="png")
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(artifact.title, fontsize=11)
    fig.suptitle(title, fontsize=13)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=int(dpi))
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Palette analysis zarr.")
    parser.add_argument(
        "--occupancy-run",
        help=(
            "Detection occupancy run name or full run path. Defaults to "
            "analysis/detection_occupancy_runs latest_complete."
        ),
    )
    parser.add_argument(
        "--chaser-distance-run",
        help=(
            "Chaser distance run name or full run path. Defaults to "
            "analysis/chaser_distance_runs latest_complete."
        ),
    )
    parser.add_argument(
        "--layout",
        choices=("vertical", "horizontal"),
        default="vertical",
        help="How to arrange the two plots.",
    )
    parser.add_argument("--figsize", default="16,18", help="Figure size as WIDTH,HEIGHT inches.")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--output", type=Path, help="Write the combined PNG to this path.")
    parser.add_argument("--show", action="store_true", help="Open a Matplotlib viewer.")
    parser.add_argument("--print-paths", action="store_true", help="Print resolved source artifact paths.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        root = open_zarr_root(args.zarr_path, mode="r")
        recording_id = str(root.attrs.get("recording_id") or root.attrs.get("recording_name") or args.zarr_path.stem)
        artifacts = [
            _load_artifact(
                root,
                title="Detection occupancy",
                parent_path=DETECTION_OCCUPANCY_PARENT,
                run_name_or_path=args.occupancy_run,
                artifact_name=DETECTION_OCCUPANCY_ARTIFACT,
            ),
            _load_artifact(
                root,
                title="Chaser distance",
                parent_path=CHASER_DISTANCE_PARENT,
                run_name_or_path=args.chaser_distance_run,
                artifact_name=CHASER_DISTANCE_ARTIFACT,
            ),
        ]
        png = render_combined_png(
            artifacts,
            title=f"Detection occupancy + chaser distance: {recording_id}",
            layout=str(args.layout),
            figsize=_parse_figsize(str(args.figsize)),
            dpi=int(args.dpi),
        )
        if args.output is not None:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(png)
            print(output_path)
        if args.print_paths:
            for artifact in artifacts:
                print(f"{artifact.title}: {artifact.resolved_artifact_path}")
        if args.show:
            image = plt.imread(BytesIO(png), format="png")
            fig, ax = plt.subplots(figsize=_parse_figsize(str(args.figsize)))
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(recording_id)
            plt.show()
            plt.close(fig)
        if args.output is None and not args.show and not args.print_paths:
            parser.error("Provide --output, --show, or --print-paths.")
        return 0
    except Exception as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
