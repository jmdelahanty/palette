"""Print a Mermaid diagram for the protocol embedded in a Palette Zarr."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import h5py
import zarr


def _read_zarr_json_attrs(path: Path) -> Mapping[str, Any]:
    metadata_path = path / "zarr.json"
    if not metadata_path.exists():
        return {}
    try:
        payload = json.loads(metadata_path.read_text())
    except Exception:
        return {}
    attrs = payload.get("attributes")
    return attrs if isinstance(attrs, Mapping) else {}


def _open_zarr(path: Path) -> zarr.Group:
    try:
        return zarr.open_group(str(path), mode="r", use_consolidated=False)
    except TypeError:
        return zarr.open_group(str(path), mode="r", consolidated=False)


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    text = str(value)
    return text if text else None


def _load_protocol_from_h5(path: Path) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    with h5py.File(path, "r") as h5:
        for dataset_path in (
            "/protocol_snapshot/protocol_definition_json",
            "/protocol_snapshot/protocol_json",
        ):
            if dataset_path not in h5:
                continue
            raw = _as_text(h5[dataset_path][()])
            if raw:
                return json.loads(raw)
    return None


def _group_names_from_disk(path: Path) -> list[str]:
    if not path.is_dir():
        return []
    names: list[str] = []
    for child in path.iterdir():
        if child.is_dir() and (child / "zarr.json").exists():
            names.append(child.name)
    return sorted(names)


def _load_protocol_from_metadata_files(
    zarr_path: Path,
    stimulus_run: str | None,
) -> tuple[Mapping[str, Any], str, str] | None:
    runs_path = zarr_path / "analysis" / "stimulus_runs"
    if not runs_path.is_dir():
        return None

    parent_attrs = _read_zarr_json_attrs(runs_path)
    run_name = stimulus_run or _as_text(parent_attrs.get("latest"))
    if not run_name:
        names = _group_names_from_disk(runs_path)
        if not names:
            return None
        run_name = names[-1]

    run_path = runs_path / run_name
    if not run_path.is_dir():
        return None

    run_attrs = _read_zarr_json_attrs(run_path)
    protocol_json = _as_text(run_attrs.get("protocol_json"))
    if protocol_json:
        return json.loads(protocol_json), run_name, f"zarr:{run_path}"

    source_h5 = _as_text(run_attrs.get("source_h5"))
    if source_h5:
        protocol = _load_protocol_from_h5(Path(source_h5))
        if protocol is not None:
            return protocol, run_name, f"h5:{source_h5}"

    protocol_attrs = _read_zarr_json_attrs(zarr_path / "calibration" / "protocol_info")
    if protocol_attrs:
        name = _as_text(protocol_attrs.get("name")) or "unknown"
        steps = int(protocol_attrs.get("steps") or 0)
        return {"protocol_name": name, "steps": [{"name": f"Step {idx + 1}"} for idx in range(steps)]}, run_name, "zarr:calibration/protocol_info"

    return None


def _select_stimulus_run(root: zarr.Group, requested: str | None) -> tuple[zarr.Group, str]:
    try:
        parent = root["analysis/stimulus_runs"]
    except Exception as exc:
        raise ValueError("Zarr archive is missing analysis/stimulus_runs.") from exc

    run_name = requested or _as_text(parent.attrs.get("latest"))
    if not run_name:
        names = sorted(str(name) for name in parent.group_keys())
        if not names:
            raise ValueError("analysis/stimulus_runs contains no runs.")
        run_name = names[-1]
    if run_name not in parent:
        available = ", ".join(sorted(str(name) for name in parent.group_keys()))
        raise ValueError(f"Stimulus run '{run_name}' not found. Available: {available}")
    return parent[run_name], run_name


def load_protocol(zarr_path: Path, stimulus_run: str | None) -> tuple[Mapping[str, Any], str, str]:
    metadata_result = _load_protocol_from_metadata_files(zarr_path, stimulus_run)
    if metadata_result is not None:
        return metadata_result

    root = _open_zarr(zarr_path)
    run_group, run_name = _select_stimulus_run(root, stimulus_run)

    protocol_json = _as_text(run_group.attrs.get("protocol_json"))
    if protocol_json:
        return (
            json.loads(protocol_json),
            run_name,
            f"zarr:{zarr_path}/analysis/stimulus_runs/{run_name}",
        )

    source_h5 = _as_text(run_group.attrs.get("source_h5"))
    if source_h5:
        protocol = _load_protocol_from_h5(Path(source_h5))
        if protocol is not None:
            return protocol, run_name, f"h5:{source_h5}"

    protocol_info = root.get("calibration/protocol_info")
    if protocol_info is not None:
        name = _as_text(protocol_info.attrs.get("name")) or "unknown"
        steps = int(protocol_info.attrs.get("steps") or 0)
        return {"protocol_name": name, "steps": [{"name": f"Step {idx + 1}"} for idx in range(steps)]}, run_name, "zarr:calibration/protocol_info"

    raise ValueError("No protocol_json, source H5 protocol snapshot, or calibration/protocol_info found.")


def _escape_label(value: Any) -> str:
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return text.replace("\n", "<br/>")


def _format_seconds(value: Any) -> str | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return str(value)
    if seconds.is_integer():
        return f"{int(seconds)}s"
    return f"{seconds:g}s"


def _format_degrees(value: Any) -> str | None:
    if value is None:
        return None
    try:
        degrees = float(value)
    except (TypeError, ValueError):
        return str(value)
    if degrees.is_integer():
        return f"{int(degrees)} deg"
    return f"{degrees:g} deg"


def _format_speed(value: Any) -> str | None:
    if value is None:
        return None
    try:
        speed = float(value)
    except (TypeError, ValueError):
        return str(value)
    if speed.is_integer():
        return f"{int(speed)} mm/s"
    return f"{speed:g} mm/s"


def _step_label(index: int, step: Mapping[str, Any], *, detail: str) -> str:
    lines = [f"Step {index + 1}: {step.get('name') or 'unnamed'}"]
    if detail == "names":
        return "<br/>".join(_escape_label(line) for line in lines)
    params = step.get("parameters")
    mode = step.get("stimulus_mode_str")
    if detail == "moving-grating-orientation":
        if mode == "MOVING_GRATING" and isinstance(params, Mapping):
            orientation = _format_degrees(params.get("orientation_degrees"))
            if orientation:
                lines.append(f"orientation: {orientation}")
            speed = _format_speed(params.get("speed_mm_per_sec"))
            if speed:
                lines.append(f"speed: {speed}")
        return "<br/>".join(_escape_label(line) for line in lines)
    if mode:
        lines.append(f"mode: {mode}")
    duration = _format_seconds(step.get("duration_seconds"))
    if duration:
        lines.append(f"duration: {duration}")
    if isinstance(params, Mapping):
        compact = ", ".join(
            f"{key}={value}"
            for key, value in sorted(params.items())
            if key not in {"type"} and value is not None
        )
        if compact:
            lines.append(compact)
    return "<br/>".join(_escape_label(line) for line in lines)


def protocol_to_mermaid(
    protocol: Mapping[str, Any],
    *,
    source_label: str,
    stimulus_run: str,
    detail: str = "full",
) -> str:
    protocol_name = protocol.get("protocol_name") or protocol.get("name") or "unknown"
    lines = [
        "flowchart TD",
        f'  P["Protocol: {_escape_label(protocol_name)}"]',
    ]
    if detail != "names":
        lines.extend(
            [
                f'  M["Stimulus run: {_escape_label(stimulus_run)}"]',
                f'  SRC["Source: {_escape_label(source_label)}"]',
                "  SRC --> M --> P",
            ]
        )
    iti_mode = protocol.get("iti_stimulus_mode_str")
    if detail != "names" and iti_mode:
        lines.append(f'  ITI0["ITI mode: {_escape_label(iti_mode)}"]')
        lines.append("  P --> ITI0")

    steps = protocol.get("steps") or []
    if not isinstance(steps, list):
        steps = []
    previous = "P"
    for idx, raw_step in enumerate(steps):
        step = raw_step if isinstance(raw_step, Mapping) else {"name": raw_step}
        node = f"S{idx + 1}"
        lines.append(f'  {node}["{_step_label(idx, step, detail=detail)}"]')
        lines.append(f"  {previous} --> {node}")
        iti = _format_seconds(step.get("post_stimulus_iti_seconds"))
        if detail != "names" and iti and iti != "0s":
            iti_node = f"ITI{idx + 1}"
            lines.append(f'  {iti_node}["Post-stimulus ITI: {_escape_label(iti)}"]')
            lines.append(f"  {node} --> {iti_node}")
            previous = iti_node
        else:
            previous = node
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Palette Zarr archive.")
    parser.add_argument("--stimulus-run", help="analysis/stimulus_runs run name (default: latest).")
    parser.add_argument(
        "--detail",
        choices=("full", "names", "moving-grating-orientation"),
        default="full",
        help="Mermaid node detail level (default: full).",
    )
    parser.add_argument("--fenced", action="store_true", help="Wrap output in a ```mermaid fence.")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    protocol, run_name, source_label = load_protocol(args.zarr_path, args.stimulus_run)
    mermaid = protocol_to_mermaid(
        protocol,
        source_label=source_label,
        stimulus_run=run_name,
        detail=args.detail,
    )
    if args.fenced:
        print("```mermaid")
        print(mermaid)
        print("```")
    else:
        print(mermaid)


if __name__ == "__main__":
    main()
