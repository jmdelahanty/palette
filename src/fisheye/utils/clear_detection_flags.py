"""Remove detection-issue flags for specific recordings/frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union


def _read_file_list(path: Path) -> List[Path]:
    if not path.exists():
        raise FileNotFoundError(path)
    items: List[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        items.append(Path(value))
    return items


def _parse_frames_spec(value: Optional[str]) -> Union[None, List[int], Dict[str, List[int]]]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    path = Path(text)
    if path.exists():
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except Exception:
            data = None
        if isinstance(data, dict):
            parsed: Dict[str, List[int]] = {}
            for key, val in data.items():
                frames = _coerce_frames(val)
                if frames:
                    parsed[str(key)] = frames
            return parsed
        if isinstance(data, list):
            return _coerce_frames(data)
        return _coerce_frames(raw.split())
    return _coerce_frames(text.replace(",", " ").split())


def _coerce_frames(value: object) -> List[int]:
    frames: List[int] = []
    if isinstance(value, (list, tuple)):
        items = value
    else:
        items = [value]
    for item in items:
        if isinstance(item, (int, float)):
            frames.append(int(item))
            continue
        token = str(item).strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            if len(parts) == 2:
                try:
                    start = int(parts[0].strip())
                    end = int(parts[1].strip())
                except ValueError:
                    continue
                if end < start:
                    start, end = end, start
                frames.extend(list(range(start, end + 1)))
                continue
        try:
            frames.append(int(token))
        except ValueError:
            continue
    return sorted(set(frames))


def _load_detection_frame_flags(path: Path) -> Dict[str, list]:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"Failed to load frame flags from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Frame flag file must contain a JSON object: {path}")
    return {str(k): v for k, v in data.items()}


def _normalize_key_map(data: Dict[str, list]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for key in data.keys():
        try:
            resolved = str(Path(key).resolve())
        except Exception:
            resolved = str(Path(key))
        mapping.setdefault(resolved, []).append(key)
    return mapping


def _remove_frames_from_entry(entry: list, frames: Sequence[int]) -> Tuple[list, int]:
    if not entry:
        return entry, 0
    frames_set = set(int(f) for f in frames)
    removed = 0
    kept: list = []
    for item in entry:
        if isinstance(item, dict):
            frame_idx = item.get("frame_idx")
            if frame_idx is None:
                kept.append(item)
                continue
            if int(frame_idx) in frames_set:
                removed += 1
                continue
            kept.append(item)
        else:
            try:
                if int(item) in frames_set:
                    removed += 1
                else:
                    kept.append(item)
            except Exception:
                kept.append(item)
    return kept, removed


def _remove_from_flag_file(flag_file: Path, targets: List[str], dry_run: bool) -> int:
    if not flag_file.exists():
        return 0
    lines = [line.strip() for line in flag_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    before = len(lines)
    target_set = set(targets)
    kept = [line for line in lines if line not in target_set]
    removed = before - len(kept)
    if removed and not dry_run:
        flag_file.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return removed


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Clear detection-issue flags from retune_frame_flags.json and retune_flags.txt.",
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Zarr path(s) to clear.")
    parser.add_argument(
        "--file-list",
        type=Path,
        action="append",
        help="Text file with one zarr path per line (comments with # allowed).",
    )
    parser.add_argument(
        "--frame-flag-file",
        type=Path,
        default=Path("retune_frame_flags.json"),
        help="JSON file mapping zarr paths to detection frames.",
    )
    parser.add_argument(
        "--flag-file",
        type=Path,
        default=Path("retune_flags.txt"),
        help="Text file listing recordings flagged for detection retune.",
    )
    parser.add_argument(
        "--frames",
        type=str,
        help="Comma/space-separated frame indices or JSON/text list. JSON mapping of zarr->frames is supported.",
    )
    parser.add_argument(
        "--patch-refined",
        action="store_true",
        help="After clearing flags, re-patch refined keypoints for the specified frames.",
    )
    parser.add_argument(
        "--refined-run",
        type=str,
        help="Refined keypoints run to patch (default: latest).",
    )
    parser.add_argument(
        "--keypoints-run",
        type=str,
        help="Keypoints run to use when patching refined data (default: latest).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow patching even if source runs do not match.",
    )
    parser.add_argument(
        "--keep-flag-file",
        action="store_true",
        help="Do not remove entries from --flag-file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing files.",
    )

    args = parser.parse_args(argv)

    targets: List[Path] = []
    if args.file_list:
        for file_list in args.file_list:
            targets.extend(_read_file_list(file_list))
    targets.extend(args.paths)
    targets = [Path(t) for t in targets]
    if not targets:
        print("No recordings provided.")
        return 1

    frames_spec = _parse_frames_spec(args.frames)

    frame_flags = _load_detection_frame_flags(args.frame_flag_file)
    resolved_map = _normalize_key_map(frame_flags)

    removed_entries = 0
    removed_frames_total = 0
    targets_str = [str(t) for t in targets]

    for target in targets:
        target_str = str(target)
        try:
            target_resolved = str(target.resolve())
        except Exception:
            target_resolved = target_str
        keys = []
        if target_str in frame_flags:
            keys.append(target_str)
        if target_resolved in resolved_map:
            for key in resolved_map[target_resolved]:
                if key not in keys:
                    keys.append(key)

        if not keys:
            print(f"{target}: no frame flags found.")
            continue

        frames_for_target: Optional[List[int]] = None
        if isinstance(frames_spec, dict):
            frames_for_target = frames_spec.get(target_str) or frames_spec.get(target_resolved)
        elif isinstance(frames_spec, list):
            frames_for_target = frames_spec

        for key in keys:
            entry = frame_flags.get(key, [])
            if frames_for_target:
                kept, removed = _remove_frames_from_entry(entry, frames_for_target)
                if removed:
                    removed_frames_total += removed
                    if kept:
                        frame_flags[key] = kept
                    else:
                        frame_flags.pop(key, None)
                        removed_entries += 1
                    print(f"{target}: removed {removed} frame flags from {key}")
                else:
                    print(f"{target}: no matching frames in {key}")
            else:
                frame_flags.pop(key, None)
                removed_entries += 1
                print(f"{target}: removed all frame flags ({key})")

    if removed_entries or removed_frames_total:
        if args.dry_run:
            print("Dry run: no files modified.")
        else:
            args.frame_flag_file.write_text(
                json.dumps(frame_flags, indent=2, sort_keys=True),
                encoding="utf-8",
            )

    removed_paths = 0
    if not args.keep_flag_file:
        removed_paths = _remove_from_flag_file(args.flag_file, targets_str, args.dry_run)

    if args.patch_refined:
        if args.dry_run:
            print("Dry run: skipping refined keypoint patch.")
        else:
            try:
                from . import patch_keypoints_from_crops
            except Exception as exc:
                raise RuntimeError(f"Failed to import patch_keypoints_from_crops: {exc}") from exc
            for target in targets:
                frames_for_target: Optional[List[int]] = None
                if isinstance(frames_spec, dict):
                    frames_for_target = frames_spec.get(str(target)) or frames_spec.get(
                        str(Path(target).resolve())
                    )
                elif isinstance(frames_spec, list):
                    frames_for_target = frames_spec
                if not frames_for_target:
                    print(f"{target}: no frames specified for refined patch; skipping.")
                    continue
                frames_arg = ",".join(str(f) for f in frames_for_target)
                argv = [
                    str(target),
                    "--frames",
                    frames_arg,
                    "--refined-only",
                    "--apply",
                ]
                if args.refined_run:
                    argv.extend(["--refined-run", args.refined_run])
                if args.keypoints_run:
                    argv.extend(["--keypoints-run", args.keypoints_run])
                if args.force:
                    argv.append("--force")
                patch_keypoints_from_crops.main(argv)

    print(
        f"Done. Removed entries: {removed_entries}, removed frames: {removed_frames_total}, "
        f"removed paths from flag file: {removed_paths}."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
