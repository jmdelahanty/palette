#!/usr/bin/env python3
"""Watch a noisy training log with progress-bar updates collapsed."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
EPOCH_PROGRESS_RE = re.compile(r"^\s*(\d+/\d+)\s+(\S+)\s+")
VALIDATION_ALL_RE = re.compile(r"^\s*all\s+\d+\s+\d+\s+")
KEEP_RE = re.compile(
    r"("
    r"AMP:|"
    r"EarlyStopping|"
    r"Export|"
    r"Image sizes|"
    r"ONNX|"
    r"Optimizer|"
    r"Results saved|"
    r"Starting training|"
    r"TensorRT|"
    r"Training complete|"
    r"Transferred|"
    r"Validating .*best\.pt|"
    r"YOLO.*summary|"
    r"best\.pt|"
    r"engine|"
    r"epochs completed|"
    r"last\.pt|"
    r"optimizer:"
    r")",
    re.IGNORECASE,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Periodically print a compact view of a training log. "
            "Ultralytics progress-bar carriage returns are collapsed into one line per epoch."
        )
    )
    parser.add_argument("log_path", type=Path, help="Path to train.log.")
    parser.add_argument(
        "--exit-code",
        type=Path,
        default=None,
        help="Optional exit_code.txt path. Defaults to <train.log parent>/exit_code.txt.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Refresh interval in seconds. Default: 10.",
    )
    parser.add_argument(
        "--lines",
        type=int,
        default=30,
        help="Number of filtered lines to display. Default: 30.",
    )
    parser.add_argument(
        "--read-bytes",
        type=int,
        default=2_000_000,
        help="Bytes to read from the end of the log on each refresh. Default: 2000000.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print one compact view and exit.",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not clear the terminal before each refresh.",
    )
    return parser


def _read_tail(path: Path, max_bytes: int) -> str:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        start = max(0, size - max_bytes)
        handle.seek(start)
        data = handle.read()
    return data.decode("utf-8", errors="replace")


def _clean_line(line: str) -> str:
    return ANSI_RE.sub("", line).strip()


def _format_epoch_progress(line: str) -> Optional[tuple[str, str]]:
    match = EPOCH_PROGRESS_RE.match(line)
    if not match:
        return None

    prefix, _, progress_part = line.partition(":")
    tokens = prefix.split()
    if len(tokens) < 7:
        return None

    epoch, gpu_mem, box_loss, cls_loss, dfl_loss, instances, size = tokens[:7]
    progress_tokens: list[str] = []
    percent = re.search(r"(\d+)%", progress_part)
    batch = re.search(r"(\d+/\d+)", progress_part)
    if batch:
        progress_tokens.append(f"batches={batch.group(1)}")
    if percent:
        progress_tokens.append(f"progress={percent.group(1)}%")

    suffix = f" {' '.join(progress_tokens)}" if progress_tokens else ""
    text = (
        f"epoch {epoch}{suffix} gpu={gpu_mem} "
        f"box={box_loss} cls={cls_loss} dfl={dfl_loss} "
        f"instances={instances} size={size}"
    )
    return epoch, text


def _format_validation(line: str) -> Optional[str]:
    if not VALIDATION_ALL_RE.match(line):
        return None
    tokens = line.split()
    if len(tokens) < 7:
        return line
    _, images, instances, precision, recall, map50, map5095 = tokens[:7]
    return (
        "val all "
        f"images={images} instances={instances} "
        f"P={precision} R={recall} mAP50={map50} mAP50-95={map5095}"
    )


def _filtered_events(text: str) -> list[str]:
    events: list[str] = []
    replace_index: dict[str, int] = {}

    for raw_line in text.replace("\r", "\n").splitlines():
        line = _clean_line(raw_line)
        if not line:
            continue
        if line.startswith("Class ") or " Class     Images  Instances" in line:
            continue
        validation = _format_validation(line)
        if validation is not None:
            events.append(validation)
            continue
        epoch_progress = _format_epoch_progress(line)
        if epoch_progress is not None:
            epoch, formatted = epoch_progress
            key = f"epoch:{epoch}"
            if key in replace_index:
                events[replace_index[key]] = formatted
            else:
                replace_index[key] = len(events)
                events.append(formatted)
            continue
        if KEEP_RE.search(line):
            events.append(line)

    return events


def _status_text(exit_code_path: Path) -> str:
    try:
        text = exit_code_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return "running"
    except OSError as exc:
        return f"exit_code_unreadable: {exc}"
    return f"exit_code={text or '<empty>'}"


def _print_view(
    *,
    log_path: Path,
    exit_code_path: Path,
    events: list[str],
    line_count: int,
) -> None:
    width = shutil.get_terminal_size((120, 24)).columns
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = _status_text(exit_code_path)
    try:
        size = log_path.stat().st_size
    except OSError:
        size = 0

    print(f"{timestamp} | {status} | log_bytes={size}")
    print(f"log: {log_path}")
    print(f"exit: {exit_code_path}")
    print("-" * min(width, 120))
    if events:
        for line in events[-line_count:]:
            print(line)
    else:
        print("No matching training summary lines yet.")


def _run_once(args: argparse.Namespace) -> int:
    log_path = args.log_path.expanduser()
    exit_code_path = (
        args.exit_code.expanduser()
        if args.exit_code is not None
        else log_path.parent / "exit_code.txt"
    )
    try:
        text = _read_tail(log_path, max(1, args.read_bytes))
    except FileNotFoundError:
        print(f"log not found: {log_path}", file=sys.stderr)
        return 2
    events = _filtered_events(text)
    _print_view(
        log_path=log_path,
        exit_code_path=exit_code_path,
        events=events,
        line_count=max(1, args.lines),
    )
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    while True:
        if not args.no_clear and sys.stdout.isatty():
            print("\033[2J\033[H", end="")
        code = _run_once(args)
        if args.once or code:
            return code
        try:
            time.sleep(max(0.1, args.interval))
        except KeyboardInterrupt:
            print()
            return 130


if __name__ == "__main__":
    raise SystemExit(main())
