#!/usr/bin/env python3
"""Generate a static HTML index for training data cards and plot artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Optional
from urllib.parse import quote


@dataclass(frozen=True)
class CardEntry:
    kind: str
    schema_name: str
    set_id: str
    dataset_count: Optional[int]
    split: Optional[str]
    updated_utc: Optional[str]
    card_path: Path
    plot_dir: Path
    plot_paths: tuple[Path, ...]


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        text = value.decode("utf-8", "ignore").strip()
    else:
        text = str(value).strip()
    return text or None


def _parse_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _kind_for_card(schema_name: Optional[str], set_id: Optional[str]) -> str:
    schema = (schema_name or "").lower()
    set_name = (set_id or "").lower()
    if "detection_training_data_card" in schema or set_name.startswith("detect_"):
        return "detect"
    if "keypoint_training_data_card" in schema or set_name.startswith("pose_"):
        return "pose"
    if "eye_mask_training_data_card" in schema or set_name.startswith("eye_mask_"):
        return "eye_mask"
    return "other"


def _load_card_payload(card_path: Path) -> Optional[Mapping[str, Any]]:
    try:
        payload = json.loads(card_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    return payload


def collect_training_card_entries(datasets_root: Path) -> list[CardEntry]:
    entries: list[CardEntry] = []
    for card_path in sorted(datasets_root.rglob("*.data_card.json")):
        payload = _load_card_payload(card_path)
        if payload is None:
            continue
        set_id = _normalize_text(payload.get("set_id")) or card_path.stem
        schema_name = _normalize_text(payload.get("schema_name")) or "unknown"
        selection = payload.get("selection")
        dataset_count: Optional[int] = None
        split: Optional[str] = None
        if isinstance(selection, Mapping):
            dataset_count = _parse_optional_int(selection.get("dataset_count"))
            split = _normalize_text(selection.get("split"))
        updated_utc = (
            _normalize_text(payload.get("updated_utc"))
            or _normalize_text(payload.get("generated_utc"))
            or _normalize_text(payload.get("created_utc"))
        )
        plot_dir = card_path.parent / f"{card_path.stem}.plots"
        plot_paths = tuple(sorted(plot_dir.glob("*.png"))) if plot_dir.is_dir() else tuple()
        entries.append(
            CardEntry(
                kind=_kind_for_card(schema_name=schema_name, set_id=set_id),
                schema_name=schema_name,
                set_id=set_id,
                dataset_count=dataset_count,
                split=split,
                updated_utc=updated_utc,
                card_path=card_path,
                plot_dir=plot_dir,
                plot_paths=plot_paths,
            )
        )
    return entries


def _href_for_path(path: Path, *, output_dir: Path) -> str:
    rel = os.path.relpath(str(path), str(output_dir))
    rel_posix = rel.replace(os.sep, "/")
    return quote(rel_posix, safe="/._-")


def _sort_key(entry: CardEntry) -> tuple[int, str]:
    order = {"detect": 0, "pose": 1, "eye_mask": 2, "other": 3}
    return (order.get(entry.kind, 99), entry.set_id.lower())


def render_training_card_index_html(
    *,
    entries: list[CardEntry],
    datasets_root: Path,
    output_html: Path,
    title: str,
    thumb_width: int,
) -> str:
    output_dir = output_html.parent
    sections = {
        "detect": "Detection",
        "pose": "Keypoint",
        "eye_mask": "Eye-Mask",
        "other": "Other",
    }
    entries_sorted = sorted(entries, key=_sort_key)
    counts = {key: 0 for key in sections}
    for entry in entries_sorted:
        counts[entry.kind] = counts.get(entry.kind, 0) + 1

    body: list[str] = []
    body.append("<!doctype html>")
    body.append("<html lang='en'>")
    body.append("<head>")
    body.append("  <meta charset='utf-8' />")
    body.append("  <meta name='viewport' content='width=device-width, initial-scale=1' />")
    body.append(f"  <title>{escape(title)}</title>")
    body.append("  <style>")
    body.append("    :root { --bg:#f5f7fb; --fg:#1f2937; --muted:#6b7280; --line:#d1d5db; --card:#ffffff; }")
    body.append("    body { margin:0; font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--fg); }")
    body.append("    main { max-width:1400px; margin:0 auto; padding:20px; }")
    body.append("    h1 { margin:0 0 8px; font-size:1.5rem; }")
    body.append("    .sub { color:var(--muted); margin:0 0 16px; font-size:0.95rem; }")
    body.append("    .toolbar { display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-bottom:16px; }")
    body.append("    .toolbar input { min-width:320px; padding:8px 10px; border:1px solid var(--line); border-radius:8px; background:#fff; }")
    body.append("    .pill { border:1px solid var(--line); border-radius:999px; padding:4px 10px; background:#fff; color:var(--muted); font-size:0.85rem; }")
    body.append("    .section { margin:18px 0 8px; font-size:1.1rem; }")
    body.append("    details.card { background:var(--card); border:1px solid var(--line); border-radius:10px; margin:10px 0; }")
    body.append("    details.card > summary { cursor:pointer; padding:10px 12px; font-weight:600; }")
    body.append("    details.card[open] > summary { border-bottom:1px solid var(--line); }")
    body.append("    .meta { padding:10px 12px; display:flex; gap:16px; flex-wrap:wrap; color:var(--muted); font-size:0.9rem; }")
    body.append("    .meta code { color:var(--fg); }")
    body.append("    .links { padding:0 12px 10px; display:flex; gap:14px; flex-wrap:wrap; }")
    body.append("    .plots { padding:0 12px 12px; display:flex; gap:10px; flex-wrap:wrap; }")
    body.append("    .plots a { border:1px solid var(--line); border-radius:8px; overflow:hidden; background:#fff; text-decoration:none; }")
    body.append("    .plots img { display:block; width:auto; height:auto; max-width:100%; }")
    body.append("    .empty { padding:0 12px 12px; color:var(--muted); font-size:0.9rem; }")
    body.append("    .hidden { display:none !important; }")
    body.append("  </style>")
    body.append("</head>")
    body.append("<body>")
    body.append("<main>")
    body.append(f"  <h1>{escape(title)}</h1>")
    body.append(
        "  <p class='sub'>"
        f"datasets root: <code>{escape(str(datasets_root))}</code> | "
        f"cards: {len(entries_sorted)} | output: <code>{escape(str(output_html))}</code>"
        "</p>"
    )
    body.append("  <div class='toolbar'>")
    body.append("    <input id='q' type='search' placeholder='Filter by set_id, schema, path...' />")
    body.append(f"    <span class='pill'>detect: {counts.get('detect', 0)}</span>")
    body.append(f"    <span class='pill'>pose: {counts.get('pose', 0)}</span>")
    body.append(f"    <span class='pill'>eye_mask: {counts.get('eye_mask', 0)}</span>")
    body.append(f"    <span class='pill'>other: {counts.get('other', 0)}</span>")
    body.append("  </div>")

    for kind_key, section_label in sections.items():
        section_entries = [entry for entry in entries_sorted if entry.kind == kind_key]
        if not section_entries:
            continue
        body.append(f"  <h2 class='section'>{escape(section_label)} ({len(section_entries)})</h2>")
        for entry in section_entries:
            search_blob = " ".join(
                filter(
                    None,
                    [
                        entry.kind,
                        entry.schema_name,
                        entry.set_id,
                        str(entry.card_path),
                        str(entry.plot_dir),
                        entry.split or "",
                    ],
                )
            ).lower()
            card_href = _href_for_path(entry.card_path, output_dir=output_dir)
            dataset_count_text = str(entry.dataset_count) if entry.dataset_count is not None else "-"
            split_text = entry.split or "-"
            updated_text = entry.updated_utc or "-"

            body.append(
                f"  <details class='card' open data-search='{escape(search_blob)}'>"
                f"<summary>[{escape(entry.kind)}] <code>{escape(entry.set_id)}</code> "
                f"| datasets={escape(dataset_count_text)} | plots={len(entry.plot_paths)}</summary>"
            )
            body.append("    <div class='meta'>")
            body.append(f"      <span>schema: <code>{escape(entry.schema_name)}</code></span>")
            body.append(f"      <span>split: <code>{escape(split_text)}</code></span>")
            body.append(f"      <span>updated: <code>{escape(updated_text)}</code></span>")
            body.append(f"      <span>card: <code>{escape(str(entry.card_path))}</code></span>")
            body.append("    </div>")
            body.append("    <div class='links'>")
            body.append(f"      <a href='{card_href}' target='_blank' rel='noopener'>Open data_card.json</a>")
            if entry.plot_dir.exists():
                plot_dir_href = _href_for_path(entry.plot_dir, output_dir=output_dir)
                body.append(
                    f"      <a href='{plot_dir_href}' target='_blank' rel='noopener'>Open plots directory</a>"
                )
            body.append("    </div>")
            if entry.plot_paths:
                body.append("    <div class='plots'>")
                for plot_path in entry.plot_paths:
                    plot_href = _href_for_path(plot_path, output_dir=output_dir)
                    plot_name = plot_path.name
                    body.append(
                        f"      <a href='{plot_href}' target='_blank' rel='noopener' title='{escape(plot_name)}'>"
                        f"<img src='{plot_href}' alt='{escape(plot_name)}' loading='lazy' width='{int(thumb_width)}' /></a>"
                    )
                body.append("    </div>")
            else:
                body.append("    <div class='empty'>No plot PNGs found for this card.</div>")
            body.append("  </details>")

    body.append("</main>")
    body.append("<script>")
    body.append("  const q = document.getElementById('q');")
    body.append("  const cards = Array.from(document.querySelectorAll('details.card'));")
    body.append("  q.addEventListener('input', () => {")
    body.append("    const needle = q.value.toLowerCase().trim();")
    body.append("    for (const card of cards) {")
    body.append("      const hay = card.dataset.search || '';")
    body.append("      const show = needle === '' || hay.includes(needle);")
    body.append("      card.classList.toggle('hidden', !show);")
    body.append("    }")
    body.append("  });")
    body.append("</script>")
    body.append("</body>")
    body.append("</html>")

    return "\n".join(body) + "\n"


def _default_output_html(datasets_root: Path) -> Path:
    return datasets_root / "_index" / "training_data_cards_index.html"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path("/nvme1/training/datasets"),
        help="Root directory that contains training dataset folders.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=None,
        help="Output HTML path (default: <datasets-root>/_index/training_data_cards_index.html).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Training Data Cards Index",
        help="HTML page title.",
    )
    parser.add_argument(
        "--thumb-width",
        type=int,
        default=320,
        help="Thumbnail width in pixels for plot previews (default: 320).",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated HTML file with xdg-open.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    datasets_root = Path(args.datasets_root)
    if not datasets_root.is_dir():
        print(f"Training data-card index failed: datasets root not found: {datasets_root}")
        return 1

    output_html = Path(args.output_html) if args.output_html is not None else _default_output_html(datasets_root)
    if int(args.thumb_width) <= 0:
        parser.error("--thumb-width must be > 0.")

    entries = collect_training_card_entries(datasets_root=datasets_root)
    html = render_training_card_index_html(
        entries=entries,
        datasets_root=datasets_root,
        output_html=output_html,
        title=str(args.title),
        thumb_width=int(args.thumb_width),
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")

    print(
        "Training data-card index: "
        f"cards={len(entries)} output={output_html} "
        f"datasets_root={datasets_root}"
    )
    if args.open:
        try:
            subprocess.run(["xdg-open", str(output_html)], check=False)
        except Exception as exc:
            print(f"Warning: failed to open HTML with xdg-open: {exc}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
