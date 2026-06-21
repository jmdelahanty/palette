"""Compare dense, bitpacked, and component-RLE mask storage on sampled rows."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Optional, Sequence

import numpy as np
import zarr

from fisheye.shared.mask_rle import decode_mask_component_stack_rle, encode_mask_component_stack_rle
from fisheye.shared.mask_store import write_encoded_component_rle_mask_store


@dataclass(frozen=True)
class EncodingBenchmark:
    archive: str
    run_path: str
    source_array: str
    encoding: str
    status: str
    sampled_rows: int
    total_rows: int
    shape: tuple[int, ...]
    logical_bytes: int
    stored_bytes: int | None
    encode_seconds: float | None
    write_seconds: float | None
    decode_seconds: float | None
    encode_rows_per_second: float | None
    write_rows_per_second: float | None
    decode_rows_per_second: float | None
    temp_zarr_path: str | None
    notes: tuple[str, ...]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "-"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size:.1f} TiB"


def _array_stored_bytes(array: Any) -> int | None:
    stored = getattr(array, "nbytes_stored", None)
    if callable(stored):
        try:
            stored = stored()
        except TypeError:
            stored = None
    if stored is None:
        return None
    try:
        return int(stored)
    except Exception:
        return None


def _group_stored_bytes(group: Any) -> int:
    total = 0
    for key in group.array_keys():
        stored = _array_stored_bytes(group[key])
        if stored is not None:
            total += int(stored)
    for key in group.group_keys():
        total += _group_stored_bytes(group[key])
    return int(total)


def _normalize_shape(shape: Sequence[int]) -> tuple[int, int, int, int]:
    dims = tuple(int(value) for value in shape)
    if len(dims) == 3:
        rows, height, width = dims
        return (rows, 1, height, width)
    if len(dims) == 4:
        rows, channels, height, width = dims
        return (rows, channels, height, width)
    raise ValueError(f"Expected mask array shape (N,H,W) or (N,C,H,W), got {dims!r}.")


def _component_names(run_group: Any, channel_count: int) -> tuple[str, ...]:
    attrs = dict(getattr(run_group, "attrs", {}) or {})
    for key in ("mask_labels", "component_names", "labels"):
        value = attrs.get(key)
        if isinstance(value, (list, tuple)) and len(value) == int(channel_count):
            return tuple(str(item) for item in value)
    return tuple(f"component_{idx}" for idx in range(int(channel_count)))


def _resolve_run(root: Any, family: str, run_name: str) -> tuple[str, Any]:
    if family not in root:
        raise ValueError(f"{family} not found in archive.")
    parent = root[family]
    resolved = str(run_name)
    if resolved == "latest":
        resolved = str(parent.attrs.get("latest") or "")
        if not resolved:
            names = sorted(str(name) for name in parent.group_keys())
            if not names:
                raise ValueError(f"{family} has no runs.")
            resolved = names[-1]
    if resolved not in parent:
        raise ValueError(f"{family}/{resolved} not found.")
    return resolved, parent[resolved]


def _parse_sample_rows(value: str, total_rows: int) -> np.ndarray:
    text = str(value).strip().lower()
    if text == "all":
        return np.arange(int(total_rows), dtype=np.int64)
    if "." in text:
        fraction = float(text)
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"Sample fraction must be in (0, 1], got {value!r}.")
        count = max(1, int(math.ceil(float(total_rows) * fraction)))
    else:
        count = int(text)
        if count <= 0:
            raise ValueError(f"Sample row count must be positive, got {value!r}.")
    if count >= int(total_rows):
        return np.arange(int(total_rows), dtype=np.int64)
    return np.linspace(0, int(total_rows) - 1, num=count, dtype=np.int64)


def _read_sample(array: Any, rows: np.ndarray) -> np.ndarray:
    if rows.size == 0:
        return np.zeros((0,), dtype=np.uint8)
    if np.all(np.diff(rows) == 1):
        sample = np.asarray(array[int(rows[0]) : int(rows[-1]) + 1], dtype=np.uint8)
    else:
        sample = np.stack([np.asarray(array[int(row)], dtype=np.uint8) for row in rows], axis=0)
    if sample.ndim == 3:
        sample = sample[:, None, :, :]
    return np.asarray(sample > 0, dtype=np.uint8)


def _safe_name(*parts: str) -> str:
    return "__".join(str(part).replace("/", "_").replace(" ", "_") for part in parts if str(part))


def _create_temp_group(temp_root: Path, *, archive: Path, run_path: str, encoding: str) -> tuple[Path, Any]:
    slug = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = _safe_name(archive.stem or "archive", run_path, encoding, slug)
    target = Path(tempfile.mkdtemp(prefix=f"{name}__", suffix=".zarr", dir=str(temp_root)))
    return target, zarr.open_group(str(target), mode="w")


def _benchmark_dense(
    sample: np.ndarray,
    *,
    temp_root: Path,
    archive: Path,
    run_path: str,
    delete_temp: bool,
) -> tuple[int, float, int | None, str | None]:
    target, root = _create_temp_group(temp_root, archive=archive, run_path=run_path, encoding="dense")
    chunks = (min(max(1, int(sample.shape[0])), 16), 1, int(sample.shape[2]), int(sample.shape[3]))
    start = time.perf_counter()
    root.create_array("masks_roi", data=sample, chunks=chunks, overwrite=True)
    seconds = float(time.perf_counter() - start)
    stored = _group_stored_bytes(root)
    temp_path = str(target)
    if delete_temp:
        shutil.rmtree(target, ignore_errors=True)
        temp_path = None
    return int(sample.nbytes), seconds, stored, temp_path


def _pack_sample(sample: np.ndarray) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    packed = np.packbits(np.asarray(sample > 0, dtype=np.uint8), axis=-1, bitorder="little")
    return packed, float(time.perf_counter() - start)


def _unpack_sample(packed: np.ndarray, *, width: int) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    unpacked = np.unpackbits(packed, axis=-1, count=int(width), bitorder="little").astype(np.uint8, copy=False)
    return unpacked, float(time.perf_counter() - start)


def _benchmark_bitpacked(
    sample: np.ndarray,
    *,
    temp_root: Path,
    archive: Path,
    run_path: str,
    component_names: Sequence[str],
    delete_temp: bool,
) -> tuple[int, float, float, float, int | None, str | None]:
    packed, encode_seconds = _pack_sample(sample)
    _decoded, decode_seconds = _unpack_sample(packed, width=int(sample.shape[-1]))
    target, root = _create_temp_group(temp_root, archive=archive, run_path=run_path, encoding="bitpacked")
    root.attrs.update(
        {
            "mask_storage_encoding": "bitpacked_binary_v1_probe",
            "logical_shape": [int(value) for value in sample.shape],
            "component_names": [str(value) for value in component_names],
            "packed_axis": "width",
            "packed_bitorder": "little",
            "packed_width_bytes": int(packed.shape[-1]),
        }
    )
    chunks = (min(max(1, int(packed.shape[0])), 16), 1, int(packed.shape[2]), int(packed.shape[3]))
    start = time.perf_counter()
    root.create_array("masks_packed", data=packed, chunks=chunks, overwrite=True)
    write_seconds = float(time.perf_counter() - start)
    stored = _group_stored_bytes(root)
    temp_path = str(target)
    if delete_temp:
        shutil.rmtree(target, ignore_errors=True)
        temp_path = None
    return int(packed.nbytes), encode_seconds, write_seconds, decode_seconds, stored, temp_path


def _benchmark_rle(
    sample: np.ndarray,
    *,
    temp_root: Path,
    archive: Path,
    run_path: str,
    component_names: Sequence[str],
    delete_temp: bool,
) -> tuple[int, float, float, float, int | None, str | None]:
    start = time.perf_counter()
    encoded = encode_mask_component_stack_rle(sample, component_names=tuple(component_names))
    encode_seconds = float(time.perf_counter() - start)
    start = time.perf_counter()
    _decoded = decode_mask_component_stack_rle(encoded)
    decode_seconds = float(time.perf_counter() - start)
    target, root = _create_temp_group(temp_root, archive=archive, run_path=run_path, encoding="rle")
    start = time.perf_counter()
    write_encoded_component_rle_mask_store(
        root,
        encoded,
        overwrite=True,
        extra_attrs={"mask_storage_encoding": "component_rle_v1_probe"},
    )
    write_seconds = float(time.perf_counter() - start)
    stored = _group_stored_bytes(root)
    logical = 0
    for component in encoded.components:
        logical += int(
            component.counts.nbytes
            + component.indptr.nbytes
            + component.present.nbytes
            + component.area_px.nbytes
            + component.bbox_xyxy.nbytes
        )
    temp_path = str(target)
    if delete_temp:
        shutil.rmtree(target, ignore_errors=True)
        temp_path = None
    return int(logical), encode_seconds, write_seconds, decode_seconds, stored, temp_path


def run_benchmark(
    archive: Path,
    *,
    family: str,
    run_name: str,
    source_array: str,
    sample_rows: str,
    temp_root: Path,
    delete_temp: bool,
) -> list[EncodingBenchmark]:
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    resolved_run, run_group = _resolve_run(root, family, run_name)
    run_path = f"{family}/{resolved_run}"
    if source_array not in run_group:
        raise ValueError(f"{run_path}/{source_array} not found.")
    array = run_group[source_array]
    total_rows, channel_count, _height, _width = _normalize_shape(array.shape)
    rows = _parse_sample_rows(sample_rows, total_rows)
    sample = _read_sample(array, rows)
    component_names = _component_names(run_group, channel_count)

    dense_logical, dense_write, dense_stored, dense_path = _benchmark_dense(
        sample,
        temp_root=temp_root,
        archive=archive,
        run_path=run_path,
        delete_temp=delete_temp,
    )
    bit_logical, bit_encode, bit_write, bit_decode, bit_stored, bit_path = _benchmark_bitpacked(
        sample,
        temp_root=temp_root,
        archive=archive,
        run_path=run_path,
        component_names=component_names,
        delete_temp=delete_temp,
    )
    rle_logical, rle_encode, rle_write, rle_decode, rle_stored, rle_path = _benchmark_rle(
        sample,
        temp_root=temp_root,
        archive=archive,
        run_path=run_path,
        component_names=component_names,
        delete_temp=delete_temp,
    )
    common = {
        "archive": str(archive),
        "run_path": run_path,
        "source_array": source_array,
        "status": "ok",
        "sampled_rows": int(sample.shape[0]),
        "total_rows": int(total_rows),
        "shape": tuple(int(value) for value in sample.shape),
    }

    def rows_per_second(seconds: float | None) -> float | None:
        return float(sample.shape[0]) / float(seconds) if seconds and seconds > 0.0 else None

    return [
        EncodingBenchmark(
            **common,
            encoding="dense_uint8",
            logical_bytes=int(dense_logical),
            stored_bytes=dense_stored,
            encode_seconds=None,
            write_seconds=dense_write,
            decode_seconds=None,
            encode_rows_per_second=None,
            write_rows_per_second=rows_per_second(dense_write),
            decode_rows_per_second=None,
            temp_zarr_path=dense_path,
            notes=("temporary sampled dense store",),
        ),
        EncodingBenchmark(
            **common,
            encoding="bitpacked_binary_v1_probe",
            logical_bytes=int(bit_logical),
            stored_bytes=bit_stored,
            encode_seconds=bit_encode,
            write_seconds=bit_write,
            decode_seconds=bit_decode,
            encode_rows_per_second=rows_per_second(bit_encode),
            write_rows_per_second=rows_per_second(bit_write),
            decode_rows_per_second=rows_per_second(bit_decode),
            temp_zarr_path=bit_path,
            notes=("probe layout; not yet production contract",),
        ),
        EncodingBenchmark(
            **common,
            encoding="component_rle_v1",
            logical_bytes=int(rle_logical),
            stored_bytes=rle_stored,
            encode_seconds=rle_encode,
            write_seconds=rle_write,
            decode_seconds=rle_decode,
            encode_rows_per_second=rows_per_second(rle_encode),
            write_rows_per_second=rows_per_second(rle_write),
            decode_rows_per_second=rows_per_second(rle_decode),
            temp_zarr_path=rle_path,
            notes=("temporary sampled component RLE store",),
        ),
    ]


def render_markdown(results: Sequence[EncodingBenchmark]) -> str:
    lines = [
        "# Mask Storage Encoding Benchmark",
        "",
        "| encoding | status | sampled | shape | logical | stored | encode rows/s | write rows/s | decode rows/s |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        sampled = f"{result.sampled_rows}/{result.total_rows}"
        shape = "x".join(str(value) for value in result.shape)
        lines.append(
            "| "
            + " | ".join(
                [
                    result.encoding,
                    result.status,
                    sampled,
                    shape,
                    _format_bytes(result.logical_bytes),
                    _format_bytes(result.stored_bytes),
                    "-" if result.encode_rows_per_second is None else f"{result.encode_rows_per_second:.1f}",
                    "-" if result.write_rows_per_second is None else f"{result.write_rows_per_second:.1f}",
                    "-" if result.decode_rows_per_second is None else f"{result.decode_rows_per_second:.1f}",
                ]
            )
            + " |"
        )
    if results:
        dense = next((result for result in results if result.encoding == "dense_uint8"), None)
        if dense and dense.stored_bytes:
            lines.extend(["", "Storage ratios versus temporary dense sample:"])
            for result in results:
                if result.stored_bytes:
                    ratio = float(dense.stored_bytes) / float(result.stored_bytes)
                    lines.append(f"- `{result.encoding}`: {ratio:.2f}x")
    lines.append("")
    return "\n".join(lines)


def _write_json(path: Path, results: Sequence[EncodingBenchmark]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_id": "palette_mask_storage_encoding_benchmark_v1",
        "results": [_json_safe(asdict(result)) for result in results],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, results: Sequence[EncodingBenchmark]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(_json_safe(asdict(result)), sort_keys=True) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Analysis or training Zarr archive to inspect.")
    parser.add_argument("--family", default="refined_subject_masks_runs", help="Mask run parent group.")
    parser.add_argument("--run", default="latest", help="Run name or 'latest'.")
    parser.add_argument("--source-array", default="masks_roi", help="Dense binary mask array name.")
    parser.add_argument("--sample-rows", default="128", help="'all', an integer row count, or a fraction in (0,1].")
    parser.add_argument(
        "--tmp-root",
        type=Path,
        default=Path("/tmp/palette_mask_storage_benchmark"),
        help="Temporary output root.",
    )
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary Zarr stores after measuring them.")
    parser.add_argument("--json-report", type=Path, help="Optional JSON report path.")
    parser.add_argument("--jsonl-report", type=Path, help="Optional JSONL report path.")
    parser.add_argument("--markdown-report", type=Path, help="Optional markdown report path.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    archive = Path(args.archive)
    if not archive.exists():
        parser.error(f"Archive not found: {archive}")
    args.tmp_root.mkdir(parents=True, exist_ok=True)
    results = run_benchmark(
        archive,
        family=str(args.family),
        run_name=str(args.run),
        source_array=str(args.source_array),
        sample_rows=str(args.sample_rows),
        temp_root=Path(args.tmp_root),
        delete_temp=not bool(args.keep_temp),
    )
    if args.json_report:
        _write_json(Path(args.json_report), results)
    if args.jsonl_report:
        _write_jsonl(Path(args.jsonl_report), results)
    markdown = render_markdown(results)
    if args.markdown_report:
        Path(args.markdown_report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown_report).write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
