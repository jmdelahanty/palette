#!/usr/bin/env python3
"""
Low-level YOLO ONNX exporter with built-in NMS.

Use this module when you want a direct/manual ONNX export from weights.
For run-aware export workflows (manifest/report/imgsz resolution, TRT export,
and optional registry updates), prefer:

    python -m fisheye.training.export_detection ...
"""

import argparse
import platform
import time
import traceback
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
from rich.console import Console


def _load_export_dependencies():
    try:
        import onnx
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Missing required dependencies for ONNX export (onnx/torch/ultralytics)."
        ) from exc

    try:
        from models.common import PostDetect, optim
    except ImportError as exc:
        raise RuntimeError(
            "Could not import 'PostDetect' and 'optim' from models.common. "
            "Ensure mechanic models/common.py is importable."
        ) from exc

    try:
        import onnxsim
    except ImportError:
        onnxsim = None

    return onnx, torch, YOLO, PostDetect, optim, onnxsim


def _load_training_report(run_dir: Path) -> Optional[Dict[str, Any]]:
    candidates = sorted(
        run_dir.glob("*_detection_training_report.yaml"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    text = candidates[0].read_text(encoding="utf-8")
    try:
        return yaml.safe_load(text) or None
    except Exception:
        # Legacy reports may contain python/object tags from yaml.dump.
        try:
            return yaml.load(text, Loader=yaml.BaseLoader) or None
        except Exception:
            return None


def _normalize_imgsz(value: Any) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        # Accept scalar or simple "h,w" / "[h,w]" string forms.
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1].strip()
        if "," in text:
            parts = [p.strip() for p in text.split(",") if p.strip()]
            if len(parts) == 1:
                try:
                    size = int(parts[0])
                except ValueError:
                    return None
                return (size, size) if size > 0 else None
            if len(parts) >= 2:
                try:
                    h, w = int(parts[0]), int(parts[1])
                except ValueError:
                    return None
                return (h, w) if h > 0 and w > 0 else None
            return None
        try:
            size = int(text)
        except ValueError:
            return None
        return (size, size) if size > 0 else None
    if isinstance(value, (int, float)):
        size = int(value)
        if size > 0:
            return size, size
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        if len(value) == 1:
            size = int(value[0])
            if size > 0:
                return size, size
            return None
        h, w = int(value[0]), int(value[1])
        if h > 0 and w > 0:
            return h, w
        return None
    return None


def _resolve_weights_path(weights_arg: Optional[str], run_dir: Optional[Path]) -> Optional[Path]:
    if weights_arg:
        return Path(weights_arg)
    if not run_dir:
        return None
    best = run_dir / "weights" / "best.pt"
    if best.exists():
        return best
    last = run_dir / "weights" / "last.pt"
    if last.exists():
        return last
    return None


def _resolve_input_shape(
    input_shape_arg: Optional[list[int]],
    imgsz_arg: Optional[list[int]],
    run_dir: Optional[Path],
) -> tuple[list[int], str]:
    if input_shape_arg is not None:
        if len(input_shape_arg) != 4:
            raise ValueError("--input-shape must be exactly 4 integers: B C H W")
        shape = [int(v) for v in input_shape_arg]
        if any(v <= 0 for v in shape):
            raise ValueError("--input-shape values must be positive")
        return shape, "cli:input_shape"

    imgsz = _normalize_imgsz(imgsz_arg)
    if imgsz is not None:
        return [1, 3, int(imgsz[0]), int(imgsz[1])], "cli:imgsz"

    if run_dir is not None:
        report = _load_training_report(run_dir)
        if isinstance(report, dict):
            history = report.get("training_history") if isinstance(report.get("training_history"), dict) else {}
            effective_training_params = (
                history.get("effective_training_params")
                if isinstance(history.get("effective_training_params"), dict)
                else {}
            )
            candidates = (
                ("training_report:effective_imgsz", history.get("effective_imgsz")),
                ("training_report:effective_training_params.imgsz", effective_training_params.get("imgsz")),
                ("training_report:training_params.imgsz", report.get("training_params", {}).get("imgsz") if isinstance(report.get("training_params"), dict) else None),
            )
            for source, candidate in candidates:
                resolved = _normalize_imgsz(candidate)
                if resolved is not None:
                    return [1, 3, int(resolved[0]), int(resolved[1])], source

    return [1, 3, 640, 640], "default:640"


def _build_metadata_props(
    *,
    run_id: Optional[str],
    set_id: Optional[str],
    manifest_sha256: Optional[str],
    system_hostname: Optional[str] = None,
    torch_version: Optional[str] = None,
    cuda_version: Optional[str] = None,
    exported_at_utc: Optional[str] = None,
) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    if run_id:
        metadata["run_id"] = str(run_id).strip()
    if set_id:
        metadata["set_id"] = str(set_id).strip()
    if manifest_sha256:
        metadata["manifest_sha256"] = str(manifest_sha256).strip().lower()
    if system_hostname:
        metadata["system_hostname"] = str(system_hostname).strip()
    if torch_version:
        metadata["torch_version"] = str(torch_version).strip()
    if cuda_version:
        metadata["cuda_version"] = str(cuda_version).strip()
    metadata["exported_at_utc"] = exported_at_utc or time.strftime(
        "%Y-%m-%dT%H:%M:%SZ",
        time.gmtime(),
    )
    return {k: v for k, v in metadata.items() if v}


def _apply_metadata_props(onnx_model, metadata: Dict[str, str]) -> None:
    if not metadata:
        return
    existing = {item.key: item for item in onnx_model.metadata_props}
    for key, value in metadata.items():
        if key in existing:
            existing[key].value = value
            continue
        entry = onnx_model.metadata_props.add()
        entry.key = key
        entry.value = value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Low-level ONNX export from weights or run_dir.",
        epilog=(
            "Note: this command does NOT update the training registry. "
            "Use `python -m fisheye.training.export_detection` for run-aware "
            "exports with optional registry logging."
        ),
    )
    parser.add_argument('-w', '--weights', type=str, help='PyTorch YOLO weights file (.pt)')
    parser.add_argument(
        '--run-dir',
        type=str,
        help='Optional training run directory (used to auto-resolve weights and imgsz).',
    )
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for the NMS operation')
    parser.add_argument('--conf-thres', type=float, default=0.8, help='Confidence threshold for the NMS operation')
    parser.add_argument('--topk', type=int, default=1, help='Max number of detections to return')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    parser.add_argument('--sim', action='store_true', help='Simplify the ONNX model')
    parser.add_argument('--input-shape', nargs='+', type=int, default=None, help='Model input shape [batch, channels, height, width]')
    parser.add_argument('--imgsz', nargs='+', type=int, help='Export image size override (single value or H W).')
    parser.add_argument('--device', type=str, default='cpu', help='Export device (e.g., "cpu" or "cuda:0")')
    parser.add_argument('--output-path', type=str, help='Custom path for the output ONNX file')
    parser.add_argument('--meta-run-id', type=str, default=None, help='Optional run_id to embed in ONNX metadata_props.')
    parser.add_argument('--meta-set-id', type=str, default=None, help='Optional set_id to embed in ONNX metadata_props.')
    parser.add_argument(
        '--meta-manifest-sha256',
        type=str,
        default=None,
        help='Optional manifest SHA256 to embed in ONNX metadata_props.',
    )
    args = parser.parse_args()

    console = Console()
    console.rule("[bold green]Advanced YOLO ONNX Exporter[/bold green]")

    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    if run_dir is not None and not run_dir.exists():
        console.print(f"[bold red]Error:[/bold red] run_dir not found: {run_dir}")
        return 2

    weights_path = _resolve_weights_path(args.weights, run_dir)
    if not weights_path:
        console.print("[bold red]Error:[/bold red] could not resolve weights path. Provide --weights or --run-dir.")
        return 2
    weights_path = weights_path.resolve()
    if not weights_path.exists():
        console.print(f"[bold red]Error:[/bold red] Weights file not found: {weights_path}")
        return 2

    try:
        input_shape, input_shape_source = _resolve_input_shape(args.input_shape, args.imgsz, run_dir)
    except ValueError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        return 2

    save_path = args.output_path or str(weights_path.with_suffix('.onnx'))
    console.print(f"[dim]weights:[/dim] {weights_path}")
    console.print(f"[dim]input_shape:[/dim] {input_shape} ({input_shape_source})")

    try:
        onnx, torch, YOLO, PostDetect, optim, onnxsim = _load_export_dependencies()
    except RuntimeError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        return 2

    inferred_run_id = args.meta_run_id or (run_dir.name if run_dir else None)
    metadata_props = _build_metadata_props(
        run_id=inferred_run_id,
        set_id=args.meta_set_id,
        manifest_sha256=args.meta_manifest_sha256,
        system_hostname=platform.node() or None,
        torch_version=str(torch.__version__),
        cuda_version=str(torch.version.cuda) if torch.version.cuda else None,
    )

    # --- Set NMS parameters on the custom PostDetect module ---
    PostDetect.conf_thres = args.conf_thres
    PostDetect.iou_thres = args.iou_thres
    PostDetect.topk = args.topk

    try:
        console.print(f"📦 Loading YOLO model from [cyan]{weights_path}[/cyan]...")
        YOLOv8 = YOLO(str(weights_path))
        model = YOLOv8.model.fuse().eval()

        # Apply optimizations and move to device
        for m in model.modules():
            optim(m)
            m.to(args.device)
        model.to(args.device)

        fake_input = torch.randn(input_shape).to(args.device)
        
        console.print("Warming up the model...")
        for _ in range(2):
            model(fake_input)
            
        console.print(f"Exporting to ONNX with NMS...")
        with BytesIO() as f:
            torch.onnx.export(
                model,
                fake_input,
                f,
                opset_version=args.opset,
                input_names=['images'],
                output_names=['num_dets', 'bboxes', 'scores', 'labels']
            )
            f.seek(0)
            onnx_model = onnx.load(f)

        onnx.checker.check_model(onnx_model)
        console.print("ONNX model validation passed.")

        b = input_shape[0]
        shapes = [b, 1, b, args.topk, 4, b, args.topk, b, args.topk]
        for i in onnx_model.graph.output:
            for j in i.type.tensor_type.shape.dim:
                # This sets the dynamic shape parameters in the ONNX file
                j.dim_param = str(shapes.pop(0))

        if args.sim and onnxsim:
            console.print("Simplifying ONNX model...")
            try:
                onnx_model, check = onnxsim.simplify(onnx_model)
                assert check, 'ONNX simplification check failed'
                console.print("Model simplified successfully.")
            except Exception as e:
                console.print(f"[bold yellow]Simplifier failure:[/bold yellow] {e}")

        _apply_metadata_props(onnx_model, metadata_props)
        onnx.save(onnx_model, save_path)
        file_size = Path(save_path).stat().st_size / (1024 * 1024)
        
        console.print(f"\n[bold green]ONNX export successful![/bold green]")
        console.print(f"   Saved as: [cyan]{save_path}[/cyan]")
        console.print(f"   File size: {file_size:.2f} MB")
        if metadata_props:
            metadata_line = ", ".join(f"{k}={v}" for k, v in metadata_props.items())
            console.print(f"   Metadata props: [dim]{metadata_line}[/dim]")
        
        console.print(
            "\n💡 [bold]Next step:[/bold] Convert to TensorRT using "
            "[cyan]python -m fisheye.training.onnx_to_tensorrt[/cyan].\n"
            "[dim]For run-aware export + optional registry logging, use "
            "`python -m fisheye.training.export_detection`.[/dim]"
        )

    except Exception as e:
        console.print(f"[bold red]Export failed:[/bold red]")
        console.print(traceback.format_exc())
        return 1

    return 0

if __name__ == '__main__':
    raise SystemExit(main())
