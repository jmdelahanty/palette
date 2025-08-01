#!/usr/bin/env python3
"""
Advanced YOLO ONNX Exporter with Built-in NMS
This creates an ONNX model with NMS post-processing included,
resulting in clean, four-tensor outputs ready for TensorRT.
"""

import argparse
from io import BytesIO
import onnx
import torch
from ultralytics import YOLO
from pathlib import Path
from rich.console import Console
try:
    from models.common import PostDetect, optim
except ImportError:
    print("\n" + "="*80)
    print("ERROR: Could not import 'PostDetect' and 'optim'.")
    print("Please make sure the 'models/common.py' file from the 'mechanic' project is in your path.")
    print("="*80 + "\n")
    exit(1)

try:
    import onnxsim
except ImportError:
    onnxsim = None

def main():
    parser = argparse.ArgumentParser(description="Export YOLO model with built-in NMS")
    parser.add_argument('-w', '--weights', type=str, required=True, help='PyTorch YOLOv8 weights file (.pt)')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for the NMS operation')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold for the NMS operation')
    parser.add_argument('--topk', type=int, default=100, help='Max number of detections to return')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    parser.add_argument('--sim', action='store_true', help='Simplify the ONNX model')
    parser.add_argument('--input-shape', nargs='+', type=int, default=[1, 3, 640, 640], help='Model input shape [batch, channels, height, width]')
    parser.add_argument('--device', type=str, default='cpu', help='Export device (e.g., "cpu" or "cuda:0")')
    parser.add_argument('--output-path', type=str, help='Custom path for the output ONNX file')
    args = parser.parse_args()

    console = Console()
    console.rule("[bold green]Advanced YOLO ONNX Exporter[/bold green]")

    # --- Set NMS parameters on the custom PostDetect module ---
    PostDetect.conf_thres = args.conf_thres
    PostDetect.iou_thres = args.iou_thres
    PostDetect.topk = args.topk
    
    weights_path = Path(args.weights)
    if not weights_path.exists():
        console.print(f"[bold red]Error:[/bold red] Weights file not found: {weights_path}")
        return

    save_path = args.output_path or str(weights_path.with_suffix('.onnx'))
    
    try:
        console.print(f"📦 Loading YOLO model from [cyan]{weights_path}[/cyan]...")
        YOLOv8 = YOLO(args.weights)
        model = YOLOv8.model.fuse().eval()

        # Apply optimizations and move to device
        for m in model.modules():
            optim(m)
            m.to(args.device)
        model.to(args.device)

        fake_input = torch.randn(args.input_shape).to(args.device)
        
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

        b = args.input_shape[0]
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

        onnx.save(onnx_model, save_path)
        file_size = Path(save_path).stat().st_size / (1024 * 1024)
        
        console.print(f"\n[bold green]ONNX export successful![/bold green]")
        console.print(f"   Saved as: [cyan]{save_path}[/cyan]")
        console.print(f"   File size: {file_size:.2f} MB")
        
        console.print(f"\n💡 [bold]Next step:[/bold] Convert to TensorRT using your [cyan]onnx_to_tensorrt.py[/cyan] script.")

    except Exception as e:
        console.print(f"[bold red]Export failed:[/bold red]")
        console.print(traceback.format_exc())

if __name__ == '__main__':
    main()