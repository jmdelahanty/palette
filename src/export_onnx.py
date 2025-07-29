# export.py

import argparse
from io import BytesIO
from pathlib import Path

import onnx
import torch
from ultralytics import YOLO

try:
    import onnxsim
except ImportError:
    onnxsim = None


def parse_args():
    """Parses command-line arguments for ONNX export."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-w',
        '--weights',
        type=str,
        required=True,
        help='PyTorch yolov8 weights file path (.pt)')
    parser.add_argument(
        '--iou-thres',
        type=float,
        default=0.65,
        help='IOU threshold for NMS')
    parser.add_argument(
        '--conf-thres',
        type=float,
        default=0.25,
        help='Confidence threshold for NMS')
    parser.add_argument(
        '--topk',
        type=int,
        default=1,
        help='Max number of detection bboxes')
    parser.add_argument(
        '--opset',
        type=int,
        default=11,
        help='ONNX opset version')
    parser.add_argument(
        '--sim',
        action='store_true',
        help='Simplify ONNX model')
    parser.add_argument(
        '--input-shape',
        nargs='+',
        type=int,
        default=[1, 3, 640, 640],
        help='Model input shape')
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Export ONNX device (e.g., "cpu" or "cuda:0")')
    args = parser.parse_args()
    assert len(args.input_shape) == 4
    return args


def main(args):
    """Main function to export YOLO model to ONNX."""
    print("🚀 Starting YOLOv8 ONNX export...")

    # Load the YOLO model
    yolo_model = YOLO(args.weights)
    model = yolo_model.model.fuse().eval()
    
    # Set model to the specified device
    model.to(args.device)

    # Create a fake input tensor
    fake_input = torch.randn(args.input_shape).to(args.device)

    # Define output path
    save_path = Path(args.weights).with_suffix('.onnx')

    print(f"📦 Exporting model to {save_path}...")

    # Export the model to ONNX
    with BytesIO() as f:
        torch.onnx.export(
            model,
            fake_input,
            f,
            opset_version=args.opset,
            input_names=['images'],
            output_names=['output0']) # A single output is standard for most conversions
        
        f.seek(0)
        onnx_model = onnx.load(f)

    # Check the ONNX model
    onnx.checker.check_model(onnx_model)

    # Simplify the ONNX model if requested
    if args.sim:
        if onnxsim is None:
            print("⚠️ 'onnx-simplifier' not found. Skipping simplification.")
        else:
            print("✨ Simplifying ONNX model...")
            try:
                onnx_model, check = onnxsim.simplify(onnx_model)
                assert check, 'ONNX simplification check failed'
            except Exception as e:
                print(f" simplifier failure: {e}")

    # Save the final ONNX model
    onnx.save(onnx_model, save_path)
    print(f"✅ ONNX export success! Model saved as {save_path}")


if __name__ == '__main__':
    arguments = parse_args()
    main(arguments)