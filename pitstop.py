# pitstop.py

# Using Wilson Chen's, HHMI Janelia Johnson Lab/Aso Lab, mechanic pipeline

import argparse
from io import BytesIO
from pathlib import Path
import sys
import torch
import onnx

try:
    import onnxsim
except ImportError:
    onnxsim = None

import tensorrt as trt
from ultralytics import YOLO

# --- Configuration ---
SRC_PATH = Path(__file__).parent
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def find_latest_model(runs_dir="runs/detect"):
    """Finds the path to the latest 'best.pt' model file."""
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        return None
    # Find all 'best.pt' files and get the one with the latest modification time
    best_pt_files = list(runs_path.rglob("weights/best.pt"))
    if not best_pt_files:
        return None
    return max(best_pt_files, key=lambda p: p.stat().st_mtime)


def export_to_onnx(weights_path, input_shape):
    """Exports a PyTorch model to an in-memory ONNX model."""
    print("📦 Exporting model to ONNX...")
    yolo_model = YOLO(weights_path)
    model = yolo_model.model.fuse().eval()
    device = 'cpu'  # ONNX export is often more stable on CPU
    model.to(device)
    fake_input = torch.randn(input_shape).to(device)

    with BytesIO() as f:
        torch.onnx.export(
            model,
            fake_input,
            f,
            opset_version=11,
            input_names=['images'],
            output_names=['output0']
        )
        f.seek(0)
        onnx_model = onnx.load(f)

    onnx.checker.check_model(onnx_model)

    # Simplify the model
    if onnxsim:
        print("✨ Simplifying ONNX model...")
        try:
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, 'ONNX simplification check failed'
        except Exception as e:
            print(f"⚠️ Simplifier failure: {e}")
    else:
        print("⚠️ 'onnx-simplifier' not found. Skipping simplification.")
        
    return onnx_model


def build_tensorrt_engine(onnx_model, engine_path, precision, workspace=4):
    """Builds a TensorRT engine from an ONNX model using the Python API."""
    print(f"🔧 Building TensorRT engine with {precision} precision...")
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()

    # Set workspace size in Gibibytes (GiB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace * (1024 ** 3))

    # Set precision
    if precision == 'fp16' and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # Create network
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    if not parser.parse(onnx_model.SerializeToString()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        raise RuntimeError("Failed to parse ONNX model.")

    # Build the serialized engine
    plan = builder.build_serialized_network(network, config)
    if not plan:
        raise RuntimeError("Failed to build the TensorRT engine.")

    # Save the engine to a file
    with open(engine_path, "wb") as f:
        f.write(plan)
    print(f"✅ Engine built successfully and saved to: {engine_path}")


def main(args):
    """Main function to orchestrate the PyTorch -> ONNX -> TensorRT pipeline."""
    print("🚀 Starting Mechanic Pipeline: PyTorch -> ONNX -> TensorRT...")
    print(f"   Using TensorRT version: {trt.__version__}")

    # --- Step 1: Find the model ---
    if args.weights:
        weights_path = Path(args.weights)
        if not weights_path.exists():
            print(f"❌ Error: Specified weights file not found at {weights_path}")
            return
        print(f"✅ Using specified model: {weights_path}")
    else:
        print("🔍 Searching for the latest trained model...")
        weights_path = find_latest_model()
        if not weights_path:
            print("❌ Error: Could not find any 'best.pt' models in the 'runs/detect' directory.")
            return
        print(f"✅ Found latest model: {weights_path}")

    engine_path = weights_path.with_suffix('.engine')

    # --- Step 2: Export to ONNX (in memory) ---
    onnx_model = export_to_onnx(weights_path, args.input_shape)

    # --- Step 3: Build TensorRT Engine ---
    build_tensorrt_engine(onnx_model, engine_path, args.precision, args.workspace)
    
    print("\n🎉🎉🎉")
    print("Mechanic pipeline completed successfully!")
    print(f"✅ Your TensorRT engine is ready at: {engine_path}")
    print("You can now run test_tensort.py on this engine file.")
    print("🎉🎉🎉")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Full pipeline to convert a PyTorch model to a TensorRT engine.")
    parser.add_argument(
        '-w', '--weights',
        type=str,
        default=None,
        help="Path to a specific .pt model file. If not provided, the latest trained model will be used."
    )
    parser.add_argument(
        '--precision',
        type=str,
        default='fp16',
        choices=['fp32', 'fp16'],
        help='Precision for the TensorRT engine (default: fp16).'
    )
    parser.add_argument(
        '--input-shape',
        nargs='+',
        type=int,
        default=[1, 3, 640, 640],
        help='Model input shape (default: 1 3 640 640)'
    )
    parser.add_argument(
        '--workspace',
        type=int,
        default=4,
        help='GPU workspace size in GiB for building the engine (default: 4).'
    )
    arguments = parser.parse_args()
    main(arguments)