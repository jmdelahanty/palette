#!/usr/bin/env python3
"""
Debug script to inspect TensorRT engine structure
"""
import argparse
from pathlib import Path
import tensorrt as trt

def inspect_engine(engine_path):
    """Inspect a TensorRT engine and print its structure."""
    print(f"🔍 Inspecting TensorRT engine: {engine_path}")
    print(f"   TensorRT Version: {trt.__version__}")

    # Initialize TensorRT
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        print("❌ Error: Failed to deserialize the engine.")
        return

    print(f"\n📋 Engine Information:")
    print(f"   Number of IO tensors: {engine.num_io_tensors}")
    
    inputs = []
    outputs = []
    
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        tensor_shape = engine.get_tensor_shape(tensor_name)
        tensor_dtype = engine.get_tensor_dtype(tensor_name)
        tensor_mode = engine.get_tensor_mode(tensor_name)
        mode_str = "INPUT" if tensor_mode == trt.TensorIOMode.INPUT else "OUTPUT"
        
        print(f"\n   Tensor {i}: '{tensor_name}'")
        print(f"     - Shape: {tensor_shape}")
        print(f"     - Data type: {tensor_dtype}")
        print(f"     - Mode: {mode_str}")
        print(f"     - Volume (total elements): {trt.volume(tensor_shape)}")
        
        if tensor_mode == trt.TensorIOMode.INPUT:
            inputs.append((tensor_name, tensor_shape, tensor_dtype))
        else:
            outputs.append((tensor_name, tensor_shape, tensor_dtype))
    
    print(f"\n📊 Summary:")
    print(f"   Inputs: {len(inputs)}")
    for i, (name, shape, dtype) in enumerate(inputs):
        print(f"     Input {i}: {name} -> {shape} ({dtype})")
        
    print(f"   Outputs: {len(outputs)}")
    for i, (name, shape, dtype) in enumerate(outputs):
        print(f"     Output {i}: {name} -> {shape} ({dtype})")
        
    # Based on the output shape, suggest the correct reshape
    if outputs:
        main_output_shape = outputs[0][1]  # Get first output shape
        print(f"\n💡 For your postprocessing:")
        print(f"   Replace this line:")
        print(f"     output_data = outputs[0].host.reshape(1, 84, 8400)")
        print(f"   With:")
        print(f"     output_data = outputs[0].host.reshape{tuple(main_output_shape)}")
        
        # If it looks like YOLOv8 format, provide additional guidance
        if len(main_output_shape) == 3:
            b, c, n = main_output_shape
            if c == 84:  # Standard YOLOv8 with 80 classes
                print(f"   This looks like YOLOv8 format with {n} detections")
                print(f"   You may also need to update your postprocessing to handle {n} detections instead of 8400")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inspect TensorRT engine structure")
    parser.add_argument('--engine', type=str, required=True, help='Path to the TensorRT engine file.')
    args = parser.parse_args()
    
    engine_path = Path(args.engine)
    if not engine_path.exists():
        print(f"❌ Engine file not found: {engine_path}")
        exit(1)
        
    inspect_engine(engine_path)