# onnx_to_tensorrt.py

import argparse
import subprocess
from pathlib import Path

# --- Configuration ---
# Hardcoded path to your TensorRT executable
# This should match the version used for real-time inference on the machine you're running your engine on
TRTEXEC_PATH = "/usr/local/TensorRT-10.0.1.6/bin/trtexec"

def parse_args():
    """Parses command-line arguments for TensorRT conversion."""
    parser = argparse.ArgumentParser(description="Convert ONNX model to a TensorRT engine with advanced profiling.")
    parser.add_argument(
        '--onnx',
        type=str,
        required=True,
        help='Path to the input ONNX model file.'
    )
    parser.add_argument(
        '--engine',
        type=str,
        required=True,
        help='Path to save the output TensorRT engine file.'
    )
    parser.add_argument(
        '--precision',
        type=str,
        default='fp16',
        choices=['fp16', 'int8'],
        help='Precision for the TensorRT engine (default: fp16).'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output from trtexec.'
    )
    parser.add_argument(
        '--cuda-graph',
        action='store_true',
        help='Enable CUDA graph capture for inference.'
    )
    parser.add_argument(
        '--profiling',
        action='store_true',
        help='Enable exporting of timing, output, and profile data to JSON files.'
    )
    return parser.parse_args()

def main(args):
    """Main function to run the ONNX to TensorRT conversion."""
    print("Starting ONNX to TensorRT conversion with advanced options...")

    onnx_path = Path(args.onnx)
    engine_path = Path(args.engine)
    trtexec_path = Path(TRTEXEC_PATH)

    # Validate paths
    if not onnx_path.exists():
        print(f"Error: ONNX file not found at {onnx_path}")
        return
    if not trtexec_path.exists():
        print(f"Error: trtexec not found at {trtexec_path}")
        return

    # Construct the base trtexec command
    command = [
        str(trtexec_path),
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}"
    ]

    if args.precision == 'fp16':
        command.append("--fp16")
    elif args.precision == 'int8':
        command.append("--int8")
    if args.verbose:
        command.append("--verbose")
    if args.cuda_graph:
        command.append("--useCudaGraph")
    if args.profiling:
        # Define paths for the JSON report files based on the engine name
        report_base_path = engine_path.with_suffix('')
        timing_json = report_base_path.with_name(f"{report_base_path.name}_timing.json")
        output_json = report_base_path.with_name(f"{report_base_path.name}_output.json")
        profile_json = report_base_path.with_name(f"{report_base_path.name}_profile.json")
        
        command.extend([
            f"--exportTimes={timing_json}",
            f"--exportOutput={output_json}",
            f"--exportProfile={profile_json}",
            "--separateProfileRun"  # This ensures profiling is done in a separate run for accurate end-to-end timing
        ])
        print(f"Profiling enabled with separate run for accurate e2e timing.")
        print(f"Reports will be saved to:")
        print(f"   - Timing: {timing_json}")
        print(f"   - Output: {output_json}")
        print(f"   - Profile: {profile_json}")

    print(f"Running command: {' '.join(command)}")

    # Execute the command
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

        # Stream the output from trtexec
        for line in process.stdout:
            print(line.strip())

        process.wait()

        if process.returncode == 0 and engine_path.exists():
            print(f"\nTensorRT engine created successfully!")
            print(f"   Engine saved to: {engine_path}")
        else:
            print(f"\nError: TensorRT conversion failed with return code {process.returncode}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    arguments = parse_args()
    main(arguments)