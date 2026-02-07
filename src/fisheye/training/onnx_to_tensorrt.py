# onnx_to_tensorrt.py

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Optional

# --- Configuration ---
# Hardcoded path to your TensorRT executable
# This should match the version used for real-time inference on the machine you're running your engine on
TRTEXEC_PATH = "/usr/local/TensorRT-10.0.1.6/bin/trtexec"


def _infer_onnx_manifest_path(onnx_path: Path) -> Path:
    """Return expected sidecar ONNX manifest path for a given ONNX model."""
    return Path(f"{onnx_path}.manifest.json")


def _load_output_contract_from_manifest(manifest_path: Path) -> Optional[list[dict[str, Any]]]:
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    onnx_block = payload.get("onnx")
    if not isinstance(onnx_block, dict):
        return None
    outputs = onnx_block.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        return None
    normalized: list[dict[str, Any]] = []
    for item in outputs:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "name": str(item.get("name", "?")),
                "dtype": str(item.get("dtype", "?")),
                "shape": item.get("shape"),
            }
        )
    return normalized or None


def _load_output_contract_from_onnx(onnx_path: Path) -> Optional[list[dict[str, Any]]]:
    try:
        import onnx  # type: ignore
        from onnx import TensorProto  # type: ignore
    except Exception:
        return None
    try:
        model = onnx.load(str(onnx_path))
    except Exception:
        return None
    outputs: list[dict[str, Any]] = []
    for value_info in model.graph.output:
        tensor_type = value_info.type.tensor_type
        elem_type = int(getattr(tensor_type, "elem_type", 0) or 0)
        dtype_name = TensorProto.DataType.Name(elem_type) if elem_type > 0 else "UNDEFINED"
        dims: list[Any] = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(int(dim.dim_value))
            elif dim.HasField("dim_param"):
                dims.append(str(dim.dim_param))
            else:
                dims.append(None)
        outputs.append(
            {
                "name": str(value_info.name),
                "dtype": dtype_name,
                "shape": dims,
            }
        )
    return outputs or None


def _format_output_contract(outputs: Optional[list[dict[str, Any]]]) -> Optional[str]:
    if not outputs:
        return None
    parts: list[str] = []
    for item in outputs:
        name = str(item.get("name", "?"))
        dtype = str(item.get("dtype", "?"))
        shape = item.get("shape")
        if isinstance(shape, list):
            shape_text = "(" + ",".join(str(v) for v in shape) + ")"
        else:
            shape_text = "(?)"
        parts.append(f"{name}[{dtype},{shape_text}]")
    return ", ".join(parts) if parts else None

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
        '--trtexec',
        type=str,
        default=None,
        help='Optional path to trtexec (overrides the built-in default).'
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
    print("Starting ONNX to TensorRT conversion...")

    onnx_path = Path(args.onnx)
    engine_path = Path(args.engine)
    trtexec_path = Path(args.trtexec) if args.trtexec else Path(TRTEXEC_PATH)

    # Validate paths
    if not onnx_path.exists():
        print(f"Error: ONNX file not found at {onnx_path}")
        return
    if not trtexec_path.exists():
        print(f"Error: trtexec not found at {trtexec_path}")
        return

    manifest_path = _infer_onnx_manifest_path(onnx_path)
    output_contract = _load_output_contract_from_manifest(manifest_path)
    contract_source = None
    if output_contract:
        contract_source = f"manifest: {manifest_path}"
    else:
        output_contract = _load_output_contract_from_onnx(onnx_path)
        if output_contract:
            contract_source = "onnx_graph"

    contract_text = _format_output_contract(output_contract)
    if contract_text:
        print(f"Output contract ({contract_source}): {contract_text}")
    else:
        print("Output contract: unavailable")

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
