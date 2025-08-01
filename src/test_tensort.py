#!/usr/bin/env python3
"""
TensorRT Tester using trtexec via Subprocess
This script uses a specific, hardcoded trtexec executable to run inference,
bypassing Python environment issues. It uses --dumpOutput and parses the
console output to retrieve inference results.
"""

import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import subprocess
from tqdm import tqdm
import zarr
import tempfile
import os
import re

# Path to the exact trtexec executable
# Must use specific version that matches the one used on system running real time inference
# This is a hardcoded path and should be adjusted based on your TensorRT installation
TRTEXEC_PATH = "/usr/local/TensorRT-10.0.1.6/bin/trtexec"

def preprocess_array(original_image, target_size=(640, 640)):
    """Preprocess a numpy array image for YOLO inference."""
    if original_image.ndim == 2:  # Grayscale
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    
    h, w, _ = original_image.shape
    scale = min(target_size[0] / h, target_size[1] / w)
    resized_w, resized_h = int(w * scale), int(h * scale)
    
    resized_img = cv2.resize(original_image, (resized_w, resized_h))
    padded_img = np.full((*target_size, 3), 114, dtype=np.uint8)
    padded_img[:resized_h, :resized_w] = resized_img
    
    preprocessed = (padded_img.transpose(2, 0, 1) / 255.0).astype(np.float32)
    return preprocessed, original_image, scale

def parse_trtexec_output(output_text):
    """Parses the raw text output from trtexec --dumpOutput."""
    try:
        # Find the output tensors by name
        num_dets_str = re.search(r"num_dets\]:\n([\s\S]*?)\n\n", output_text).group(1)
        bboxes_str = re.search(r"bboxes\]:\n([\s\S]*?)\n\n", output_text).group(1)
        scores_str = re.search(r"scores\]:\n([\s\S]*?)\n\n", output_text).group(1)
        labels_str = re.search(r"labels\]:\n([\s\S]*?)$", output_text).group(1)

        # Convert the text blocks to numpy arrays
        num_dets = np.array([int(num_dets_str.strip())], dtype=np.int32)
        bboxes = np.fromstring(bboxes_str.strip(), sep=' ').reshape(-1, 4).astype(np.float32)
        scores = np.fromstring(scores_str.strip(), sep=' ').astype(np.float32)
        labels = np.fromstring(labels_str.strip(), sep=' ').astype(np.int32)

        return {
            "num_dets": num_dets,
            "bboxes": bboxes,
            "scores": scores,
            "labels": labels
        }
    except Exception as e:
        print(f"Failed to parse trtexec output: {e}")
        return None


def run_trtexec_inference(engine_path, preprocessed_image):
    """
    Runs inference by calling the trtexec executable as a subprocess.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_bin_path = Path(tmpdir) / "input.bin"
        preprocessed_image.tofile(input_bin_path)

        command = [
            TRTEXEC_PATH,
            f"--loadEngine={engine_path}",
            f"--loadInputs=images:{input_bin_path}",
            "--dumpOutput"
        ]

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print("trtexec command failed!")
            print(result.stderr)
            return None

        return parse_trtexec_output(result.stdout)

def postprocess_output(outputs, scale):
    """Post-process the output from trtexec."""
    if outputs is None:
        return []
        
    num_dets = outputs["num_dets"][0]
    if num_dets == 0:
        return []

    bboxes = outputs["bboxes"][:num_dets]
    scores = outputs["scores"][:num_dets]
    
    final_boxes = []
    for i in range(num_dets):
        x1, y1, x2, y2 = bboxes[i]
        final_boxes.append((int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale), scores[i]))
        
    return final_boxes

def draw_boxes(image, boxes):
    """Draw bounding boxes on an image."""
    for box in boxes:
        x1, y1, x2, y2, conf = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Fish {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def main(args):
    engine_path = Path(args.engine)
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(TRTEXEC_PATH):
        print(f"Error: trtexec not found at the hardcoded path: {TRTEXEC_PATH}"); return
    if not engine_path.exists():
        print(f"Engine file not found: {engine_path}"); return
    if not input_path.exists():
        print(f"Input path not found: {input_path}"); return

    timings = []
    successful_detections = 0

    print(f"Processing Zarr dataset: {input_path}")
    zarr_root = zarr.open(str(input_path), mode='r')
    images_array = zarr_root['raw_video/images_ds']
    num_frames = images_array.shape[0]

    for frame_idx in tqdm(range(num_frames), desc="Processing Zarr Frames"):
        original_image = images_array[frame_idx]
        preprocessed, _, scale = preprocess_array(original_image)
        
        start_time = time.time()
        outputs = run_trtexec_inference(engine_path, preprocessed)
        timings.append(time.time() - start_time)
        
        final_boxes = postprocess_output(outputs, scale)
        if final_boxes: successful_detections += 1
        
        annotated = draw_boxes(cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR), final_boxes)
        cv2.imwrite(str(output_dir / f"frame_{frame_idx:06d}.jpg"), annotated)

    if timings:
        avg_latency = sum(timings) / len(timings)
        print(f"\nPerformance Summary:")
        print(f"   Images processed: {len(timings)}")
        print(f"   Images with detections: {successful_detections}")
        print(f"   Average latency: {avg_latency * 1000:.2f} ms")
        print(f"   Throughput: {1 / avg_latency:.2f} FPS")
        print(f"   Results saved to: {output_dir}")
    else:
        print("No images were processed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TensorRT engine tester using trtexec subprocess.")
    parser.add_argument('--engine', type=str, required=True, help='Path to TensorRT engine file.')
    parser.add_argument('--input-path', type=str, required=True, help='Path to a Zarr file.')
    parser.add_argument('--output-dir', type=str, default='trtexec_output', help='Directory for results.')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold (Note: NMS is in-engine).')
    
    args = parser.parse_args()
    main(args)