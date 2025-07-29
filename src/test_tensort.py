# test_tensort.py

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
from tqdm import tqdm

# Import PyCUDA
import pycuda.driver as cuda
import pycuda.autoinit  # This is needed for initializing CUDA context

# --- Helper Classes & Functions ---

class HostDeviceMem:
    """Helper class for managing host (CPU) and device (GPU) memory."""
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    """Allocates host and device buffers for a TensorRT engine."""
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    
    # TensorRT 10.0+ uses tensor-based API instead of binding-based API
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(tensor_name)
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
        
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(trt.volume(shape), dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        
        # Use get_tensor_mode instead of binding_is_input
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
            
    return inputs, outputs, bindings, stream

def preprocess_image(image_path, input_size=(640, 640)):
    """Preprocesses a single image for YOLOv8 inference."""
    original_image = cv2.imread(str(image_path))
    h, w, _ = original_image.shape
    
    # Calculate scaling factor
    scale = min(input_size[0] / h, input_size[1] / w)
    resized_w, resized_h = int(w * scale), int(h * scale)
    
    # Resize and pad the image
    resized_img = cv2.resize(original_image, (resized_w, resized_h))
    padded_img = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
    padded_img[:resized_h, :resized_w] = resized_img
    
    # Normalize and transpose from HWC to CHW
    preprocessed = (padded_img.transpose(2, 0, 1) / 255.0).astype(np.float32)
    return preprocessed, original_image, scale

def postprocess_output(output, original_image, scale, conf_thres, iou_thres):
    """Post-processes the YOLOv8 output to get bounding boxes."""
    # The output from your model is (1, 5, 8400) - single class fish detection
    # Transpose it to (1, 8400, 5) to make it easier to process
    output = np.transpose(output, (0, 2, 1))

    boxes, scores = [], []
    for det in output[0]:
        # Each det is [cx, cy, w, h, conf] - no class probabilities since single class
        confidence = det[4]
        if confidence >= conf_thres:
            # Bbox coordinates
            cx, cy, w, h = det[:4]
            # Convert from center-width-height to top-left-bottom-right
            x1 = int((cx - w / 2) / scale)
            y1 = int((cy - h / 2) / scale)
            width = int(w / scale)
            height = int(h / scale)
            
            boxes.append([x1, y1, width, height])
            scores.append(confidence)

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
    
    final_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            conf = scores[i]
            final_boxes.append((x, y, x + w, y + h, conf))
            
    return final_boxes

def draw_boxes(image, boxes):
    """Draws bounding boxes on an image."""
    for box in boxes:
        x1, y1, x2, y2, conf = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Fish {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def main(args):
    """Main function to test the TensorRT engine."""
    print(f"🚀 Starting TensorRT performance test...")
    print(f"   TensorRT Version: {trt.__version__}")

    engine_path = Path(args.engine)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize TensorRT
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        print("❌ Error: Failed to deserialize the engine.")
        return

    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    
    # Set tensor addresses for TensorRT 10.0+
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        context.set_tensor_address(tensor_name, bindings[i])
    
    image_files = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")))
    if not image_files:
        print(f"❌ No images found in {input_dir}")
        return

    print(f"🖼️  Found {len(image_files)} images to process.")

    timings = []
    for image_path in tqdm(image_files, desc="Inferencing"):
        preprocessed_image, original_image, scale = preprocess_image(image_path)
        
        np.copyto(inputs[0].host, preprocessed_image.ravel())

        start_time = time.time()
        
        cuda.memcpy_htod_async(inputs[0].device, inputs[0].host, stream)
        # Use execute_async_v3 for TensorRT 10.0+
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(outputs[0].host, outputs[0].device, stream)
        stream.synchronize()
        
        end_time = time.time()
        timings.append(end_time - start_time)

        output_data = outputs[0].host.reshape(1, 5, 8400)  # Your model's actual output shape
        final_boxes = postprocess_output(output_data, original_image, scale, args.conf_thres, args.iou_thres)
        
        annotated_image = draw_boxes(original_image.copy(), final_boxes)
        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), annotated_image)

    # --- Performance Summary ---
    print("\n--- 📊 Performance Summary ---")
    if timings:
        total_time = sum(timings)
        avg_latency = total_time / len(timings)
        fps = 1 / avg_latency if avg_latency > 0 else 0
        
        print(f"Total images processed: {len(timings)}")
        print(f"Total inference time: {total_time:.4f} seconds")
        print(f"Average latency per image: {avg_latency * 1000:.4f} ms")
        print(f"Frames Per Second (FPS): {fps:.2f}")
    else:
        print("No images were processed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test a TensorRT engine on a directory of images.")
    parser.add_argument('--engine', type=str, required=True, help='Path to the TensorRT engine file.')
    parser.add_argument('--input-dir', type=str, required=True, help='Directory containing input images.')
    parser.add_argument('--output-dir', type=str, default='tensorrt_test_output', help='Directory to save annotated images.')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold for object detection.')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS.')
    arguments = parser.parse_args()
    main(arguments)