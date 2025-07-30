#!/usr/bin/env python3
"""
TensorRT Tester for NMS-Enabled Models
Enhanced tester that handles both standard YOLO and NMS-enabled TensorRT engines.
"""

import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import tensorrt as trt
from tqdm import tqdm

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    PYCUDA_AVAILABLE = True
except ImportError:
    print("❌ PyCUDA not available. Please install with: pip install pycuda")
    PYCUDA_AVAILABLE = False

class HostDeviceMem:
    """Helper class for managing host (CPU) and device (GPU) memory."""
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

class EnhancedTensorRTInference:
    """Enhanced TensorRT inference that handles both standard and NMS-enabled models."""
    
    def __init__(self, engine_path, verbose=False):
        self.engine_path = Path(engine_path)
        self.verbose = verbose
        self.logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
        
        print(f"🚀 Initializing Enhanced TensorRT inference...")
        print(f"   Engine: {self.engine_path}")
        print(f"   TensorRT version: {trt.__version__}")
        
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
        
        # Detect model type
        self.model_type = self._detect_model_type()
        print(f"✅ TensorRT engine loaded successfully")
        print(f"🔍 Detected model type: {self.model_type}")
        self._print_engine_info()
    
    def _load_engine(self):
        """Load the TensorRT engine with error handling."""
        try:
            with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                engine_data = f.read()
                engine = runtime.deserialize_cuda_engine(engine_data)
                
            if engine is None:
                raise RuntimeError("Failed to deserialize engine")
                
            return engine
            
        except Exception as e:
            print(f"❌ Error loading engine: {e}")
            raise
    
    def _allocate_buffers(self):
        """Allocate host and device buffers."""
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        
        try:
            if hasattr(self.engine, 'num_io_tensors'):
                # TensorRT 10.0+ API
                for i in range(self.engine.num_io_tensors):
                    tensor_name = self.engine.get_tensor_name(i)
                    shape = self.engine.get_tensor_shape(tensor_name)
                    dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
                    mode = self.engine.get_tensor_mode(tensor_name)
                    
                    size = trt.volume(shape) if hasattr(trt, 'volume') else np.prod(shape)
                    
                    host_mem = cuda.pagelocked_empty(size, dtype)
                    device_mem = cuda.mem_alloc(host_mem.nbytes)
                    bindings.append(int(device_mem))
                    
                    if mode == trt.TensorIOMode.INPUT:
                        inputs.append(HostDeviceMem(host_mem, device_mem))
                    else:
                        outputs.append(HostDeviceMem(host_mem, device_mem))
                        
        except Exception as e:
            print(f"❌ Error allocating buffers: {e}")
            raise
            
        return inputs, outputs, bindings, stream
    
    def _detect_model_type(self):
        """Detect whether this is a standard YOLO or NMS-enabled model."""
        num_outputs = len(self.outputs)
        
        if num_outputs == 1:
            # Standard YOLO model with single output
            output_shape = self.outputs[0].host.shape
            if len(output_shape) >= 2 and output_shape[-1] in [8400, 6300]:  # Common YOLO detection counts
                return "standard_yolo"
            else:
                return "unknown_single_output"
        
        elif num_outputs == 4:
            # Likely NMS-enabled model with [num_dets, bboxes, scores, labels]
            return "nms_enabled"
        
        else:
            return f"unknown_{num_outputs}_outputs"
    
    def _print_engine_info(self):
        """Print detailed engine information."""
        print(f"\n📋 Engine Information:")
        print(f"   Input tensors: {len(self.inputs)}")
        print(f"   Output tensors: {len(self.outputs)}")
        
        for i, inp in enumerate(self.inputs):
            print(f"   Input {i}: shape={inp.host.shape}, dtype={inp.host.dtype}")
        
        for i, out in enumerate(self.outputs):
            print(f"   Output {i}: shape={out.host.shape}, dtype={out.host.dtype}")
    
    def _set_tensor_addresses(self):
        """Set tensor addresses for TensorRT 10.0+."""
        try:
            if hasattr(self.engine, 'num_io_tensors'):
                for i in range(self.engine.num_io_tensors):
                    tensor_name = self.engine.get_tensor_name(i)
                    self.context.set_tensor_address(tensor_name, self.bindings[i])
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Could not set tensor addresses: {e}")
    
    def preprocess_image(self, image_path, target_size=(640, 640)):
        """Preprocess image for YOLO inference."""
        try:
            original_image = cv2.imread(str(image_path))
            if original_image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            h, w, _ = original_image.shape
            
            # Calculate scaling factor
            scale = min(target_size[0] / h, target_size[1] / w)
            resized_w, resized_h = int(w * scale), int(h * scale)
            
            # Resize and pad
            resized_img = cv2.resize(original_image, (resized_w, resized_h))
            padded_img = np.full((*target_size, 3), 114, dtype=np.uint8)
            padded_img[:resized_h, :resized_w] = resized_img
            
            # Normalize and transpose
            preprocessed = (padded_img.transpose(2, 0, 1) / 255.0).astype(np.float32)
            return preprocessed, original_image, scale
            
        except Exception as e:
            print(f"❌ Error preprocessing image {image_path}: {e}")
            raise
    
    def infer(self, preprocessed_image):
        """Run inference."""
        try:
            # Copy input data
            np.copyto(self.inputs[0].host, preprocessed_image.ravel())
            
            # Set tensor addresses
            self._set_tensor_addresses()
            
            # Transfer input data to GPU
            cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)
            
            # Run inference
            if hasattr(self.context, 'execute_async_v3'):
                success = self.context.execute_async_v3(stream_handle=self.stream.handle)
            else:
                success = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            
            if not success:
                raise RuntimeError("Inference execution failed")
            
            # Transfer output data back to CPU
            outputs = []
            for output in self.outputs:
                cuda.memcpy_dtoh_async(output.host, output.device, self.stream)
                outputs.append(output.host.copy())
            
            self.stream.synchronize()
            return outputs
            
        except Exception as e:
            print(f"❌ Error during inference: {e}")
            raise
    
    def postprocess_output(self, outputs, original_image, scale, conf_thres=0.25, iou_thres=0.45):
        """Post-process output based on model type."""
        try:
            if self.model_type == "nms_enabled":
                return self._postprocess_nms_enabled(outputs, original_image, scale)
            elif self.model_type == "standard_yolo":
                return self._postprocess_standard_yolo(outputs[0], original_image, scale, conf_thres, iou_thres)
            else:
                print(f"⚠️  Unknown model type: {self.model_type}")
                return []
                
        except Exception as e:
            print(f"❌ Error in post-processing: {e}")
            raise
    
    def _postprocess_nms_enabled(self, outputs, original_image, scale):
        """Post-process NMS-enabled model outputs."""
        # outputs should be [num_dets, bboxes, scores, labels]
        if len(outputs) != 4:
            print(f"⚠️  Expected 4 outputs for NMS model, got {len(outputs)}")
            return []
        
        num_dets = int(outputs[0][0])  # Number of detections
        bboxes = outputs[1][:num_dets]  # Bounding boxes
        scores = outputs[2][:num_dets]  # Confidence scores  
        labels = outputs[3][:num_dets]  # Class labels
        
        final_boxes = []
        for i in range(num_dets):
            # Convert normalized coordinates to pixel coordinates
            x1 = int(bboxes[i][0] / scale)
            y1 = int(bboxes[i][1] / scale)
            x2 = int(bboxes[i][2] / scale)
            y2 = int(bboxes[i][3] / scale)
            conf = scores[i]
            
            final_boxes.append((x1, y1, x2, y2, conf))
        
        return final_boxes
    
    def _postprocess_standard_yolo(self, output_data, original_image, scale, conf_thres, iou_thres):
        """Post-process standard YOLO output."""
        # Auto-detect shape and process
        if output_data.ndim == 1:
            # Try common YOLO shapes
            total_elements = len(output_data)
            possible_shapes = [(1, 5, 8400), (1, 84, 8400), (1, 5, 6300)]
            
            output_reshaped = None
            for shape in possible_shapes:
                if np.prod(shape) == total_elements:
                    output_reshaped = output_data.reshape(shape)
                    break
            
            if output_reshaped is None:
                raise ValueError(f"Could not reshape output with {total_elements} elements")
        else:
            output_reshaped = output_data
        
        # Process detections
        if output_reshaped.shape[1] == 5:
            # Single class format
            output_transposed = np.transpose(output_reshaped, (0, 2, 1))
        else:
            # Multi-class format
            output_transposed = np.transpose(output_reshaped, (0, 2, 1))
        
        boxes, scores = [], []
        for det in output_transposed[0]:
            if len(det) == 5:
                confidence = det[4]
            else:
                confidence = np.max(det[4:])
            
            if confidence >= conf_thres:
                cx, cy, w, h = det[:4]
                x1 = int((cx - w / 2) / scale)
                y1 = int((cy - h / 2) / scale)
                width = int(w / scale)
                height = int(h / scale)
                
                boxes.append([x1, y1, width, height])
                scores.append(confidence)
        
        # Apply NMS
        final_boxes = []
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    conf = scores[i]
                    final_boxes.append((x, y, x + w, y + h, conf))
        
        return final_boxes
    
    def draw_boxes(self, image, boxes):
        """Draw bounding boxes on image."""
        for box in boxes:
            x1, y1, x2, y2, conf = box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Fish {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image

def main(args):
    """Main function to test the TensorRT engine."""
    if not PYCUDA_AVAILABLE:
        return
    
    print(f"🚀 Enhanced TensorRT Engine Tester")
    print(f"   Engine: {args.engine}")
    print(f"   Input directory: {args.input_dir}")
    print(f"   Output directory: {args.output_dir}")
    print("=" * 60)
    
    # Initialize paths
    engine_path = Path(args.engine)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate inputs
    if not engine_path.exists():
        print(f"❌ Engine file not found: {engine_path}")
        return
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    # Find images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(ext))
        image_files.extend(input_dir.glob(ext.upper()))
    
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"❌ No images found in {input_dir}")
        return
    
    print(f"🖼️  Found {len(image_files)} images to process")
    
    try:
        # Initialize inference engine
        inference_engine = EnhancedTensorRTInference(engine_path, verbose=args.verbose)
        
        # Process images
        timings = []
        successful_detections = 0
        
        for image_path in tqdm(image_files, desc="Processing images"):
            try:
                # Preprocess
                preprocessed_image, original_image, scale = inference_engine.preprocess_image(image_path)
                
                # Inference
                start_time = time.time()
                outputs = inference_engine.infer(preprocessed_image)
                inference_time = time.time() - start_time
                timings.append(inference_time)
                
                # Postprocess
                final_boxes = inference_engine.postprocess_output(
                    outputs, original_image, scale, args.conf_thres, args.iou_thres
                )
                
                if final_boxes:
                    successful_detections += 1
                
                # Draw and save
                annotated_image = inference_engine.draw_boxes(original_image.copy(), final_boxes)
                output_path = output_dir / image_path.name
                cv2.imwrite(str(output_path), annotated_image)
                
            except Exception as e:
                print(f"❌ Error processing {image_path.name}: {e}")
                continue
        
        # Performance summary
        print(f"\n📊 Performance Summary:")
        if timings:
            total_time = sum(timings)
            avg_latency = total_time / len(timings)
            fps = 1 / avg_latency if avg_latency > 0 else 0
            
            print(f"   Images processed: {len(timings)}")
            print(f"   Images with detections: {successful_detections}")
            print(f"   Detection rate: {successful_detections/len(timings)*100:.1f}%")
            print(f"   Average latency: {avg_latency * 1000:.2f} ms")
            print(f"   Throughput: {fps:.2f} FPS")
            print(f"   Results saved to: {output_dir}")
        else:
            print("   No images were processed successfully")
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Enhanced TensorRT engine tester for NMS-enabled models")
    parser.add_argument('--engine', type=str, required=True, help='Path to TensorRT engine file')
    parser.add_argument('--input-dir', type=str, required=True, help='Directory with input images')
    parser.add_argument('--output-dir', type=str, default='enhanced_tensorrt_output', help='Directory for results')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    main(args)