import argparse
import zarr
import numpy as np
from pathlib import Path
import cv2

def preprocess_batch(image_batch, target_size=(640, 640)):
    """Preprocesses a batch of numpy array images for TensorRT inference."""
    # This logic should exactly match your training preprocessing
    preprocessed_batch = np.zeros((len(image_batch), 3, *target_size), dtype=np.float32)
    scales = []

    for i, original_image in enumerate(image_batch):
        if original_image.ndim == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

        h, w, _ = original_image.shape
        scale = min(target_size[0] / h, target_size[1] / w)
        scales.append(scale)
        resized_w, resized_h = int(w * scale), int(h * scale)

        resized_img = cv2.resize(original_image, (resized_w, resized_h))
        padded_img = np.full((target_size[0], target_size[1], 3), 114, dtype=np.uint8)
        padded_img[:resized_h, :resized_w] = resized_img

        preprocessed = (padded_img.transpose(2, 0, 1) / 255.0)
        preprocessed_batch[i] = preprocessed

    return preprocessed_batch.astype(np.float32), scales

def main():
    parser = argparse.ArgumentParser(description="Extract and preprocess a batch of frames from a Zarr archive.")
    parser.add_argument("zarr_path", type=str, help="Path to the Zarr file.")
    parser.add_argument("output_bin", type=str, help="Path to save the output binary file.")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of frames to extract.")
    parser.add_argument("--start_frame", type=int, default=0, help="Starting frame index.")
    args = parser.parse_args()

    try:
        zarr_root = zarr.open(args.zarr_path, mode='r')
        images_array = zarr_root['raw_video/images_ds']
        
        # Determine batch range
        end_frame = min(args.start_frame + args.batch_size, images_array.shape[0])
        
        print(f"Extracting frames {args.start_frame} to {end_frame-1} from {args.zarr_path}...")
        
        # Extract and preprocess
        image_batch_raw = images_array[args.start_frame:end_frame]
        preprocessed_batch, _ = preprocess_batch(image_batch_raw)
        
        # Save to binary file
        preprocessed_batch.tofile(args.output_bin)
        print(f"Batch of {len(image_batch_raw)} frames saved to: {args.output_bin}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()