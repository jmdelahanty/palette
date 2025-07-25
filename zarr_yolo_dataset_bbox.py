# zarr_yolo_dataset_bbox.py (with train/val split logic)

import zarr
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List
from pydantic import BaseModel, ValidationError, field_validator
from sklearn.model_selection import train_test_split


class TrackingSummary(BaseModel):
    total_frames: int; frames_tracked: int; percent_tracked: float
class TrackingResultsAttrs(BaseModel):
    column_names: List[str]
    @field_validator('column_names')
    @classmethod
    def check_column_count(cls, v: List[str]) -> List[str]:
        if len(v) != 9: raise ValueError(f"Expected 9 columns, found {len(v)}."); return v
def validate_zarr_structure(root: zarr.hierarchy.Group):
    print("🔬 Running Zarr structure validation...")
    required_paths = {'crop_data': 'group', 'crop_data/roi_images': 'dataset', 'tracking': 'group', 'tracking/tracking_results': 'dataset'}
    for path, type_name in required_paths.items():
        if path not in root: raise ValueError(f"Validation Error: Required {type_name} '{path}' not found.")
    if root['crop_data/roi_images'].ndim != 3: raise ValueError(f"Validation Error: 'roi_images' should have 3 dimensions, has {root['crop_data/roi_images'].ndim}.")
    if root['tracking/tracking_results'].ndim != 2 or root['tracking/tracking_results'].shape[1] != 9: raise ValueError(f"Validation Error: 'tracking_results' shape is not (N, 9).")
    try:
        if 'summary_statistics' not in root['tracking'].attrs: raise ValueError("Missing 'summary_statistics' in '/tracking'.")
        TrackingSummary.model_validate(root['tracking'].attrs['summary_statistics'])
        if 'column_names' not in root['tracking/tracking_results'].attrs: raise ValueError("Missing 'column_names' in '/tracking/tracking_results'.")
        TrackingResultsAttrs.model_validate(root['tracking/tracking_results'].attrs)
    except ValidationError as e: raise ValueError(f"Zarr attribute validation failed:\n{e}")
    print("✅ Zarr structure is valid.")


class ZarrYOLODataset(Dataset):
    """
    A custom Dataset for BOUNDING BOX detection from Zarr, with train/val splitting.
    """
    def __init__(self, zarr_path: str, mode: str = 'train', split_ratio: float = 0.8, random_seed: int = 42):
        super().__init__()
        assert mode in ['train', 'val'], "Mode must be 'train' or 'val'."
        
        self.root = zarr.open(zarr_path, mode='r')
        validate_zarr_structure(self.root)
        
        all_valid_indices = np.where(~np.isnan(self.root['tracking/tracking_results'][:, 0]))[0]
        
        train_indices, val_indices = train_test_split(
            all_valid_indices,
            train_size=split_ratio,
            random_state=random_seed,
            shuffle=True
        )
        
        if mode == 'train':
            self.indices = train_indices
            print(f"Initialized 'train' dataset with {len(self.indices)} samples.")
        else: # mode == 'val'
            self.indices = val_indices
            print(f"Initialized 'val' dataset with {len(self.indices)} samples.")

        # --- NEW CODE BLOCK TO ADD ---
        # Pre-cache all labels to satisfy the plotting utility.
        # The plotting function expects a list of dictionaries.
        print(f"Pre-caching {len(self.indices)} labels for plotting utility...")
        static_bbox = np.array([[0, 0.5, 0.5, 0.9, 0.9]], dtype=np.float32)
        self.labels = [{"bboxes": static_bbox, "cls": np.array([0.])} for _ in range(len(self.indices))]
        # --- END OF NEW CODE BLOCK ---

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int):
        # Map the item index to the actual Zarr frame index
        zarr_index = self.indices[index]
        
        roi_image = self.root['crop_data/roi_images'][zarr_index]
        image_shape = roi_image.shape # Get shape before transposing, e.g., (320, 320)
        
        # Create a 3-channel image in HWC format (Height, Width, Channels)
        image = np.stack([roi_image]*3, axis=-1)
        
        # Transpose the image from HWC to CHW format (Channels, Height, Width)
        image = image.transpose(2, 0, 1)
        
        # The label is a (1, 5) array: [[class, x, y, w, h]]
        label = np.array([[0, 0.5, 0.5, 0.9, 0.9]], dtype=np.float32)

        # Return a complete dictionary with all keys the trainer/validator expects
        return {
            "img": image,
            "cls": label[:, 0],
            "bboxes": label[:, 1:5],
            "im_file": f"zarr_frame_{zarr_index}", # For plotting
            "ori_shape": image_shape,            # For validation metrics
            "ratio_pad": (1.0, (0.0, 0.0))       # No padding, ratio is 1.0
        }


if __name__ == '__main__':
    TEST_ZARR_PATH = '/home/delahantyj@hhmi.org/Desktop/concentricOMR3/video.zarr'
    print(f"--- Running Test on {TEST_ZARR_PATH} ---")
    
    try:
        # 1. Instantiate both a training and a validation dataset
        print("\n--- Creating Training Set ---")
        train_set = ZarrYOLODataset(zarr_path=TEST_ZARR_PATH, mode='train')
        
        print("\n--- Creating Validation Set ---")
        val_set = ZarrYOLODataset(zarr_path=TEST_ZARR_PATH, mode='val')

        # 2. Check their lengths to confirm the split
        print(f"\nTotal training samples: {len(train_set)}")
        print(f"Total validation samples: {len(val_set)}")
        
        # 3. Verify the first item of the training set
        if len(train_set) > 0:
            image, label = train_set[0]
            print(f"\nFirst train image shape: {image.shape}, Label: {label}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")