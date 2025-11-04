# Pose Schema Guide for Visualization Tools

This guide explains how to load and use pose schema information from Palette keypoint zarr archives.

## Overview

As of this update, keypoint runs in Palette now store complete pose schema information including:
- **Node names**: Ordered list of keypoint labels (e.g., `["bladder", "eye_left", "eye_right"]`)
- **Edges**: Connectivity between keypoints for skeleton visualization (e.g., `[[0, 1], [0, 2]]`)
- **Metadata**: Additional schema information (units, coordinate system, version, notes)

This information is stored in the `pose_schema` attribute of keypoint run groups.

## Where Schema is Stored

The pose schema is available in:
- `keypoints_runs/<run_name>` - Both traditional and YOLO-based keypoint detections
- `refined_keypoints_runs/<run_name>` - Refined keypoint runs (copied from source)

## Schema Structure in Zarr

The `pose_schema` attribute is a dictionary with the following structure:

```python
{
    "name": str,              # Schema identifier (e.g., "traditional_v1")
    "nodes": List[str],       # Ordered keypoint names (e.g., ["bladder", "eye_left", "eye_right"])
    "edges": List[List[int]], # Skeleton edges (e.g., [[0, 1], [0, 2]])
    "metadata": dict,         # Additional metadata (units, coordinate system, etc.)
    "source": str             # Path to original schema JSON file
}
```

**Important**: Node IDs are implicit - they correspond to the index in the `nodes` list:
- `nodes[0]` = "bladder" → node ID 0
- `nodes[1]` = "eye_left" → node ID 1
- `nodes[2]` = "eye_right" → node ID 2

## Loading Schema from Zarr (Python)

### Basic Loading

```python
import zarr

# Open zarr archive
root = zarr.open('/path/to/data.zarr', mode='r')

# Get keypoint run (or use specific run name)
keypoint_run = root.attrs.get('keypoints_runs/latest')  # or specific run name
kp_group = root[f'keypoints_runs/{keypoint_run}']

# Load pose schema
pose_schema = kp_group.attrs.get('pose_schema')

if pose_schema:
    # Extract information
    keypoint_labels = pose_schema['nodes']      # ["bladder", "eye_left", "eye_right"]
    skeleton_edges = pose_schema['edges']        # [[0, 1], [0, 2]]
    schema_name = pose_schema['name']            # "traditional_v1"
    schema_metadata = pose_schema['metadata']    # {"units": "pixels", ...}
else:
    # Fallback for old data without schema
    keypoint_labels = kp_group.attrs.get('keypoint_labels', ['bladder', 'eye_left', 'eye_right'])
    skeleton_edges = [[0, 1], [0, 2]]  # Default skeleton
```

### Using Palette's Schema Utilities

If your visualization tool has access to Palette's Python code:

```python
from fisheye.pose.schema import schema_from_metadata

# Load schema
schema_dict = kp_group.attrs.get('pose_schema')
if schema_dict:
    schema = schema_from_metadata(schema_dict)

    # Access properties
    print(schema.name)              # "traditional_v1"
    print(schema.node_names)        # ["bladder", "eye_left", "eye_right"]
    print(schema.edges)             # [[0, 1], [0, 2]]
    print(schema.num_keypoints)     # 3

    # Lookup node index by name
    bladder_idx = schema.index("bladder")  # Returns 0
```

## Visualization Example

### Plotting Keypoints with Skeleton

```python
import matplotlib.pyplot as plt
import numpy as np
import zarr

# Load data
root = zarr.open('/path/to/data.zarr', mode='r')
kp_group = root['keypoints_runs/detect_2024-01-15_10-30-00']

# Load schema
pose_schema = kp_group.attrs.get('pose_schema')
if pose_schema:
    labels = pose_schema['nodes']
    edges = pose_schema['edges']
else:
    labels = ['bladder', 'eye_left', 'eye_right']
    edges = [[0, 1], [0, 2]]

# Load keypoint data
keypoints_roi = kp_group['keypoints_roi'][:]  # Shape: (n_rois, n_keypoints, 2)

# Visualize a single ROI
roi_idx = 0
kp = keypoints_roi[roi_idx]  # Shape: (n_keypoints, 2)

fig, ax = plt.subplots()

# Draw skeleton edges
for edge in edges:
    from_idx, to_idx = edge
    if from_idx < len(kp) and to_idx < len(kp):
        x_vals = [kp[from_idx, 0], kp[to_idx, 0]]
        y_vals = [kp[from_idx, 1], kp[to_idx, 1]]
        ax.plot(x_vals, y_vals, 'b-', linewidth=2, alpha=0.6)

# Draw keypoints
for idx, (x, y) in enumerate(kp):
    if np.isfinite(x) and np.isfinite(y):
        label = labels[idx] if idx < len(labels) else f"kp_{idx}"
        ax.scatter(x, y, s=100, c='red', zorder=10)
        ax.text(x, y, f"  {label}", fontsize=10)

ax.set_aspect('equal')
ax.invert_yaxis()  # Image coordinates
plt.title(f'Keypoints with Skeleton (ROI {roi_idx})')
plt.show()
```

## Fallback Handling for Old Data

Not all keypoint runs will have the `pose_schema` attribute (e.g., runs created before this update). Always provide fallbacks:

```python
def get_keypoint_info(kp_group):
    """Get keypoint labels and edges with fallback for old data."""
    pose_schema = kp_group.attrs.get('pose_schema')

    if pose_schema:
        # New data with schema
        return {
            'labels': pose_schema['nodes'],
            'edges': pose_schema['edges'],
            'name': pose_schema['name']
        }
    else:
        # Old data - use legacy keypoint_labels
        labels = kp_group.attrs.get('keypoint_labels', ['bladder', 'eye_left', 'eye_right'])
        edges = [[0, 1], [0, 2]]  # Default traditional skeleton
        return {
            'labels': labels,
            'edges': edges,
            'name': 'unknown'
        }
```

## Available Schemas

Currently, Palette defines two pose schemas:

### 1. traditional_v1 (3 keypoints)
Used by both traditional and YOLO-based detectors for basic fish tracking.

```json
{
  "name": "traditional_v1",
  "nodes": [
    {"id": 0, "name": "bladder"},
    {"id": 1, "name": "eye_left"},
    {"id": 2, "name": "eye_right"}
  ],
  "edges": [[0, 1], [0, 2]]
}
```

### 2. traditional_feret_v1 (11 keypoints)
Extended schema with Feret diameter endpoints for eye measurements.

```json
{
  "name": "fish_pose_traditional_feret_v1",
  "nodes": [
    {"id": 0, "name": "bladder"},
    {"id": 1, "name": "eye_left"},
    {"id": 2, "name": "eye_right"},
    {"id": 3-10, "name": "eye_left/right_feret_major/minor_a/b"}
  ],
  "edges": [
    [0, 1], [0, 2],
    [1, 3], [1, 4], [1, 5], [1, 6],
    [2, 7], [2, 8], [2, 9], [2, 10]
  ]
}
```

## Coordinate Systems

Keypoints are stored in three coordinate systems:

1. **keypoints_roi**: ROI-local pixel coordinates (e.g., [0-64, 0-64])
2. **keypoints_img**: Full-image pixel coordinates (e.g., [0-4512, 0-4512])
3. **keypoints_norm**: Normalized coordinates [0, 1]

The schema's `metadata.coordinate_system` indicates which coordinate space the schema references (usually "roi").

## Checking Schema Availability

```python
def check_schema_availability(zarr_path):
    """Check which runs have pose_schema available."""
    root = zarr.open(zarr_path, mode='r')

    results = {
        'keypoints_runs': {},
        'refined_keypoints_runs': {}
    }

    # Check keypoints_runs
    if 'keypoints_runs' in root:
        for run_name in root['keypoints_runs'].keys():
            kp_group = root[f'keypoints_runs/{run_name}']
            has_schema = 'pose_schema' in kp_group.attrs
            results['keypoints_runs'][run_name] = has_schema

    # Check refined_keypoints_runs
    if 'refined_keypoints_runs' in root:
        for run_name in root['refined_keypoints_runs'].keys():
            kp_group = root[f'refined_keypoints_runs/{run_name}']
            has_schema = 'pose_schema' in kp_group.attrs
            results['refined_keypoints_runs'][run_name] = has_schema

    return results

# Usage
availability = check_schema_availability('/path/to/data.zarr')
print("Schema availability:", availability)
```

## Summary for Visualization Developers

1. **Always check** if `pose_schema` exists in the keypoint run attributes
2. **Provide fallbacks** for old data that doesn't have schema information
3. **Use node index** (position in `nodes` list) to reference keypoints, not names
4. **Draw skeleton edges** to visualize keypoint connectivity
5. **Schema is propagated** through refinement, so refined runs will have the same schema as their source

## Example: Complete Loader Function

```python
def load_keypoints_with_schema(zarr_path, run_name=None):
    """
    Load keypoints and schema from zarr archive.

    Returns:
        dict with keys:
            - 'keypoints_roi': (n_rois, n_kp, 2) array
            - 'keypoints_img': (n_rois, n_kp, 2) array
            - 'labels': List of keypoint names
            - 'edges': List of [from_idx, to_idx] pairs
            - 'schema_name': Schema identifier
    """
    import zarr
    import numpy as np

    root = zarr.open(zarr_path, mode='r')

    # Find keypoint run
    if run_name is None:
        # Try refined first, then regular
        if 'refined_keypoints_runs' in root and root['refined_keypoints_runs'].attrs.get('latest'):
            run_name = root['refined_keypoints_runs'].attrs['latest']
            kp_group = root[f'refined_keypoints_runs/{run_name}']
        else:
            run_name = root['keypoints_runs'].attrs['latest']
            kp_group = root[f'keypoints_runs/{run_name}']
    else:
        # Try both locations
        if f'refined_keypoints_runs/{run_name}' in root:
            kp_group = root[f'refined_keypoints_runs/{run_name}']
        else:
            kp_group = root[f'keypoints_runs/{run_name}']

    # Load keypoint data
    keypoints_roi = np.array(kp_group['keypoints_roi'])
    keypoints_img = np.array(kp_group['keypoints_img'])

    # Load schema
    pose_schema = kp_group.attrs.get('pose_schema')
    if pose_schema:
        labels = pose_schema['nodes']
        edges = pose_schema['edges']
        schema_name = pose_schema['name']
    else:
        # Fallback for old data
        labels = kp_group.attrs.get('keypoint_labels', ['bladder', 'eye_left', 'eye_right'])
        edges = [[0, 1], [0, 2]]
        schema_name = 'unknown'

    return {
        'keypoints_roi': keypoints_roi,
        'keypoints_img': keypoints_img,
        'labels': labels,
        'edges': edges,
        'schema_name': schema_name,
        'run_name': run_name
    }
```

## Testing

To verify schema loading works correctly, run:

```bash
python3 test_pose_schema_loading.py
```

This tests:
- Schema loading from package files
- Schema serialization/deserialization
- Node index lookups
- Metadata reconstruction

---

**Last updated**: 2025-01-XX (with Phase 1 schema propagation changes)
