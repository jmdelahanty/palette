# Crimson Data Format Documentation
## H5 and Zarr File Integration

### Overview
The Crimson application uses two complementary data formats for storing and loading experimental data:
- **HDF5 (.h5)**: Primary session data, metadata, and tracking information
- **Zarr (.zarr)**: Detection data from computer vision models (YOLO)

### HDF5 Session Files

#### File Structure
```
out_analysis.h5
├── / (root attributes)
│   ├── session_uuid
│   ├── protocol_name
│   └── other session metadata
├── /tracking_data/
│   ├── bounding_boxes (structured array)
│   └── chaser_states (structured array)
├── /video_metadata/
│   └── frame_metadata (structured array)
├── /analysis/
│   ├── gap_info
│   └── interpolation_mask
├── /calibration_snapshot/
│   ├── arena_config_json
│   └── {camera_id}/
│       ├── homography_matrix_yml
│       └── calibration attributes
└── /protocol_snapshot/
    └── protocol_definition_json
```

#### Key Data Structures

##### LoggedBoundingBox (C++ struct)
```cpp
struct LoggedBoundingBox {
    int64_t payload_timestamp_ns_epoch;
    int64_t received_timestamp_ns_epoch;
    uint64_t payload_frame_id;
    uint16_t payload_camera_id;
    uint8_t box_index_in_payload;
    float x_min;
    float y_min;
    float width;      // Note: width, not x_max
    float height;     // Note: height, not y_max
    uint16_t class_id;
    float confidence;
};
```

##### Frame Metadata
- `stimulus_frame_num`: Frame number in the stimulus/experiment timeline
- `triggering_camera_frame_id`: Corresponding camera frame ID
- `timestamp_ns`: Nanosecond timestamp

##### Chaser States
- Position and state information for chaser/target tracking
- Linked to stimulus frame numbers

### Zarr Detection Files

#### File Structure
```
2025-08-12T20-25-51Z_arena_4_chaser_detections.zarr/
├── .zattrs (metadata)
│   ├── fps: 60.0
│   ├── video_path
│   ├── model_path
│   ├── class_names: {"0": "fish"}
│   └── other metadata
├── /bboxes/
│   └── .zarray (shape: [n_frames, max_detections, 4], dtype: float32)
├── /scores/
│   └── .zarray (shape: [n_frames, max_detections], dtype: float32)
├── /class_ids/
│   └── .zarray (shape: [n_frames, max_detections], dtype: int32)
└── /n_detections/
    └── .zarray (shape: [n_frames], dtype: int32)
```

#### Key Characteristics
- **Bounding box format**: `[x_min, y_min, x_max, y_max]` (different from H5's width/height)
- **Chunking**: Data is chunked (e.g., 100 frames per chunk) for efficient access
- **Fill values**: -1.0 for missing detections
- **Single detection per frame**: In this example, max_detections = 1

### Integration in Crimson

#### Loading Process
1. **Directory Selection**: User selects a directory containing video and data files
2. **H5 Loading**: `H5SessionLoader` attempts to find and load `.h5` files
3. **Zarr Loading**: `ZarrDetectionLoader` searches for `.zarr` directories
4. **Synchronization**: Both data sources are aligned by frame numbers

#### Data Access Pattern
```cpp
// H5 data access
auto h5_boxes = H5SessionLoader::getBoundingBoxesForFrame(h5_data, frame_num);

// Zarr data access  
auto zarr_boxes = zarr_loader.getBoundingBoxesForFrame(frame_num);
```

#### Visual Differentiation
- **H5 bounding boxes**: Rendered in green (0.2f, 1.0f, 0.2f, 1.0f)
- **Zarr bounding boxes**: Rendered in blue (0.2f, 0.2f, 1.0f, 1.0f)

### TensorStore Integration

#### CMake Configuration
```cmake
# TensorStore for Zarr support
FetchContent_Declare(
    tensorstore
    GIT_REPOSITORY https://github.com/google/tensorstore.git
    GIT_TAG v0.1.64
)
FetchContent_MakeAvailable(tensorstore)

# Link required components
target_link_libraries(redgui PRIVATE
    tensorstore::all_drivers  # Includes file, zarr, and other drivers
)
```

#### Key TensorStore Operations
- **Opening stores**: Using `kvstore::Open` with JSON spec
- **Reading arrays**: Using `ts::Open<float, 3>` for typed access
- **Slicing**: Using `ts::Dims(0).IndexSlice()` for frame selection

### Common Issues and Solutions

#### Issue 1: Linker Errors
**Problem**: Undefined references to TensorStore internal functions
**Solution**: Link `tensorstore::all_drivers` instead of individual components

#### Issue 2: Type Mismatches
**Problem**: IndexSlice expects `tensorstore::Index` (signed) not `size_t` (unsigned)
**Solution**: Cast with `static_cast<tensorstore::Index>(frame_id)`

#### Issue 3: Coordinate Format
**Problem**: Zarr stores `[x_min, y_min, x_max, y_max]` but H5 uses `[x_min, y_min, width, height]`
**Solution**: Convert during loading:
```cpp
box.width = x_max - x_min;
box.height = y_max - y_min;
```

### Future Improvements
- Support for multiple detections per frame
- Dynamic detection of coordinate format (bbox vs x/y/w/h)
- Unified data interface abstracting H5/Zarr differences
- Caching layer for frequently accessed frames
- Support for additional Zarr compression formats

### File Naming Conventions
- H5 files: `out_analysis.h5` (analysis output)
- Zarr files: `{timestamp}_arena_{id}_chaser_detections.zarr`
- Video files: `Cam{camera_id}.mp4`

### Performance Considerations
- Zarr chunking allows efficient partial reads
- TensorStore provides async I/O capabilities
- Both formats support parallel access
- Memory mapping available for large datasets