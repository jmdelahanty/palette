import h5py
import sys
import os
import json
import argparse
import numpy as np
import polars as pl

# --- Enum Mappings (from C++ headers) ---
# Source: src/core/stimulus_globals.h
EXPERIMENT_EVENT_TYPE = {
    0: "PROTOCOL_START", 1: "PROTOCOL_STOP", 2: "PROTOCOL_PAUSE", 3: "PROTOCOL_RESUME", 4: "PROTOCOL_FINISH",
    5: "PROTOCOL_CLEAR", 6: "PROTOCOL_LOAD", 7: "STEP_ADD", 8: "STEP_REMOVE", 9: "STEP_MOVE_UP",
    10: "STEP_MOVE_DOWN", 11: "STEP_START", 12: "STEP_END", 13: "ITI_START", 14: "ITI_END",
    15: "PARAMS_APPLIED", 16: "MANAGER_REINIT", 17: "MANAGER_REINIT_FAIL", 18: "LOOM_AUTO_REPEAT_TRIGGER",
    19: "LOOM_MANUAL_START", 20: "USER_INTERVENTION", 21: "ERROR_RUNTIME", 22: "LOG_MESSAGE",
    23: "IPC_BOUNDING_BOX_RECEIVED", 24: "CHASER_PRE_PERIOD_START", 25: "CHASER_TRAINING_START",
    26: "CHASER_POST_PERIOD_START", 27: "CHASER_CHASE_SEQUENCE_START", 28: "CHASER_CHASE_SEQUENCE_END",
    29: "CHASER_RANDOM_TARGET_SET"
}

# Source: src/core/stimulus_globals.h
STIMULUS_MODE_TYPE = {
    -1: "UNDEFINED", 2: "COHERENT_DOTS", 3: "MOVING_GRATING", 4: "SOLID_BLACK", 5: "SOLID_WHITE",
    6: "CONCENTRIC_GRATING", 7: "LOOMING_DOT", 8: "STATIC_IMAGE", 9: "CALIBRATION_GRID",
    10: "ARENA_DEFINITION_SQUARE", 11: "SPOTLIGHT", 12: "CHASER", 99: "NONE"
}

def get_h5_object(hf, path):
    """Safely gets an object from the HDF5 file."""
    obj = hf.get(path)
    if obj is None:
        print(f"Warning: Dataset or group not found at path: {path}")
    return obj

def read_session_info(hf):
    """Reads and displays the main session attributes from the root of the H5 file."""
    print("\n--- Reading Session Info (Root Attributes) ---")
    if not hf.attrs:
        print("No root attributes found.")
        return

    for key, value in hf.attrs.items():
        if isinstance(value, bytes):
            value = value.decode('utf-8', 'ignore')
        print(f"  {key}: {value}")

def read_subject_metadata(hf):
    """Reads and displays attributes from the /subject_metadata group."""
    print("\n--- Reading /subject_metadata ---")
    subject_group = get_h5_object(hf, '/subject_metadata')
    if subject_group is None:
        print("No subject metadata found.")
        return

    if not subject_group.attrs:
        print("Subject metadata group is empty.")
        return

    print("Subject Metadata:")
    for key, value in subject_group.attrs.items():
        if isinstance(value, bytes):
            value = value.decode('utf-8', 'ignore')
        print(f"  {key}: {value}")

def read_events(hf):
    """Reads and displays the /events dataset using Polars."""
    print("\n--- Reading /events ---")
    events_ds = get_h5_object(hf, '/events')
    if events_ds is None or len(events_ds) == 0:
        print("No events found.")
        return

    # Decode byte strings before creating the DataFrame for robustness
    events_data = events_ds[:]
    df = pl.DataFrame({
        'timestamp_ns_epoch': events_data['timestamp_ns_epoch'],
        'timestamp_ns_session': events_data['timestamp_ns_session'],
        'event_type_id': events_data['event_type_id'],
        'current_step_index': events_data['current_step_index'],
        'name_or_context': [s.decode('utf-8').strip('\x00') for s in events_data['name_or_context']],
        'stimulus_mode_id': events_data['stimulus_mode_id'],
        'details_json': [s.decode('utf-8').strip('\x00') for s in events_data['details_json']],
        'stimulus_frame_num': events_data['stimulus_frame_num'],
        'camera_frame_id': events_data['camera_frame_id'],
    })

    df = df.with_columns([
        pl.col("event_type_id").map_elements(lambda x: EXPERIMENT_EVENT_TYPE.get(x, "UNKNOWN"), return_dtype=pl.String).alias("event_type_str"),
        pl.col("stimulus_mode_id").map_elements(lambda x: STIMULUS_MODE_TYPE.get(x, "UNKNOWN"), return_dtype=pl.String).alias("stimulus_mode_str"),
        pl.from_epoch("timestamp_ns_epoch", time_unit="ns").alias("timestamp")
    ])

    display_cols = ['timestamp', 'event_type_str', 'current_step_index', 'name_or_context', 
                    'stimulus_mode_str', 'stimulus_frame_num', 'camera_frame_id', 'details_json']
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=180):
        print(df.select(display_cols))

def read_tracking_data(hf):
    """Reads and displays a summary of the /tracking_data/bounding_boxes dataset using Polars."""
    print("\n--- Reading /tracking_data/bounding_boxes ---")
    bbox_ds = get_h5_object(hf, '/tracking_data/bounding_boxes')
    if bbox_ds is None or len(bbox_ds) == 0:
        print("No bounding box data found.")
        return

    df = pl.from_numpy(bbox_ds[:])
    df = df.with_columns(
        pl.from_epoch("received_timestamp_ns_epoch", time_unit="ns").alias("timestamp")
    )

    print(f"Total bounding boxes logged: {len(df)}")
    print("\nFirst 5 entries:")
    print(df.head(5))
    print("\nLast 5 entries:")
    print(df.tail(5))

def read_chaser_states(hf):
    """Reads and displays a summary of the /tracking_data/chaser_states dataset using Polars."""
    print("\n--- Reading /tracking_data/chaser_states ---")
    chaser_ds = get_h5_object(hf, '/tracking_data/chaser_states')
    if chaser_ds is None or len(chaser_ds) == 0:
        print("No chaser state data found.")
        return

    df = pl.from_numpy(chaser_ds[:])
    df = df.with_columns(
        (pl.col("timestamp_ns_session") / 1e9).alias("session_time_s")
    )

    print(f"Total chaser states logged: {len(df)}")
    print("\nFirst 5 entries:")
    print(df.head(5))
    print("\nLast 5 entries:")
    print(df.tail(5))

def read_frame_metadata(hf):
    """Reads and displays a summary of the /video_metadata/frame_metadata dataset using Polars."""
    print("\n--- Reading /video_metadata/frame_metadata ---")
    frame_ds = get_h5_object(hf, '/video_metadata/frame_metadata')
    if frame_ds is None or len(frame_ds) == 0:
        print("No frame metadata found.")
        return

    df = pl.from_numpy(frame_ds[:])
    df = df.with_columns(
        pl.from_epoch("timestamp_ns", time_unit="ns").alias("timestamp")
    )
    print(f"Total frames logged: {len(df)}")
    print("\nFirst 5 entries:")
    print(df.head(5))
    print("\nLast 5 entries:")
    print(df.tail(5))

def read_protocol_snapshot(hf):
    """Reads and pretty-prints the /protocol_snapshot/protocol_definition_json dataset."""
    print("\n--- Reading /protocol_snapshot ---")
    protocol_ds = get_h5_object(hf, '/protocol_snapshot/protocol_definition_json')
    if protocol_ds is None:
        return

    json_string = protocol_ds[()].decode('utf-8')
    protocol_data = json.loads(json_string)

    print(json.dumps(protocol_data, indent=4))

def read_calibration_snapshot(hf, output_dir):
    """Reads datasets from /calibration_snapshot and saves image buffers to disk."""
    print("\n--- Reading /calibration_snapshot ---")
    calib_group = get_h5_object(hf, '/calibration_snapshot')
    if calib_group is None:
        return

    arena_config_ds = get_h5_object(calib_group, 'arena_config_json')
    if arena_config_ds:
        print("\nArena Config JSON:")
        try:
            json_string = arena_config_ds[()].decode('utf-8')
            arena_config_data = json.loads(json_string)
            print(json.dumps(arena_config_data, indent=2))
        except Exception as e:
            print(f"Could not parse arena config JSON. Error: {e}")

    for cam_id, cam_item in calib_group.items():
        if not isinstance(cam_item, h5py.Group):
            continue

        print(f"\nProcessing camera: {cam_id}")

        homography_ds = get_h5_object(cam_item, 'homography_matrix_yml')
        if homography_ds:
            print("\nHomography Matrix (YML format):")
            print(homography_ds[()].decode('utf-8'))

        # CORRECTED: Look for the correct image buffer dataset names
        image_types_to_extract = {
            "homography_image": "homography_image_png_buffer",
            "scale_image": "scale_image_png_buffer"
        }

        for name, dataset_key in image_types_to_extract.items():
            img_ds = get_h5_object(cam_item, dataset_key)
            if img_ds:
                img_filepath = os.path.join(output_dir, f"extracted_{cam_id}_{name}.png")
                try:
                    with open(img_filepath, 'wb') as f:
                        f.write(img_ds[()])
                    print(f"\nSaved {name} to: {img_filepath}")
                except Exception as e:
                    print(f"Could not save {name}. Error: {e}")

def read_stimulus_coordinates(hf):
    """Reads and displays the /stimulus_coordinates group with texture dimensions."""
    print("\n--- Reading /stimulus_coordinates ---")
    coord_group = get_h5_object(hf, '/stimulus_coordinates')
    if coord_group is None:
        print("No stimulus coordinate info found.")
        return
    
    print("Stimulus Coordinate Information:")
    
    # Read attributes from the main group
    if coord_group.attrs:
        print("\nMain attributes:")
        for key, value in coord_group.attrs.items():
            if isinstance(value, bytes):
                value = value.decode('utf-8', 'ignore')
            print(f"  {key}: {value}")
    
    # Check for arena-specific groups (if multiple arenas)
    for arena_name in coord_group.keys():
        if isinstance(coord_group[arena_name], h5py.Group):
            arena_group = coord_group[arena_name]
            print(f"\nArena: {arena_name}")
            
            # Read arena attributes
            for key, value in arena_group.attrs.items():
                if isinstance(value, bytes):
                    value = value.decode('utf-8', 'ignore')
                print(f"  {key}: {value}")
            
            # Check for custom_coordinates subgroup
            if 'custom_coordinates' in arena_group:
                custom_group = arena_group['custom_coordinates']
                print("  Custom coordinates:")
                for key, value in custom_group.attrs.items():
                    print(f"    {key}: {value}")

def main():
    parser = argparse.ArgumentParser(description="Inspect components of a citrus HDF5 log file using Polars.")
    parser.add_argument("filepath", help="Path to the HDF5 file.")
    parser.add_argument("--all", action="store_true", help="Inspect all components.")
    parser.add_argument("--info", action="store_true", help="Inspect the session info root attributes.")
    parser.add_argument("--subject", action="store_true", help="Inspect the subject metadata.")
    parser.add_argument("--events", action="store_true", help="Inspect the events table.")
    parser.add_argument("--bbox", action="store_true", help="Inspect the bounding box tracking data.")
    parser.add_argument("--chaser", action="store_true", help="Inspect the chaser state tracking data.")
    parser.add_argument("--frames", action="store_true", help="Inspect the video frame metadata.")
    parser.add_argument("--protocol", action="store_true", help="Inspect the protocol snapshot.")
    parser.add_argument("--calib", action="store_true", help="Inspect the calibration snapshot.")
    parser.add_argument("--coords", action="store_true", help="Inspect the stimulus coordinate info.")  # NEW
    parser.add_argument("--output_dir", default=".", help="Directory to save extracted files (like calibration images).")

    args = parser.parse_args()

    if not os.path.exists(args.filepath):
        print(f"Error: File not found at '{args.filepath}'")
        return

    print(f"--- Opening HDF5 File: {os.path.basename(args.filepath)} ---")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with h5py.File(args.filepath, 'r') as hf:
        if args.info or args.all:
            read_session_info(hf)
        if args.subject or args.all:
            read_subject_metadata(hf)
        if args.events or args.all:
            read_events(hf)
        if args.bbox or args.all:
            read_tracking_data(hf)
        if args.chaser or args.all:
            read_chaser_states(hf)
        if args.frames or args.all:
            read_frame_metadata(hf)
        if args.protocol or args.all:
            read_protocol_snapshot(hf)
        if args.calib or args.all:
            read_calibration_snapshot(hf, args.output_dir)
        if args.coords or args.all:  # NEW
            read_stimulus_coordinates(hf)

    print("\n--- Inspection Complete ---")

if __name__ == "__main__":
    main()