# #!/usr/bin/env python3
# """
# Visualize experiment timeline from zarr events file.

# Creates a color-coded timeline showing when different types of events occurred
# during the experiment, helping to understand the progression and structure of
# the experimental session.

# UPDATED: Now reads event type mappings directly from HDF5 file, automatically
# adapting to new event types without code changes.
# """

# from __future__ import annotations

# import argparse
# from pathlib import Path
# from typing import Optional

# import matplotlib.pyplot as plt
# import numpy as np
# import zarr
# from matplotlib.patches import Patch


# def load_enum_mappings(root: zarr.Group) -> dict[str, dict[int, str]]:
#     """
#     Load enum mappings from the HDF5 file's /enums group.
#     This makes the script automatically adapt to any new event types or stimulus modes.
    
#     Returns:
#         Dictionary with keys 'events', 'stimulus_modes', etc. mapping IDs to names
#     """
#     mappings = {}
    
#     # Try to load the enums group
#     enums_group = root.get("enums")
#     if enums_group is None:
#         print("Warning: No /enums group found in file. Using empty mappings.")
#         return mappings
    
#     # Load each enum mapping dataset
#     for dataset_name in enums_group.array_keys():
#         try:
#             dataset = enums_group[dataset_name]
#             data = dataset[:]
            
#             # Convert to dictionary: id -> name
#             mapping_dict = {}
#             for record in data:
#                 enum_id = int(record['id'])
#                 enum_name = record['name']
                
#                 # Handle bytes vs string
#                 if isinstance(enum_name, bytes):
#                     enum_name = enum_name.decode('utf-8', errors='ignore').rstrip('\x00')
#                 else:
#                     enum_name = str(enum_name).rstrip('\x00')
                
#                 mapping_dict[enum_id] = enum_name
            
#             mappings[dataset_name] = mapping_dict
#             print(f"✓ Loaded {len(mapping_dict)} {dataset_name} mappings from HDF5")
            
#         except Exception as e:
#             print(f"Warning: Could not load enum mapping '{dataset_name}': {e}")
    
#     return mappings


# # Color scheme for different event categories
# EVENT_COLORS = {
#     "PROTOCOL": "#2E86AB",      # Blue - protocol level events
#     "STEP": "#A23B72",           # Purple - step boundaries
#     "ITI": "#8B8C89",            # Gray - inter-trial intervals
#     "PRE_PERIOD": "#F18F01",     # Orange - pre-training
#     "TRAINING": "#C73E1D",       # Red - training period
#     "POST_PERIOD": "#6A994E",    # Green - post-training
#     "CHASE": "#BC4749",          # Dark red - chase sequences
#     "CHASER": "#BC4749",         # Dark red - chaser events
#     "LOOM": "#E63946",           # Bright red - looming events
#     "CAVE": "#9D4EDD",           # Purple - cave dweller behaviors
#     "POSITIONING": "#FB8500",    # Orange - positioning phase
#     "GRID": "#06FFA5",           # Cyan - grid stimulus events
#     "PARAMS": "#FFC300",         # Yellow - parameter changes
#     "ERROR": "#D00000",          # Dark red - errors
#     "OTHER": "#4A5859",          # Dark gray - other events
# }


# def list_stimulus_runs(root: zarr.Group) -> list[str]:
#     """List all available stimulus runs in the zarr file."""
#     analysis = root.get("analysis")
#     if analysis is None:
#         return []
#     stimulus_parent = analysis.get("stimulus_runs")
#     if stimulus_parent is None:
#         return []
#     return sorted(
#         name for name in stimulus_parent.group_keys() 
#         if isinstance(stimulus_parent.get(name), zarr.Group)
#     )


# def load_events(root: zarr.Group, run_name: str) -> dict[str, np.ndarray]:
#     """Load events from a stimulus run."""
#     run_group = root[f"analysis/stimulus_runs/{run_name}"]
    
#     if "events" not in run_group:
#         raise ValueError(f"Run '{run_name}' does not contain an 'events' dataset.")
    
#     events_node = run_group["events"]
    
#     # Handle both structured array and group storage
#     if isinstance(events_node, zarr.Array):
#         events_array = events_node[:]
#         field_names = events_array.dtype.names
#     elif isinstance(events_node, zarr.Group):
#         # Reconstruct from separate arrays
#         field_names = list(events_node.array_keys())
#         events_dict = {}
#         for field in field_names:
#             events_dict[field] = events_node[field][:]
#         return events_dict
#     else:
#         raise TypeError("Unsupported Zarr node type for events dataset.")
    
#     # Convert structured array to dictionary of arrays
#     events_dict = {}
#     for field in field_names:
#         events_dict[field] = events_array[field]
    
#     return events_dict


# def categorize_event(event_name: str, mode_name: Optional[str] = None) -> str:
#     """Categorize events using both event name/context and stimulus mode."""
#     name_upper = (event_name or "").upper()
#     mode_upper = (mode_name or "").upper()

#     # Mode-first categorization
#     if "CHASER" in mode_upper:
#         # Further split chaser actions
#         if any(keyword in name_upper for keyword in ("RETREAT", "CHASE", "POSITION", "LOOM")):
#             if "LOOM" in name_upper:
#                 return "LOOM"
#             if "RETREAT" in name_upper:
#                 return "CHASER_RETREAT"
#             if "POSITION" in name_upper or "APPROACH" in name_upper:
#                 return "POSITIONING"
#             if "CHASE" in name_upper:
#                 return "CHASER"
#         return "CHASER"
#     if "GRID" in mode_upper:
#         return "GRID"
#     if "LOOM" in mode_upper:
#         return "LOOM"

#     # Protocol management
#     if "PROTOCOL" in name_upper:
#         return "PROTOCOL"
#     if "STEP" in name_upper:
#         return "STEP"
#     if "ITI" in name_upper:
#         return "ITI"

#     if "PRE_PERIOD" in name_upper:
#         return "PRE_PERIOD"
#     if "TRAINING" in name_upper:
#         return "TRAINING"
#     if "POST_PERIOD" in name_upper:
#         return "POST_PERIOD"

#     if "CAVE" in name_upper:
#         return "CAVE"
#     if "LOOM" in name_upper or "ESCAPE" in name_upper:
#         return "LOOM"
#     if "POSITION" in name_upper or "APPROACH" in name_upper:
#         return "POSITIONING"
#     if "CHASE" in name_upper or "CHASER" in name_upper:
#         return "CHASER"

#     if "PARAMS" in name_upper or "APPLIED" in name_upper:
#         return "PARAMS"
#     if "ERROR" in name_upper or "FAIL" in name_upper:
#         return "ERROR"

#     return "OTHER"


# def plot_timeline(
#     events_dict: dict[str, np.ndarray],
#     event_type_mappings: dict[int, str],
#     stimulus_mode_mappings: dict[int, str],
#     output_path: Optional[Path] = None,
#     dump_json: Optional[Path] = None,
# ):
#     """
#     Create a color-coded timeline visualization of events.
    
#     Args:
#         events_dict: Dictionary of event arrays
#         event_type_mappings: Mapping from event_type_id to event name (from HDF5)
#         output_path: Optional path to save the plot
#     """
    
#     # Get timestamps and event types
#     # Try different possible timestamp field names
#     timestamp_field = None
#     for field_name in ['timestamp_ns_session', 'timestamp_ns_epoch', 
#                        'relative_timestamp_ns', 'timestamp_ns']:
#         if field_name in events_dict:
#             timestamp_field = field_name
#             break
    
#     if timestamp_field is None:
#         raise ValueError("Could not find timestamp field in events")
    
#     timestamps = events_dict[timestamp_field]
    
#     # Try different event type field names
#     event_type_field = None
#     for field_name in ['event_type_id', 'event_type']:
#         if field_name in events_dict:
#             event_type_field = field_name
#             break
    
#     if event_type_field is None:
#         raise ValueError("Could not find event type field in events")
    
#     event_types = events_dict[event_type_field]
    
#     # Get event names if available
#     event_names = None
#     for field_name in ['name_or_context', 'event_name']:
#         if field_name in events_dict:
#             event_names = events_dict[field_name]
#             break

#     mode_ids = events_dict.get('stimulus_mode_id')
#     mode_names: Optional[np.ndarray]
#     if mode_ids is not None:
#         mode_names = np.array([
#             stimulus_mode_mappings.get(int(mid), 'UNKNOWN') for mid in mode_ids
#         ])
#     else:
#         mode_names = None
    
#     # Convert timestamps to seconds
#     time_seconds = timestamps / 1e9
    
#     # Normalize to start at 0
#     time_seconds = time_seconds - time_seconds[0]
    
#     # Group events by category
#     categories = {}
#     event_details = {}  # Store event type names for each category
    
#     other_records = []
#     for i, (time, event_type) in enumerate(zip(time_seconds, event_types)):
#         # Get the actual event name from the mapping
#         event_name = event_type_mappings.get(int(event_type), f"UNKNOWN_{event_type}")
        
#         # Override with name_or_context if available (more specific context)
#         if event_names is not None:
#             context_name = event_names[i]
#             if isinstance(context_name, bytes):
#                 context_name = context_name.decode('utf-8', errors='ignore').rstrip('\x00')
#             if context_name:  # Only override if non-empty
#                 event_name = context_name
        
#         mode_name = None
#         if mode_names is not None:
#             mode_name = mode_names[i]

#         category = categorize_event(str(event_name), mode_name)
        
#         if category == "OTHER":
#             other_records.append(
#                 {
#                     "time_s": float(time),
#                     "event_type_id": int(event_type),
#                     "event_name": event_name,
#                     "mode": mode_name,
#                     "context": event_name,
#                 }
#             )

#         if category not in categories:
#             categories[category] = []
#             event_details[category] = set()
        
#         categories[category].append(time)
#         event_details[category].add(event_name)
    
#     # Create the plot
#     fig, ax = plt.subplots(figsize=(14, 8))
    
#     # Prepare data for eventplot
#     positions = []
#     colors = []
#     labels = []
#     lineoffsets = []
    
#     # Sort categories for consistent vertical ordering
#     sorted_categories = sorted(categories.keys())
    
#     for i, category in enumerate(sorted_categories):
#         if len(categories[category]) > 0:  # Only add if there are events
#             positions.append(categories[category])
#             colors.append(EVENT_COLORS.get(category, EVENT_COLORS["OTHER"]))
            
#             # Create label with count
#             count = len(categories[category])
#             labels.append(f"{category} ({count})")
            
#             lineoffsets.append(i)
    
#     # Plot the events
#     if positions:
#         ax.eventplot(
#             positions,
#             lineoffsets=lineoffsets,
#             colors=colors,
#             linewidths=2.5,
#             linelengths=0.8,
#             orientation='horizontal'
#         )
    
#     # Customize the plot
#     ax.set_yticks(lineoffsets)
#     ax.set_yticklabels(labels)
#     ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
#     ax.set_ylabel('Event Category', fontsize=12, fontweight='bold')
#     ax.set_title('Experiment Timeline: Event Progression', 
#                  fontsize=14, fontweight='bold', pad=20)
    
#     # Add grid for easier reading
#     ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
#     # Style improvements
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
    
#     # Add event count to labels
#     total_events = sum(len(cat_events) for cat_events in categories.values())
#     unique_event_types = sum(len(event_details[cat]) for cat in sorted_categories)
#     fig.text(0.99, 0.01, 
#              f'Total Events: {total_events} | Unique Event Types: {unique_event_types}', 
#              ha='right', va='bottom', fontsize=10, style='italic', alpha=0.7)
    
#     plt.tight_layout()
    
#     # Save or show
#     if output_path:
#         plt.savefig(output_path, dpi=300, bbox_inches='tight')
#         print(f"✓ Saved timeline to: {output_path}")
#     else:
#         plt.show()

#     if dump_json and other_records:
#         import json

#         dump = {
#             "total_events": total_events,
#             "categories": {k: len(v) for k, v in categories.items()},
#             "other_events": other_records,
#         }
#         dump_json.write_text(json.dumps(dump, indent=2))
#         print(f"Saved OTHER event details to {dump_json}")
    
#     # Print summary statistics
#     print("\n" + "="*60)
#     print("EVENT SUMMARY")
#     print("="*60)
#     print(f"Total duration: {time_seconds[-1]:.2f} seconds ({time_seconds[-1]/60:.2f} minutes)")
#     print(f"Total events: {total_events}")
#     print(f"Unique event types used: {unique_event_types}")
#     print("\nEvents by category:")
#     for category in sorted_categories:
#         count = len(categories[category])
#         unique = len(event_details[category])
#         print(f"  {category:20s}: {count:4d} events ({unique} unique types)")
    
#     # Print all unique event types found (useful for debugging)
#     print("\nAll event types seen in this session:")
#     all_events = []
#     for category in sorted_categories:
#         all_events.extend(event_details[category])
#     for event in sorted(set(all_events)):
#         print(f"  - {event}")
#     print("="*60)


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="Visualize experiment timeline from zarr events",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   # Visualize latest run
#   python visualize_experiment_timeline.py experiment.zarr
  
#   # Visualize specific run
#   python visualize_experiment_timeline.py experiment.zarr --run-name run_20250101_120000
  
#   # Save to file
#   python visualize_experiment_timeline.py experiment.zarr -o timeline.png
#         """
#     )
    
#     parser.add_argument(
#         "zarr_path",
#         type=Path,
#         help="Path to the zarr file containing events"
#     )
    
#     parser.add_argument(
#         "--run-name",
#         type=str,
#         help="Specific stimulus run to visualize (default: latest)"
#     )
    
#     parser.add_argument(
#         "-o", "--output",
#         type=Path,
#         help="Output file path for the plot (if not specified, displays interactively)"
#     )
#     parser.add_argument(
#         "--dump-other-json",
#         type=Path,
#         help="Optional path to write OTHER-category events as JSON",
#     )
    
#     return parser.parse_args()


# def main():
#     args = parse_args()
    
#     # Open zarr file
#     print(f"Opening zarr file: {args.zarr_path}")
#     root = zarr.open(str(args.zarr_path), mode='r')
    
#     # Load enum mappings from the file
#     print("\nLoading enum mappings from HDF5...")
#     enum_mappings = load_enum_mappings(root)
    
#     # Get event type mappings specifically
#     event_type_mappings = enum_mappings.get('events', {})
#     if not event_type_mappings:
#         print("Warning: No event type mappings found. Events will show as UNKNOWN.")
#     else:
#         print(f"✓ Ready to decode {len(event_type_mappings)} event types")

#     stimulus_mode_mappings = enum_mappings.get('stimulus_modes', {})
#     if stimulus_mode_mappings:
#         print(f"✓ Loaded {len(stimulus_mode_mappings)} stimulus modes")
    
#     # Get available runs
#     available_runs = list_stimulus_runs(root)
#     if not available_runs:
#         raise ValueError("No stimulus runs found in zarr file")
    
#     # Select run
#     run_name = args.run_name if args.run_name else available_runs[-1]
#     if run_name not in available_runs:
#         raise ValueError(f"Run '{run_name}' not found. Available: {', '.join(available_runs)}")
    
#     print(f"\nLoading events from run: {run_name}")
    
#     # Load events
#     events_dict = load_events(root, run_name)
    
#     # Create visualization
#     plot_timeline(
#         events_dict,
#         event_type_mappings,
#         stimulus_mode_mappings,
#         args.output,
#         args.dump_other_json,
#     )


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Visualize experiment timeline from zarr events file.

Creates a color-coded timeline showing when different types of events occurred
during the experiment, helping to understand the progression and structure of
the experimental session.

UPDATED: Now reads event type mappings directly from HDF5 file, automatically
adapting to new event types without code changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.patches import Patch


def _resolve_enums_group(root: zarr.Group) -> Optional[zarr.Group]:
    analysis = root.get("analysis")
    if analysis is not None:
        enums = analysis.get("enums")
        if enums is not None:
            events_enums = enums.get("events")
            if events_enums is not None:
                return events_enums
    return root.get("enums")


def load_enum_mappings(root: zarr.Group) -> dict[str, dict[int, str]]:
    """
    Load enum mappings from the archive.
    This makes the script automatically adapt to any new event types or stimulus modes.
    
    Returns:
        Dictionary with keys 'events', 'stimulus_modes', etc. mapping IDs to names
    """
    mappings = {}
    
    # Try to load the enums group
    enums_group = _resolve_enums_group(root)
    if enums_group is None:
        print("Warning: No enums group found in archive. Using empty mappings.")
        return mappings
    
    # Load each enum mapping dataset
    for dataset_name in enums_group.array_keys():
        try:
            dataset = enums_group[dataset_name]
            data = dataset[:]
            
            # Convert to dictionary: id -> name
            mapping_dict = {}
            for record in data:
                enum_id = int(record['id'])
                enum_name = record['name']
                
                # Handle bytes vs string
                if isinstance(enum_name, bytes):
                    enum_name = enum_name.decode('utf-8', errors='ignore').rstrip('\x00')
                else:
                    enum_name = str(enum_name).rstrip('\x00')
                
                mapping_dict[enum_id] = enum_name
            
            mappings[dataset_name] = mapping_dict
            print(f"✓ Loaded {len(mapping_dict)} {dataset_name} mappings from HDF5")
            
        except Exception as e:
            print(f"Warning: Could not load enum mapping '{dataset_name}': {e}")
    
    return mappings


# Color scheme for different event categories
EVENT_COLORS = {
    "PROTOCOL": "#2E86AB",      # Blue - protocol level events
    "STEP": "#A23B72",           # Purple - step boundaries
    "ITI": "#8B8C89",            # Gray - inter-trial intervals
    "PRE_PERIOD": "#F18F01",     # Orange - pre-training
    "TRAINING": "#C73E1D",       # Red - training period
    "POST_PERIOD": "#6A994E",    # Green - post-training
    "CHASE": "#BC4749",          # Dark red - chase sequences
    "CHASER": "#BC4749",         # Dark red - chaser events
    "LOOM": "#E63946",           # Bright red - looming events
    "CAVE": "#9D4EDD",           # Purple - cave dweller behaviors
    "POSITIONING": "#FB8500",    # Orange - positioning phase
    "GRID": "#06FFA5",           # Cyan - grid stimulus events
    "PARAMS": "#FFC300",         # Yellow - parameter changes
    "ERROR": "#D00000",          # Dark red - errors
    "OTHER": "#4A5859",          # Dark gray - other events
}


def list_stimulus_runs(root: zarr.Group) -> list[str]:
    """List all available stimulus runs in the zarr file."""
    analysis = root.get("analysis")
    if analysis is None:
        return []
    stimulus_parent = analysis.get("stimulus_runs")
    if stimulus_parent is None:
        return []
    return sorted(
        name for name in stimulus_parent.group_keys() 
        if isinstance(stimulus_parent.get(name), zarr.Group)
    )


def load_events(root: zarr.Group, run_name: str) -> dict[str, np.ndarray]:
    """Load events from a stimulus run."""
    run_group = root[f"analysis/stimulus_runs/{run_name}"]
    
    if "events" not in run_group:
        raise ValueError(f"Run '{run_name}' does not contain an 'events' dataset.")
    
    events_node = run_group["events"]
    
    # Handle both structured array and group storage
    if isinstance(events_node, zarr.Array):
        events_array = events_node[:]
        field_names = events_array.dtype.names
    elif isinstance(events_node, zarr.Group):
        # Reconstruct from separate arrays
        field_names = list(events_node.array_keys())
        events_dict = {}
        for field in field_names:
            events_dict[field] = events_node[field][:]
        return events_dict
    else:
        raise TypeError("Unsupported Zarr node type for events dataset.")
    
    # Convert structured array to dictionary of arrays
    events_dict = {}
    for field in field_names:
        events_dict[field] = events_array[field]
    
    return events_dict


def categorize_event(event_name: str, mode_name: Optional[str] = None) -> str:
    """Categorize events using both event name/context and stimulus mode."""
    name_upper = (event_name or "").upper()
    mode_upper = (mode_name or "").upper()

    # Mode-first categorization
    if "CHASER" in mode_upper:
        # Further split chaser actions
        if any(keyword in name_upper for keyword in ("RETREAT", "CHASE", "POSITION", "LOOM")):
            if "LOOM" in name_upper:
                return "LOOM"
            if "RETREAT" in name_upper:
                return "CHASER_RETREAT"
            if "POSITION" in name_upper or "APPROACH" in name_upper:
                return "POSITIONING"
            if "CHASE" in name_upper:
                return "CHASER"
        return "CHASER"
    if "GRID" in mode_upper:
        return "GRID"
    if "LOOM" in mode_upper:
        return "LOOM"

    # Protocol management
    if "PROTOCOL" in name_upper:
        return "PROTOCOL"
    if "STEP" in name_upper:
        return "STEP"
    if "ITI" in name_upper:
        return "ITI"

    if "PRE_PERIOD" in name_upper:
        return "PRE_PERIOD"
    if "TRAINING" in name_upper:
        return "TRAINING"
    if "POST_PERIOD" in name_upper:
        return "POST_PERIOD"

    if "CAVE" in name_upper:
        return "CAVE"
    if "LOOM" in name_upper or "ESCAPE" in name_upper:
        return "LOOM"
    if "POSITION" in name_upper or "APPROACH" in name_upper:
        return "POSITIONING"
    if "CHASE" in name_upper or "CHASER" in name_upper:
        return "CHASER"

    if "PARAMS" in name_upper or "APPLIED" in name_upper:
        return "PARAMS"
    if "ERROR" in name_upper or "FAIL" in name_upper:
        return "ERROR"

    return "OTHER"


def diagnose_calibration_test_shape_events(
    events_dict: dict[str, np.ndarray],
    event_type_mappings: dict[int, str],
    stimulus_mode_mappings: dict[int, str],
):
    """
    Diagnostic function to analyze CALIBRATION_TEST_SHAPE events.
    Prints detailed information about when and why these events appear.
    """
    print("\n" + "="*80)
    print("CALIBRATION_TEST_SHAPE DIAGNOSTIC REPORT")
    print("="*80)
    
    # Get mode IDs
    mode_ids = events_dict.get('stimulus_mode_id')
    if mode_ids is None:
        print("⚠ No stimulus_mode_id field found in events")
        return
    
    # Find CALIBRATION_TEST_SHAPE mode ID (should be 13)
    calibration_mode_id = None
    for mode_id, mode_name in stimulus_mode_mappings.items():
        if "CALIBRATION_TEST_SHAPE" in mode_name:
            calibration_mode_id = mode_id
            break
    
    if calibration_mode_id is None:
        print("✓ CALIBRATION_TEST_SHAPE not found in stimulus mode mappings")
        print("  This is expected if you haven't used calibration shapes.")
        print("="*80 + "\n")
        return
    
    # Find all events with this mode
    calibration_mask = mode_ids == calibration_mode_id
    num_calibration_events = np.sum(calibration_mask)
    
    if num_calibration_events == 0:
        print(f"✓ No events found with CALIBRATION_TEST_SHAPE (mode ID {calibration_mode_id})")
        print("  Your chaser experiment data is clean!")
        print("="*80 + "\n")
        return
    
    print(f"⚠ Found {num_calibration_events} events with CALIBRATION_TEST_SHAPE mode")
    print(f"  Mode ID: {calibration_mode_id}")
    print()
    
    # Get timestamps
    timestamp_field = None
    for field_name in ['timestamp_ns_session', 'timestamp_ns_epoch', 
                       'relative_timestamp_ns', 'timestamp_ns']:
        if field_name in events_dict:
            timestamp_field = field_name
            break
    
    if timestamp_field is None:
        print("⚠ Could not find timestamp field")
        return
    
    timestamps = events_dict[timestamp_field]
    time_seconds = (timestamps / 1e9) - (timestamps[0] / 1e9)
    
    # Get event types
    event_type_field = None
    for field_name in ['event_type_id', 'event_type']:
        if field_name in events_dict:
            event_type_field = field_name
            break
    
    event_types = events_dict[event_type_field]
    
    # Get event names if available
    event_names = None
    for field_name in ['name_or_context', 'event_name']:
        if field_name in events_dict:
            event_names = events_dict[field_name]
            break
    
    # Print detailed information for each calibration event
    print("Detailed event listing:")
    print("-" * 80)
    
    calibration_indices = np.where(calibration_mask)[0]
    total_events = len(timestamps)
    
    for idx in calibration_indices:
        time_s = time_seconds[idx]
        event_type_id = event_types[idx]
        event_type_name = event_type_mappings.get(int(event_type_id), f"UNKNOWN_{event_type_id}")
        
        event_context = "N/A"
        if event_names is not None:
            context = event_names[idx]
            if isinstance(context, bytes):
                context = context.decode('utf-8', errors='ignore').rstrip('\x00')
            event_context = context
        
        # Calculate position in experiment
        position_pct = (idx / total_events) * 100
        
        print(f"  Event #{idx+1} (at {position_pct:.1f}% through experiment)")
        print(f"    Time: {time_s:.3f} seconds")
        print(f"    Event Type: {event_type_name} (ID: {event_type_id})")
        print(f"    Context: {event_context}")
        print()
    
    # Analysis and recommendations
    print("-" * 80)
    print("ANALYSIS:")
    
    first_idx = calibration_indices[0]
    last_idx = calibration_indices[-1]
    
    if first_idx < 10:
        print("  • Events appear at the START of the experiment")
        print("    → Likely from setup/calibration before starting protocol")
    
    if last_idx > total_events - 10:
        print("  • Events appear at the END of the experiment")
        print("    → Likely from cleanup/testing after protocol finished")
    
    if num_calibration_events <= 3:
        print("  • Only a few events affected")
        print("    → Minimal impact on main experimental data")
    
    print("\nRECOMMENDATION:")
    print("  These CALIBRATION_TEST_SHAPE events are preview-mode artifacts.")
    print("  To avoid them in future experiments:")
    print("    1. Close any calibration test shapes before starting logging")
    print("    2. Don't click 'Show Test Circle/Square' during active protocols")
    print("    3. These events won't affect your chaser state data or analysis")
    
    print("="*80 + "\n")


def plot_timeline(
    events_dict: dict[str, np.ndarray],
    event_type_mappings: dict[int, str],
    stimulus_mode_mappings: dict[int, str],
    output_path: Optional[Path] = None,
    dump_json: Optional[Path] = None,
):
    """
    Create a color-coded timeline visualization of events.
    
    Args:
        events_dict: Dictionary of event arrays
        event_type_mappings: Mapping from event_type_id to event name (from HDF5)
        output_path: Optional path to save the plot
    """
    
    # Get timestamps and event types
    # Try different possible timestamp field names
    timestamp_field = None
    for field_name in ['timestamp_ns_session', 'timestamp_ns_epoch', 
                       'relative_timestamp_ns', 'timestamp_ns']:
        if field_name in events_dict:
            timestamp_field = field_name
            break
    
    if timestamp_field is None:
        raise ValueError("Could not find timestamp field in events")
    
    timestamps = events_dict[timestamp_field]
    
    # Try different event type field names
    event_type_field = None
    for field_name in ['event_type_id', 'event_type']:
        if field_name in events_dict:
            event_type_field = field_name
            break
    
    if event_type_field is None:
        raise ValueError("Could not find event type field in events")
    
    event_types = events_dict[event_type_field]
    
    # Get event names if available
    event_names = None
    for field_name in ['name_or_context', 'event_name']:
        if field_name in events_dict:
            event_names = events_dict[field_name]
            break

    mode_ids = events_dict.get('stimulus_mode_id')
    mode_names: Optional[np.ndarray]
    if mode_ids is not None:
        mode_names = np.array([
            stimulus_mode_mappings.get(int(mid), 'UNKNOWN') for mid in mode_ids
        ])
    else:
        mode_names = None
    
    # Convert timestamps to seconds
    time_seconds = timestamps / 1e9
    
    # Normalize to start at 0
    time_seconds = time_seconds - time_seconds[0]
    
    # Group events by category
    categories = {}
    event_details = {}  # Store event type names for each category
    
    other_records = []
    for i, (time, event_type) in enumerate(zip(time_seconds, event_types)):
        # Get the actual event name from the mapping
        event_name = event_type_mappings.get(int(event_type), f"UNKNOWN_{event_type}")
        
        # Override with name_or_context if available (more specific context)
        if event_names is not None:
            context_name = event_names[i]
            if isinstance(context_name, bytes):
                context_name = context_name.decode('utf-8', errors='ignore').rstrip('\x00')
            if context_name:  # Only override if non-empty
                event_name = context_name
        
        mode_name = None
        if mode_names is not None:
            mode_name = mode_names[i]

        category = categorize_event(str(event_name), mode_name)
        
        if category == "OTHER":
            other_records.append(
                {
                    "time_s": float(time),
                    "event_type_id": int(event_type),
                    "event_name": event_name,
                    "mode": mode_name,
                    "context": event_name,
                }
            )

        if category not in categories:
            categories[category] = []
            event_details[category] = set()
        
        categories[category].append(time)
        event_details[category].add(event_name)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data for eventplot
    positions = []
    colors = []
    labels = []
    lineoffsets = []
    
    # Sort categories for consistent vertical ordering
    sorted_categories = sorted(categories.keys())
    
    for i, category in enumerate(sorted_categories):
        if len(categories[category]) > 0:  # Only add if there are events
            positions.append(categories[category])
            colors.append(EVENT_COLORS.get(category, EVENT_COLORS["OTHER"]))
            
            # Create label with count
            count = len(categories[category])
            labels.append(f"{category} ({count})")
            
            lineoffsets.append(i)
    
    # Plot the events
    if positions:
        ax.eventplot(
            positions,
            lineoffsets=lineoffsets,
            colors=colors,
            linewidths=2.5,
            linelengths=0.8,
            orientation='horizontal'
        )
    
    # Customize the plot
    ax.set_yticks(lineoffsets)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Event Category', fontsize=12, fontweight='bold')
    ax.set_title('Experiment Timeline: Event Progression', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add grid for easier reading
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # Style improvements
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add event count to labels
    total_events = sum(len(cat_events) for cat_events in categories.values())
    unique_event_types = sum(len(event_details[cat]) for cat in sorted_categories)
    fig.text(0.99, 0.01, 
             f'Total Events: {total_events} | Unique Event Types: {unique_event_types}', 
             ha='right', va='bottom', fontsize=10, style='italic', alpha=0.7)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved timeline to: {output_path}")
    else:
        plt.show()

    if dump_json and other_records:
        import json

        dump = {
            "total_events": total_events,
            "categories": {k: len(v) for k, v in categories.items()},
            "other_events": other_records,
        }
        dump_json.write_text(json.dumps(dump, indent=2))
        print(f"Saved OTHER event details to {dump_json}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("EVENT SUMMARY")
    print("="*60)
    print(f"Total duration: {time_seconds[-1]:.2f} seconds ({time_seconds[-1]/60:.2f} minutes)")
    print(f"Total events: {total_events}")
    print(f"Unique event types used: {unique_event_types}")
    print("\nEvents by category:")
    for category in sorted_categories:
        count = len(categories[category])
        unique = len(event_details[category])
        print(f"  {category:20s}: {count:4d} events ({unique} unique types)")
    
    # Print all unique event types found (useful for debugging)
    print("\nAll event types seen in this session:")
    all_events = []
    for category in sorted_categories:
        all_events.extend(event_details[category])
    for event in sorted(set(all_events)):
        print(f"  - {event}")
    print("="*60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize experiment timeline from zarr events",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize latest run
  python visualize_experiment_timeline.py experiment.zarr
  
  # Visualize specific run
  python visualize_experiment_timeline.py experiment.zarr --run-name run_20250101_120000
  
  # Save to file
  python visualize_experiment_timeline.py experiment.zarr -o timeline.png
        """
    )
    
    parser.add_argument(
        "zarr_path",
        type=Path,
        help="Path to the zarr file containing events"
    )
    
    parser.add_argument(
        "--run-name",
        type=str,
        help="Specific stimulus run to visualize (default: latest)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output file path for the plot (if not specified, displays interactively)"
    )
    parser.add_argument(
        "--dump-other-json",
        type=Path,
        help="Optional path to write OTHER-category events as JSON",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Open zarr file
    print(f"Opening zarr file: {args.zarr_path}")
    root = zarr.open(str(args.zarr_path), mode='r')
    
    # Load enum mappings from the file
    print("\nLoading enum mappings from HDF5...")
    enum_mappings = load_enum_mappings(root)
    
    # Get event type mappings specifically
    event_type_mappings = enum_mappings.get('events', {})
    if not event_type_mappings:
        print("Warning: No event type mappings found. Events will show as UNKNOWN.")
    else:
        print(f"✓ Ready to decode {len(event_type_mappings)} event types")

    stimulus_mode_mappings = enum_mappings.get('stimulus_modes', {})
    if stimulus_mode_mappings:
        print(f"✓ Loaded {len(stimulus_mode_mappings)} stimulus modes")
    
    # Get available runs
    available_runs = list_stimulus_runs(root)
    if not available_runs:
        raise ValueError("No stimulus runs found in zarr file")
    
    # Select run
    run_name = args.run_name if args.run_name else available_runs[-1]
    if run_name not in available_runs:
        raise ValueError(f"Run '{run_name}' not found. Available: {', '.join(available_runs)}")
    
    print(f"\nLoading events from run: {run_name}")
    
    # Load events
    events_dict = load_events(root, run_name)
    
    # Run diagnostic for CALIBRATION_TEST_SHAPE events
    diagnose_calibration_test_shape_events(
        events_dict,
        event_type_mappings,
        stimulus_mode_mappings
    )
    
    # Create visualization
    plot_timeline(
        events_dict,
        event_type_mappings,
        stimulus_mode_mappings,
        args.output,
        args.dump_other_json,
    )


if __name__ == "__main__":
    main()
