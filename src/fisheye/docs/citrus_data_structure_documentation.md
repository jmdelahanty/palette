# Citrus Data Structure Documentation
## Complete Reference for Data Ingestion and Analysis

---

## 1. Overview

This document provides exhaustive documentation of the Citrus experimental system's data structures, stimulus types, trial types, event types, and logged fields. Use this as a reference when building data ingestion and analysis tools.

### Key Data Files
- **HDF5 Files**: Primary experimental data storage
- **Protocol JSON Files**: Experimental protocol definitions
- **Arena Config Files**: Calibration and hardware setup

---

## 2. Stimulus Modes (Trial Types)

All stimulus modes are defined in `StimulusMode::Type` enum:

### 2.1 Available Stimulus Types

| ID | Name | Description | Use Case |
|----|------|-------------|----------|
| -1 | UNDEFINED | Uninitialized/error state | - |
| 2 | COHERENT_DOTS | Moving dots with coherent motion | Motion perception studies |
| 3 | MOVING_GRATING | Drifting sinusoidal grating | Spatial frequency, orientation tuning |
| 4 | SOLID_BLACK | Uniform black screen | Baseline, ITI periods |
| 5 | SOLID_WHITE | Uniform white screen | Contrast control |
| 6 | CONCENTRIC_GRATING | Radial expanding/contracting grating | Looming controls |
| 7 | LOOMING_DOT | Simple expanding circle | Basic looming response |
| 8 | STATIC_IMAGE | Display static image file | Custom stimuli |
| 9 | CALIBRATION_GRID | Calibration dot pattern | Setup only |
| 10 | ARENA_DEFINITION_SQUARE | Sub-arena boundary marker | Visualization |
| 11 | SPOTLIGHT | Reactive spotlight following target | Attention, tracking |
| 12 | CHASER | Complex looming/chasing behavior | Predator avoidance, detailed below |
| 13 | CALIBRATION_TEST_SHAPE | Test shape at specific mm size | Validation |
| 14 | SCROLLING_GRID | Grid of images that scroll | Optic flow |
| 15 | INDEPENDENT_MOTION_GRID | Grid with independent segment motion | Complex motion patterns |
| 16 | MOVING_DOTS | Dots spawning and moving across screen | Prey-like stimuli |
| 99 | NONE | No stimulus | - |

---

## 3. Protocol Parameter Structures

### 3.1 ProtocolMovingGratingParams
```
Fields:
- spatial_freq_cpp (float): Spatial frequency in cycles per pixel [computational]
- spatial_freq_cycles_per_mm (float): Spatial frequency in cycles/mm [authoritative]
- spatial_freq_cycles_per_cm (float): Spatial frequency in cycles/cm [UI display, computed]
- speed_pps (float): Speed in pixels per second [computational]
- speed_mm_per_sec (float): Speed in mm/s [authoritative, portable]
- orientation_degrees (float): Grating orientation (0-360°)
- duty_cycle (float): Light/dark ratio (0.0-1.0)
- line_color_imgui (ImVec4): Color of light bars (RGBA, 0-1)
- bg_color_imgui (ImVec4): Color of dark bars (RGBA, 0-1)
- brightness (float): Overall brightness multiplier (0-1)
- reactive_logic_module_name (string): Name of reactive behavior module (e.g., "OrientationMirrorsXPosition" or "NONE")

Units: Both pixel and mm values stored; mm is authoritative for portability
```

### 3.2 ProtocolCoherentDotsParams
```
Fields:
- num_dots (int): Number of dots to render
- orientation_degrees (float): Motion direction (0-360°)
- speed_px_per_sec (float): Speed in pixels per second [computational]
- speed_mm_per_sec (float): Speed in mm/s [authoritative]
- dot_radius_px (float): Dot radius in pixels [computational]
- dot_radius_mm (float): Dot radius in mm [authoritative]
- dot_color (ImVec4): Dot color (RGBA, 0-1)
- bg_color (ImVec4): Background color (RGBA, 0-1)

Units: Both pixel and mm values; mm is authoritative
```

### 3.3 ProtocolSolidColorParams
```
Fields:
- color_type (enum): Black or White
  * SolidColorType::Black (default)
  * SolidColorType::White
```

### 3.4 ProtocolLoomingDotParams
```
Fields:
- start_radius_px (float): Initial radius in pixels
- end_radius_px (float): Final radius in pixels
- loom_duration_sec (float): Duration of one loom cycle
- dot_color_imgui (ImVec4): Dot color (RGBA)
- bg_color_imgui (ImVec4): Background color (RGBA)
- target_side (int): Position target
  * 0 = Center
  * 1 = Left
  * 2 = Right
- auto_repeat_loom (bool): Whether to repeat looming
- inter_loom_interval_sec (float): Wait time between looms (if auto_repeat enabled)
```

### 3.5 ProtocolConcentricGratingParams
```
Fields:
- spatial_freq_rpp (float): Radial cycles per pixel [computational]
- spatial_freq_cycles_per_mm (float): Radial cycles per mm [authoritative]
- spatial_freq_cycles_per_cm (float): Radial cycles per cm [computed]
- speed_pps (float): Expansion speed in pixels/s [computational]
- speed_mm_per_sec (float): Expansion speed in mm/s [authoritative]
- is_expanding (bool): true=expanding, false=contracting
- duty_cycle (float): Light/dark ratio
- line_color_imgui (ImVec4): Ring color
- bg_color_imgui (ImVec4): Background color
```

### 3.6 ProtocolStaticImageParams
```
Fields:
- image_path (string): Path to image file
- brightness (float): Image brightness multiplier
```

### 3.7 ProtocolSpotlightParams
```
Fields:
- radius_px (float): Spotlight radius in pixels [computational]
- radius_mm (float): Spotlight radius in mm [authoritative]
- center_x_px (float): X center in pixels (-1 = use center)
- center_y_px (float): Y center in pixels (-1 = use center)
- center_x_mm (float): X center in mm (-1 = use center) [authoritative]
- center_y_mm (float): Y center in mm (-1 = use center) [authoritative]
- color_imgui (ImVec4): Spotlight color
- bg_color_imgui (ImVec4): Background color
- reactive_logic_module_name (string): Reactive behavior module
```

### 3.8 ProtocolChaserParams (Complex Structure)
```
Main Fields:
- chasers (vector<ChaserProperties>): List of individual chaser agents (see 3.8.1)
- danger_zone_enabled (bool): Whether danger zone is active
- danger_zone_x_px, danger_zone_y_px (float): Danger zone center (-1 = center)
- danger_zone_width_mm, danger_zone_height_mm (float): Real-world size [authoritative]
- danger_zone_width_px, danger_zone_height_px (float): Pixel size [computational]
- draw_danger_zone (bool): Whether to visualize danger zone
- danger_zone_color (ImVec4): Visualization color
- position_transition_duration_s (float): Smooth movement duration between states
- enable_smooth_transitions (bool): Enable/disable smooth transitions
- show_target_dot (bool): Whether to show tracking target
- target_radius_mm (float): Target radius in mm [authoritative]
- target_radius_px (float): Target radius in pixels [computational]
- target_color (ImVec4): Target visualization color
- bg_color (ImVec4): Background color
- pre_period_duration_s (float): Duration of pre-training period
- training_period_duration_s (float): Duration of main training
- post_period_duration_s (float): Duration of post-training period
- chase_probability_per_second (float): % chance per second of initiating chase (during training)
- chase_duration_s (float): How long each chase lasts
- target_box_index (int): Which bounding box to track (from camera)
- pre_period_position (ImVec2): Position during pre-period (-1,-1 = center)
- post_period_position (ImVec2): Position during post-period (-1,-1 = center)
- enable_proximity_feedback (bool): Whether proximity affects chaser
- proximity_threshold_px (float): Distance threshold for proximity effects
- proximity_threshold_mm (float): Distance threshold in mm [authoritative]
- pixels_per_mm (float): Calibration ratio
- z_eff_mm (float): Effective viewing distance through media
```

### 3.8.1 ChaserProperties (Individual Chaser Agent)
```
Behavior Flags:
- enable_random_movement (bool): Enable random jumps
- pause_at_random_target (bool): Pause upon reaching random target
- stop_at_target_edge (bool): Stop at experimental area boundary

Visual Properties:
- radius_px (float): Chaser radius in pixels [computational]
- radius_mm (float): Chaser radius in mm [authoritative]
- color (ImVec4): Chaser color

Movement/Physics:
- speed_pps (float): Speed in pixels/s [computational]
- speed_mm_per_sec (float): Speed in mm/s [authoritative]
- random_jump_interval_s (float): Time between random jumps
- random_jump_min_distance, random_jump_max_distance (float): Jump range in pixels
- random_jump_min_distance_mm, random_jump_max_distance_mm (float): Jump range in mm [authoritative]

Looming Parameters:
- loom_mode (enum): Looming behavior type
  * 0 = FIXED_SIZE: Constant radius
  * 1 = PROXIMITY_SCALING: Size scales with proximity to target
  * 2 = VISUAL_ANGLE_LOOM: Biologically accurate l/v ratio looming (moving)
  * 3 = STATIONARY_LOOM: Biologically accurate l/v ratio looming (stationary)
  * 4 = CAVE_DWELLER_DEFENSIVE: Hides until target approaches, then looms defensively
  * 5 = CAVE_DWELLER_AGGRESSIVE: Hides until target approaches, then chases
  
- l_over_v_ms (float): Size-to-speed ratio (lambda) in milliseconds
- initial_distance_mm (float): Starting distance for loom (D_0)
- trigger_angle_deg (float): Visual angle threshold for escape trigger (~20°)
- max_angle_deg (float): Maximum visual angle allowed (~48°)
- positioning_distance_mm (float): Distance to move before starting loom
- positioning_speed_mm_per_sec (float): Speed while positioning
- chase_start_time (float): Timestamp when loom started [runtime state]
- chase_start_distance_mm (float): Actual distance when loom started [runtime state]
- is_positioning (bool): Currently moving to loom start position [runtime state]
- position_reached (bool): Ready to start looming [runtime state]
- is_retreating (bool): In retreat phase after loom [runtime state]
- retreat_start_time (float): When retreat began [runtime state]
- retreat_start_radius_px (float): Radius when retreat started [runtime state]

Retreat Parameters:
- retreat_duration_s (float): How long to shrink during retreat
- retreat_distance_mm (float): Distance to back away

Cave Dweller Parameters:
- cave_center_x_px, cave_center_y_px (float): Cave position (-1 = center)
- cave_visible_radius_px (float): Visible size when hiding
- cave_trigger_radius_px (float): Distance at which to emerge/chase
- cave_resting_radius_px (float): Small size while hiding
- cave_emergence_distance_px (float): How far to emerge
- cave_return_threshold_px (float): Distance to consider "back in cave"
- cave_emerge_duration_s (float): Time to emerge
- cave_emerge_speed_multiplier (float): Speed multiplier during emergence
- cave_retreat_radius_multiplier (float): Size multiplier during cave defense
- use_visual_angle_for_defense (bool): Use visual angle calculation
- cave_defensive_mode (enum):
  * 0 = VISUAL_ANGLE: Visual angle-based sizing
  * 1 = LINEAR_THREAT: Linear proximity scaling
  * 2 = BREATHING: Pulsing/breathing pattern
- breathing_frequency_hz (float): Breathing rate
- breathing_amplitude_ratio (float): Size variation during breathing
```

### 3.9 ProtocolScrollingGridParams
```
Fields:
- grid_rows, grid_cols (int): Grid dimensions
- motion_speed_px_per_sec (float): Scroll speed in pixels/s
- motion_speed_mm_per_sec (float): Scroll speed in mm/s [authoritative]
- motion_direction_degrees (float): Scroll direction
- image_paths (vector<string>): Paths to images for each cell
- brightness (float): Image brightness
- background_color (ImVec4): Background color
- border_color (ImVec4): Grid border color
- show_borders (bool): Whether to show grid lines
```

### 3.10 ProtocolIndependentMotionGridParams
```
Fields:
- grid_rows, grid_cols (int): Grid dimensions
- moving_segments (vector<int>): Which grid segments move
- motion_speed_px_per_sec (float): Speed for moving segments
- motion_speed_mm_per_sec (float): Speed in mm/s [authoritative]
- motion_direction_degrees (float): Motion direction for moving segments
- image_paths (vector<string>): Images for grid cells
- brightness (float): Brightness multiplier
- background_color, border_color (ImVec4): Colors
- show_borders (bool): Show grid lines
```

### 3.11 ProtocolMovingDotsParams
```
Fields:
- dot_radius_mm (float): Dot radius in mm [authoritative]
- dot_radius_px (float): Dot radius in pixels [computational]
- dot_speed_mm_per_sec (float): Speed in mm/s [authoritative]
- dot_speed_px_per_sec (float): Speed in pixels/s [computational]
- use_uniform_direction (bool): All dots move same direction
- uniform_direction_angle (float): Direction if uniform (degrees)
- num_simultaneous_dots (int): Max concurrent dots
- spawn_interval_s (float): Time between spawning new dots
- dot_color (ImVec4): Dot color
- bg_color (ImVec4): Background color
- spawn_side (enum): Where dots spawn from
  * Random
  * Top
  * Right
  * Bottom
  * Left
```

---

## 4. Event Types

All events are defined in `ExperimentEventType::Type` enum and logged to HDF5.

### 4.1 Core Protocol Events (0-23)

| ID | Name | Description | When Logged | Details JSON |
|----|------|-------------|-------------|--------------|
| 0 | PROTOCOL_START | Protocol execution begins | Protocol starts | N/A |
| 1 | PROTOCOL_STOP | Protocol manually stopped | User stops | N/A |
| 2 | PROTOCOL_PAUSE | Protocol paused | User pauses | N/A |
| 3 | PROTOCOL_RESUME | Protocol resumed | User resumes | N/A |
| 4 | PROTOCOL_FINISH | Protocol completed normally | All steps done | N/A |
| 5 | PROTOCOL_CLEAR | Protocol cleared from queue | User clears | N/A |
| 6 | PROTOCOL_LOAD | Protocol loaded from file | File loaded | `{"filepath": "..."}` |
| 7 | STEP_ADD | Step added to protocol | User adds step | N/A |
| 8 | STEP_REMOVE | Step removed | User removes | N/A |
| 9 | STEP_MOVE_UP | Step moved up | User reorders | N/A |
| 10 | STEP_MOVE_DOWN | Step moved down | User reorders | N/A |
| 11 | STEP_START | Protocol step begins | Step starts | `{"step_index": N, "step_name": "..."}` |
| 12 | STEP_END | Protocol step ends | Step ends | `{"step_index": N, "duration": N}` |
| 13 | ITI_START | Inter-trial interval starts | Between steps | `{"duration": N}` |
| 14 | ITI_END | Inter-trial interval ends | ITI completes | N/A |
| 15 | PARAMS_APPLIED | Parameters applied to stimulus | Params updated | JSON of params |
| 16 | MANAGER_REINIT | Stimulus manager reinitialized | Manager restart | N/A |
| 17 | MANAGER_REINIT_FAIL | Reinitialization failed | Error occurred | `{"error": "..."}` |
| 18 | LOOM_AUTO_REPEAT_TRIGGER | Auto-repeat loom triggered | Timer expired | N/A |
| 19 | LOOM_MANUAL_START | Manual loom triggered | User clicks | N/A |
| 20 | USER_INTERVENTION | User manual intervention | Various | `{"action": "..."}` |
| 21 | ERROR_RUNTIME | Runtime error occurred | Error caught | `{"error": "..."}` |
| 22 | LOG_MESSAGE | Generic log entry | Various | `{"message": "..."}` |
| 23 | IPC_BOUNDING_BOX_RECEIVED | Bounding box data received | IPC message | See BoundingBox log |

### 4.2 Chaser-Specific Events (24-48)

| ID | Name | Description | When Logged | Details JSON Fields |
|----|------|-------------|-------------|---------------------|
| 24 | CHASER_PRE_PERIOD_START | Pre-training period begins | Trial starts | N/A |
| 25 | CHASER_TRAINING_START | Training period begins | After pre-period | N/A |
| 26 | CHASER_POST_PERIOD_START | Post-training begins | After training | N/A |
| 27 | CHASER_CHASE_SEQUENCE_START | Chase initiated | Random trigger or danger zone | `target_pos_x, target_pos_y, pre_chase_dist_px, in_danger_zone` |
| 28 | CHASER_CHASE_SEQUENCE_END | Chase sequence ends | Duration expires or target leaves | `reason, duration_s` |
| 29 | CHASER_POSITION_TRANSITION_START | Moving between period positions | State change | `target_state, from_x, from_y, to_x, to_y, distance_px, duration_s` |
| 30 | CHASER_POSITION_TRANSITION_END | Reached target position | Transition complete | `target_state` |
| 31 | CHASER_AT_PRE_POSITION | Reached pre-period position | Position reached | N/A |
| 32 | CHASER_AT_POST_POSITION | Reached post-period position | Position reached | N/A |
| 33 | CHASER_POSITIONING_START | Starting loom positioning | Loom mode active | `chaser_index, target_distance_mm` |
| 34 | CHASER_POSITIONING_END | Reached loom start position | Position reached | `chaser_index` |
| 35 | CHASER_APPROACHING | Chaser moving toward target | During loom | `chaser_index` |
| 36 | CHASER_LOOM_START | Loom expansion begins | Position reached | `chaser_index, start_distance_mm, loom_mode, l_over_v_ms, max_angle_deg` |
| 37 | CHASER_LOOM_MAX_SIZE | Maximum size reached | Max angle hit | `chaser_index, max_radius_px, visual_angle_deg` |
| 38 | CHASER_ESCAPE_TRIGGERED | Escape threshold crossed | Visual angle > threshold | `chaser_index, visual_angle_deg, threshold_deg` |
| 39 | CHASER_RETREAT_START | Retreat phase begins | After loom | `chaser_index` |
| 40 | CHASER_RETREAT_END | Retreat complete | Shrink done | `chaser_index` |
| 41 | CHASER_RANDOM_TARGET_SET | New random target chosen | Random mode | `target_x, target_y, distance` |
| 42 | CHASER_RANDOM_TARGET_REACHED | Random target reached | Reached destination | N/A |
| 43 | CHASER_CAVE_DEFENSE_START | Cave defense mode begins | Target approaches | `chaser_index, trigger_distance` |
| 44 | CHASER_CAVE_DEFENSE_END | Cave defense ends | Target leaves | `chaser_index` |
| 45 | CHASER_CAVE_EMERGE_START | Emerging from cave | Cave dweller mode | `chaser_index` |
| 46 | CHASER_CAVE_APPROACHING | Chasing from cave | Aggressive mode | `chaser_index` |
| 47 | CHASER_CAVE_RETURN_START | Returning to cave | After emergence | `chaser_index` |
| 48 | CHASER_CAVE_RETURN_END | Back in cave | Return complete | `chaser_index` |

---

## 5. HDF5 Data Structures

### 5.1 Event Log Structure
**Dataset**: `/events`

```
Fields:
- relative_timestamp_ns (int64): Nanoseconds since session start
- event_type (int32): Event type ID (see Event Types above)
- step_index (int32): Which protocol step (-1 if not in step)
- event_name (string, 256 chars): Human-readable event name
- stimulus_mode (int32): Active stimulus mode ID
- details (string, 1024 chars): JSON string with event-specific details
- stimulus_frame_num (uint64): Frame counter for stimulus rendering
- camera_frame_id (uint64): Frame ID from camera/tracker system
```

### 5.2 Chaser State Log Structure
**Dataset**: `/chaser_states`

Logged at every frame during CHASER stimulus mode.

```
Fields:
- relative_timestamp_ns (int64): Time since session start
- frame_number (uint64): Stimulus frame counter
- camera_frame_id (uint64): Corresponding camera frame

Position/Target:
- chaser_index (int32): Which chaser (if multiple)
- pos_x_px (float): Chaser X position in pixels
- pos_y_px (float): Chaser Y position in pixels
- target_x_px (float): Target X position (fish/tracking)
- target_y_px (float): Target Y position
- target_visible (bool): Whether target is being tracked

Size/Appearance:
- current_radius_px (float): Current chaser radius

Distance:
- distance_to_target_px (float): Distance to target in pixels
- distance_to_target_mm (float): Distance to target in millimeters

Speed/Velocity:
- chase_speed_px_per_s (float): Current speed in pixels/s
- chase_speed_mm_per_s (float): Current speed in mm/s

Visual Angle Metrics (Looming):
- visual_angle_deg (float): Current visual angle at fish eye
- angular_velocity_deg_s (float): Rate of angular expansion (dθ/dt)
- tau_ms (float): Time to collision in milliseconds

Loom Parameters:
- loom_mode (uint8): Loom mode (0=FIXED, 1=PROXIMITY, 2=VISUAL_ANGLE, 3=STATIONARY, 4=CAVE_DEFENSIVE, 5=CAVE_AGGRESSIVE)
- loom_phase (uint8): Phase (0=idle, 1=positioning, 2=looming, 3=retreating)
- l_over_v_ms (float): Size-to-speed ratio (lambda)
- initial_distance_mm (float): Starting distance for loom
- max_angle_deg (float): Maximum visual angle limit

Calibration:
- z_eff_mm (float): Effective viewing distance through media
- pixels_per_mm (float): Display calibration ratio

Trial State:
- trial_state (uint8): 0=PRE_PERIOD, 1=TRAINING, 2=POST_PERIOD
- chase_sequence_active (bool): Whether chase is currently active
- time_in_state_s (float): Time elapsed in current state
```

### 5.3 Bounding Box Log Structure
**Dataset**: `/bounding_boxes`

Logged when bounding box data received from tracker.

```
Fields:
- payload_timestamp_ns_epoch (int64): Timestamp from tracking system
- received_timestamp_ns_epoch (int64): When received by Citrus
- payload_frame_id (uint64): Frame ID from tracker
- payload_camera_id (uint16): Which camera
- box_index_in_payload (uint8): Index if multiple boxes

Box Data:
- x_min (float): Left edge
- y_min (float): Top edge
- width (float): Box width
- height (float): Box height
- class_id (uint16): Detection class
- confidence (float): Detection confidence
```

### 5.4 Frame Metadata Structure
**Dataset**: `/frame_metadata`

Logs frame timing information.

```
Fields:
- stimulus_frame_num (uint64): Stimulus frame counter
- triggering_camera_frame_id (uint64): Which camera frame triggered this
- timestamp_relative_ns (int64): Time since session start
```

### 5.5 Stimulus Coordinates Structure
**Group**: `/stimulus_coordinates`

Saved once per session with texture dimensions.

```
Attributes:
- texture_width_px (int): Stimulus texture width
- texture_height_px (int): Stimulus texture height
- texture_origin (string): Coordinate origin ("top_left")
- coordinate_system (string): "pixels"
```

### 5.6 Session Info Structure
**Group**: `/session_info`

Metadata about the recording session.

```
Attributes:
- session_start_time_epoch_ns (int64): Absolute start time
- session_start_time_human_readable (string): Human-readable timestamp
- arena_config_name (string): Which arena configuration used
- protocol_name (string): Protocol name
- experimenter (string): Who ran the experiment
- subject_id (string): Subject identifier
- notes (string): Experimental notes
```

---

## 6. Coordinate Systems and Calibration

### 6.1 Coordinate System Notes
- **Stimulus Coordinates**: Origin at top-left, units in pixels
- **Projector Space**: 1920×1080 typical, calibrated with pixels_per_mm
- **Camera Space**: May have different resolution, requires homography
- **Real-World Space**: Millimeters, affected by z_eff (refractive distortion)

### 6.2 Key Calibration Parameters
```
pixels_per_mm: Projector calibration (px/mm)
z_eff_mm: Effective viewing distance accounting for refraction
  - Typical: ~10.4mm (5mm acrylic shelf + 5mm dish bottom)
  - Calculated from: z_eff = eye_height + (n_water/n_acrylic) * total_acrylic_thickness

Units Conversion:
- mm → pixels: value_mm * pixels_per_mm
- pixels → mm: value_px / pixels_per_mm
- Visual angle: θ = 2 * arctan(radius_mm / z_eff_mm)
```

### 6.3 Arena Configuration
```
experimental_area_shape: CIRCLE or RECTANGLE
experimental_area_center_x_px, experimental_area_center_y_px: Center position
experimental_area_radius_px: Radius if circular
experimental_area_width_px, experimental_area_height_px: Dimensions if rectangular
(All have corresponding _mm fields for real-world reference)

danger_zone (chaser-specific): Rectangular region that can trigger chases
```

---

## 7. Trial State Machine (Chaser Trials)

### 7.1 Top-Level States
```
PRE_PERIOD (0):
  - Chaser at pre_period_position
  - No chases triggered
  - Duration: pre_period_duration_s
  - Target tracking active but no responses

TRAINING (1):
  - Chaser can initiate chases
  - Chase probability checked each frame
  - Danger zone active (if enabled)
  - Duration: training_period_duration_s
  - Main experimental period

POST_PERIOD (2):
  - Chaser at post_period_position
  - No chases triggered
  - Duration: post_period_duration_s
  - Follow-up observation
```

### 7.2 Chase Sequence States (During Training)
```
IDLE:
  - No chase active
  - Chaser at designated position or random movement
  - Checking for chase trigger

POSITIONING (Loom modes only):
  - Moving to loom start position
  - Distance: initial_distance_mm from target
  - Speed: positioning_speed_mm_per_sec
  - Event: CHASER_POSITIONING_START → CHASER_POSITIONING_END

LOOMING:
  - Expanding according to loom mode
  - Visual angle increasing
  - Tau decreasing
  - Events: CHASER_LOOM_START, CHASER_APPROACHING, 
           CHASER_LOOM_MAX_SIZE, CHASER_ESCAPE_TRIGGERED

RETREATING:
  - Shrinking back to base size
  - Moving away from target
  - Duration: retreat_duration_s
  - Event: CHASER_RETREAT_START → CHASER_RETREAT_END
```

### 7.3 Cave Dweller States
```
HIDING:
  - At cave_center position
  - Size: cave_resting_radius_px
  - Waiting for target approach

DEFENSIVE (CAVE_DWELLER_DEFENSIVE):
  - Target within cave_trigger_radius_px
  - Expanding using defensive_mode
  - Event: CHASER_CAVE_DEFENSE_START

EMERGING (CAVE_DWELLER_AGGRESSIVE):
  - Moving out from cave
  - Speed: cave_emerge_speed_multiplier
  - Event: CHASER_CAVE_EMERGE_START

CHASING (From cave):
  - Pursuing target
  - Event: CHASER_CAVE_APPROACHING

RETURNING:
  - Moving back to cave_center
  - Event: CHASER_CAVE_RETURN_START → CHASER_CAVE_RETURN_END
```

---

## 8. Data Analysis Recommendations

### 8.1 Key Analysis Workflows

**Basic Trial Structure:**
1. Load HDF5 file
2. Read `/events` dataset
3. Find STEP_START and STEP_END events to identify trial boundaries
4. Match stimulus_mode to determine trial type
5. Read corresponding state data for that stimulus type

**Chaser Analysis:**
1. Filter events for CHASER_* event types
2. Load `/chaser_states` dataset for continuous data
3. Segment by trial_state (PRE/TRAINING/POST)
4. Identify chase sequences (CHASE_SEQUENCE_START/END pairs)
5. Analyze visual angle, tau, angular velocity during looms

**Behavioral Correlation:**
1. Load `/bounding_boxes` for fish position
2. Use camera_frame_id to sync with stimulus_frame_num in events
3. Calculate distance to chaser, direction of movement
4. Correlate with chase initiation and escape responses

### 8.2 Important Fields for Common Analyses

**Escape Response:**
- Visual angle at escape (from chaser_states)
- Angular velocity at escape
- Distance to chaser
- Fish velocity (from bbox positions)
- Escape latency (time from LOOM_START to escape)

**Loom Characterization:**
- l_over_v_ms (lambda parameter)
- initial_distance_mm (D_0)
- tau profile over time
- Angular expansion rate
- Max visual angle reached

**Trial Effects:**
- Habituation: Compare PRE vs POST periods
- Learning: Response changes across training trials
- Context: Danger zone vs. random chases

### 8.3 Common Pitfalls

1. **Frame Synchronization**: Always use camera_frame_id to match tracking data with stimulus events, not timestamps alone
2. **Unit Consistency**: Some old protocols may have only pixel values; check for presence of _mm fields
3. **Missing Data**: Bounding boxes may be absent if fish not detected; check target_visible flag
4. **State Transitions**: Events may be queued; actual timing is in relative_timestamp_ns
5. **Multiple Chasers**: Check chaser_index field to distinguish multiple agents

---

## 9. Protocol File Format (JSON)

Protocol files are saved as JSON with this structure:

```json
{
  "protocol_name": "ExampleProtocol",
  "steps": [
    {
      "name": "Step1",
      "stimulus_mode_id": 12,
      "stimulus_mode_str": "CHASER",
      "duration_seconds": 180.0,
      "post_stimulus_iti_seconds": 30.0,
      "parameters": {
        "type": "ProtocolChaserParams",
        "chasers": [
          {
            "loom_mode": 2,
            "l_over_v_ms": 90.0,
            "initial_distance_mm": 20.0,
            ...
          }
        ],
        "pre_period_duration_s": 30.0,
        "training_period_duration_s": 120.0,
        "post_period_duration_s": 30.0,
        ...
      }
    }
  ]
}
```

---

## 10. Advice for the Data Ingestion Agent

### 10.1 Read Order
1. **Session Info** (`/session_info`): Get metadata, arena config, protocol name
2. **Stimulus Coordinates** (`/stimulus_coordinates`): Get texture dimensions
3. **Events** (`/events`): Understand trial structure, find step boundaries
4. **Stimulus-Specific State Data**: Load relevant continuous data
   - `/chaser_states` for CHASER trials
   - More may be added for other stimuli
5. **Bounding Boxes** (`/bounding_boxes`): Fish tracking data
6. **Frame Metadata** (`/frame_metadata`): Frame timing

### 10.2 Handling Multiple Trial Types
- Check `stimulus_mode` field in each event
- Different stimuli have different state data
- Use `event_type` to understand trial phases
- Not all stimuli log continuous state data

### 10.3 Validation Checks
- Ensure frame numbers are monotonically increasing
- Check for gaps in camera_frame_id sequence
- Verify event timestamps are chronological
- Confirm step durations match protocol definition
- Check z_eff_mm and pixels_per_mm are non-zero for accurate spatial measurements

### 10.4 Building Data Structures
Recommended approach:
```python
class CitrusSession:
    def __init__(self, hdf5_path):
        self.load_metadata()
        self.load_events()
        self.identify_trials()
        self.load_trial_data()
    
    def load_trial_data(self):
        # For each trial, load appropriate dataset
        # based on stimulus_mode
        pass

class CitrusTrial:
    trial_type: str  # e.g., "CHASER", "MOVING_GRATING"
    step_index: int
    start_time_ns: int
    end_time_ns: int
    parameters: dict  # Trial-specific parameters
    events: list      # Events during this trial
    state_data: pd.DataFrame  # Continuous state data
    tracking_data: pd.DataFrame  # Bounding boxes
```

### 10.5 Performance Tips
- Use HDF5 chunking for large datasets
- Filter events by event_type before loading full dataset
- Use frame_id as index for fast lookups
- Downsample high-frequency state data if needed for overview analyses
- Cache computed metrics (visual angles, velocities) 

### 10.6 Edge Cases
- **Empty trials**: Some steps may have no events (e.g., solid black screen)
- **Failed trials**: Check for ERROR_RUNTIME events
- **Interrupted sessions**: PROTOCOL_STOP before PROTOCOL_FINISH
- **Multiple protocols**: Session may contain multiple loaded protocols
- **Calibration trials**: May have CALIBRATION_GRID mode, skip for analysis

---

## 11. Example Event Sequence

Here's a typical CHASER trial event sequence:

```
TIME | EVENT_TYPE | EVENT_NAME | DETAILS
-----|------------|------------|--------
0ns  | PROTOCOL_START | Protocol Start | {"protocol": "ChaserTest"}
100ms | STEP_START | Step 1 Start | {"step_index": 0, "name": "ChaserTrial"}
150ms | PARAMS_APPLIED | Parameters Applied | {ChaserParams JSON}
200ms | CHASER_PRE_PERIOD_START | Pre-Period Start | {}
30.2s | CHASER_TRAINING_START | Training Start | {}
35.4s | CHASER_CHASE_SEQUENCE_START | Chase Initiated | {"target_pos_x": 500, "target_pos_y": 300, ...}
35.4s | CHASER_POSITIONING_START | Positioning Start | {"chaser_index": 0, "target_distance_mm": 20}
36.1s | CHASER_POSITIONING_END | Position Reached | {"chaser_index": 0}
36.1s | CHASER_LOOM_START | Loom Started | {"start_distance_mm": 20, "l_over_v_ms": 90}
36.8s | CHASER_ESCAPE_TRIGGERED | Escape Triggered | {"visual_angle_deg": 21.3}
37.2s | CHASER_LOOM_MAX_SIZE | Max Size Reached | {"max_radius_px": 100}
37.5s | CHASER_RETREAT_START | Retreat Started | {}
38.5s | CHASER_RETREAT_END | Retreat Complete | {}
40.5s | CHASER_CHASE_SEQUENCE_END | Chase Ended | {"reason": "DURATION", "duration_s": 5.1}
150.2s | CHASER_POST_PERIOD_START | Post-Period Start | {}
180.2s | STEP_END | Step 1 End | {"duration": 180.1}
180.2s | PROTOCOL_FINISH | Protocol Complete | {}
```

---

## 12. Version History & Compatibility

### Current Schema Version: 2.0

**Breaking Changes:**
- Frame synchronization added (stimulus_frame_num, camera_frame_id)
- Millimeter units added as authoritative (older files pixel-only)
- Chaser state logging expanded with visual angle metrics

**Backward Compatibility:**
- Old protocols will have only pixel values; estimate mm values using pixels_per_mm
- Events without frame IDs should be treated as unsynchronized
- Missing chaser_states dataset indicates pre-v2.0 file

**Forward Compatibility:**
- New event types may be added (currently reserved 49-99)
- New stimulus modes can be added with unique IDs
- New fields can be added to existing structures without breaking old readers

---

## End of Documentation

This document provides a complete reference for ingesting and analyzing Citrus experimental data. For questions about specific fields or trial types not covered here, refer to the source code in the project or consult the research team.

**Key Source Files:**
- `src/core/stimulus_globals.h` - Stimulus modes and event types
- `src/protocols/protocol_parameters.h` - All parameter structures
- `src/logging/logging_structs.h` - HDF5 data structures
- `src/logging/session_logger.h` - Logging implementation
- `src/protocols/protocol_io.cpp` - JSON serialization

**Last Updated:** October 2025
