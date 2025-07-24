import cv2
import numpy as np
import os
import argparse

def generate_annotated_image(frame_number, base_output_path):
    """
    Loads an ROI image and its label file, draws the keypoints,
    and returns the annotated image as a NumPy array.
    Returns None if the files for the frame do not exist.
    """
    roi_image_path = os.path.join(base_output_path, 'roi/images', f"{frame_number}.png")
    roi_label_path = os.path.join(base_output_path, 'roi/labels', f"{frame_number}.txt")

    if not os.path.exists(roi_image_path):
        return None

    image = cv2.imread(roi_image_path)
    if image is None:
        return None
        
    height, width, _ = image.shape

    # Class 0: Bladder (Red), Class 1: Left Eye (Green), Class 2: Right Eye (Blue)
    colors = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 100, 0)}

    if os.path.exists(roi_label_path):
        with open(roi_label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center = int(float(parts[1]) * width)
                y_center = int(float(parts[2]) * height)
                color = colors.get(class_id, (255, 255, 255))
                cv2.circle(image, (x_center, y_center), radius=4, color=color, thickness=-1)
                cv2.circle(image, (x_center, y_center), radius=5, color=(0,0,0), thickness=1)

    # Add frame number text to the image
    cv2.putText(image, f"Frame: {frame_number}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return image


def main(start_frame, base_output_path):
    """Main display loop for interactive visualization."""
    current_frame = start_frame
    
    print("Starting interactive visualizer...")
    print("Controls: → (Next Frame), ← (Previous Frame), 'q' or Esc (Quit)")

    while True:
        annotated_image = generate_annotated_image(current_frame, base_output_path)

        if annotated_image is None:
            # Create a black screen with "Frame Not Found" text
            annotated_image = np.zeros((320, 320, 3), dtype=np.uint8)
            text = f"Frame {current_frame} Not Found"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            text_x = (annotated_image.shape[1] - text_width) // 2
            text_y = (annotated_image.shape[0] + text_height) // 2
            cv2.putText(annotated_image, text, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
        cv2.imshow("Interactive Visualizer", annotated_image)
        
        # Wait for a key press
        key = cv2.waitKey(0)

        # Handle different key presses
        if key == ord('q') or key == 27:  # 'q' or Esc key to quit
            break
        elif key == 83 or key == 2555904:  # Right arrow key
            current_frame += 1
        elif key == 81 or key == 2424832:  # Left arrow key
            current_frame = max(1, current_frame - 1) # Prevent going below frame 1

    cv2.destroyAllWindows()
    print("Visualizer closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactively visualize fish tracking keypoints.")
    # The starting frame is now an optional argument
    parser.add_argument("start_frame", type=int, nargs='?', default=1, 
                        help="The frame number to start visualizing from. Defaults to 1.")
    args = parser.parse_args()

    output_path = r'/home/delahantyj@hhmi.org/Desktop/yolo_data_zarr'
    
    main(args.start_frame, output_path)