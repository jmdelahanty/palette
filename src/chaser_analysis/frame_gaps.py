import csv

def analyze_frame_gaps(csv_file_path, frame_column_index=0):
    """
    Analyzes frame number gaps in a CSV file.

    Args:
        csv_file_path (str): The path to the CSV file.
        frame_column_index (int): The index of the column containing frame numbers.
    """
    frame_numbers = []
    try:
        with open(csv_file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            # Skip header row
            next(reader, None)
            for row in reader:
                try:
                    frame_numbers.append(int(row[frame_column_index]))
                except (ValueError, IndexError):
                    print(f"Warning: Could not read a valid frame number from row: {row}")
                    continue
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        return

    if not frame_numbers:
        print("No frame numbers were found in the file.")
        return

    gaps = {}
    for i in range(1, len(frame_numbers)):
        gap = frame_numbers[i] - frame_numbers[i-1]
        if gap != 1:
            if gap in gaps:
                gaps[gap] += 1
            else:
                gaps[gap] = 1
            print(f"Gap of {gap} frames found between frame {frame_numbers[i-1]} and {frame_numbers[i]}")

    print("\n--- Analysis Summary ---")
    if not gaps:
        print("No frame gaps were found.")
    else:
        print("Summary of frame gaps:")
        for gap, count in gaps.items():
            print(f"  - Gap of {gap} frames occurred {count} time(s).")
        if 60 in gaps:
            print("\n*** A gap of 60 frames was detected, which supports your hypothesis. ***")

if __name__ == '__main__':
    # --- Instructions ---
    # 1. Replace 'your_file.csv' with the path to your CSV file.
    # 2. If your frame numbers are not in the first column (index 0),
    #    change the frame_column_index value.
    csv_file = '/home/delahantyj@hhmi.org/Desktop/escape_2/Cam2010096_meta.csv'
    analyze_frame_gaps(csv_file, frame_column_index=0)