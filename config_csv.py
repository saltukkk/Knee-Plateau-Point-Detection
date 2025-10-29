import csv
import json

# Hardcoded data for transformation_details.csv
# This list contains data for 5 images: 55457, 55458, 55459 (original)
# and 55460, 55463 (new).
# Renamed from transformation_data to datum
datum = [
    {
        "SOPInstanceUID": 55457,
        "original_landmarks": {"Notch": {"x": 4148, "y": 1002}},
        "adjusted_landmarks": {"Notch": {"x": 227, "y": 333}},
        "crop_x": 683,
        "crop_y": 4049,
        "crop_width": 491,
        "crop_height": 254,
        "pad_top": 118,
        "pad_left": 0,
        "pad_bottom": 119,
        "pad_right": 0,
        "scale": 1.042770, # Rounded for simplicity
        "output_height": 512,
        "output_width": 512
    },
    {
        "SOPInstanceUID": 55458,
        "original_landmarks": {"Notch": {"x": 6131, "y": 916}},
        "adjusted_landmarks": {"Notch": {"x": 197, "y": 240}},
        "crop_x": 634,
        "crop_y": 6035,
        "crop_width": 601,
        "crop_height": 331,
        "pad_top": 135,
        "pad_left": 0,
        "pad_bottom": 135,
        "pad_right": 0,
        "scale": 0.851913, # Rounded for simplicity
        "output_height": 512,
        "output_width": 512
    },
    {
        "SOPInstanceUID": 55459,
        "original_landmarks": {"Notch": {"x": 5213, "y": 1066}},
        "adjusted_landmarks": {"Notch": {"x": 225, "y": 285}},
        "crop_x": 747,
        "crop_y": 5098,
        "crop_width": 574,
        "crop_height": 299,
        "pad_top": 137,
        "pad_left": 0,
        "pad_bottom": 138,
        "pad_right": 0,
        "scale": 0.891986, # Rounded for simplicity
        "output_height": 512,
        "output_width": 512
    },
    { # New image 1: 55460
        "SOPInstanceUID": 55460,
        "original_landmarks": {"Notch": {"x": 5922.37, "y": 1141.76}},
        "adjusted_landmarks": {"Notch": {"x": 180, "y": 261}},
        "crop_x": 688, # Average from originals, rounded
        "crop_y": 5061, # Average from originals, rounded
        "crop_width": 555, # Average from originals, rounded
        "crop_height": 295, # Average from originals, rounded
        "pad_top": 130, # Average from originals, rounded
        "pad_left": 0,   # Average from originals, rounded
        "pad_bottom": 131, # Average from originals, rounded
        "pad_right": 0,    # Average from originals, rounded
        "scale": 0.839344,
        "output_height": 512,
        "output_width": 512
    },
    { # New image 2: 55463
        "SOPInstanceUID": 55463,
        "original_landmarks": {"Notch": {"x": 4202.85, "y": 1296.74}},
        "adjusted_landmarks": {"Notch": {"x": 227, "y": 320}},
        "crop_x": 688, # Average from originals, rounded
        "crop_y": 5061, # Average from originals, rounded
        "crop_width": 555, # Average from originals, rounded
        "crop_height": 295, # Average from originals, rounded
        "pad_top": 130, # Average from originals, rounded
        "pad_left": 0,   # Average from originals, rounded
        "pad_bottom": 131, # Average from originals, rounded
        "pad_right": 0,    # Average from originals, rounded
        "scale": 1.292929,
        "output_height": 512,
        "output_width": 512
    }
]

# Hardcoded data for unified_data_with_splits.csv
# This list contains data for 5 images with splits:
# 3 Train (55457, 55460, 55463), 1 Validation (55458), 1 Test (55459)
# Renamed from unified_data to datum2
datum2 = [
    {
        "SOPInstanceUID": 55457,
        "Split": "Train",
        "labelName": "Notch",
        "adjusted_landmarks": {"Notch": {"x": 227, "y": 333}},
    },
    {
        "SOPInstanceUID": 55458,
        "Split": "Validation",
        "labelName": "Notch",
        "adjusted_landmarks": {"Notch": {"x": 197, "y": 240}},
    },
    {
        "SOPInstanceUID": 55459,
        "Split": "Test",
        "labelName": "Notch",
        "adjusted_landmarks": {"Notch": {"x": 225, "y": 285}},
    },
    { # New image 1: 55460
        "SOPInstanceUID": 55460,
        "Split": "Train",
        "labelName": "Notch",
        "adjusted_landmarks": {"Notch": {"x": 180, "y": 261}},
    },
    { # New image 2: 55463
        "SOPInstanceUID": 55463,
        "Split": "Train",
        "labelName": "Notch",
        "adjusted_landmarks": {"Notch": {"x": 227, "y": 320}},
    }
]

# Renamed from convert_row_to_csv_format to convert_row
def convert_row(row_dict):
    """
    Converts dictionary values in a row to JSON strings for CSV compatibility.
    """
    new_row = {}
    for key, value in row_dict.items():
        if isinstance(value, dict):
            new_row[key] = json.dumps(value)
        else:
            new_row[key] = value
    return new_row

if __name__ == "__main__":
    # Define output filenames
    output_transformation_filename = "hardcoded_transformation_details.csv"
    output_unified_filename = "hardcoded_unified_data.csv"

    # --- Write datum (transformation data) to CSV ---
    print(f"Preparing to write {len(datum)} rows to {output_transformation_filename}...")
    if datum:
        # Get headers from the keys of the first row in datum
        fieldnames1 = list(datum[0].keys())
        try:
            with open(output_transformation_filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames1)
                writer.writeheader()
                for row_data in datum:
                    writer.writerow(convert_row(row_data)) # Process each row
            print(f"Successfully wrote data to {output_transformation_filename}")
        except IOError:
            print(f"Error: Could not write to file {output_transformation_filename}")
        except Exception as e:
            print(f"An unexpected error occurred while writing {output_transformation_filename}: {e}")
    else:
        print(f"Data list for {output_transformation_filename} is empty. Nothing to write.")

    # --- Write datum2 (unified data) to CSV ---
    print(f"\nPreparing to write {len(datum2)} rows to {output_unified_filename}...")
    if datum2:
        # Get headers from the keys of the first row in datum2
        fieldnames2 = list(datum2[0].keys())
        try:
            with open(output_unified_filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames2)
                writer.writeheader()
                for row_data in datum2:
                    writer.writerow(convert_row(row_data)) # Process each row
            print(f"Successfully wrote data to {output_unified_filename}")
        except IOError:
            print(f"Error: Could not write to file {output_unified_filename}")
        except Exception as e:
            print(f"An unexpected error occurred while writing {output_unified_filename}: {e}")
    else:
        print(f"Data list for {output_unified_filename} is empty. Nothing to write.")

    print("\nScript finished.")

