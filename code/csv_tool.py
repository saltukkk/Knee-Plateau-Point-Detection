import numpy as np
import pandas as pd
import argparse

def process_csv(input_file, output_file):
    """
    Processes a CSV file by adding a random split, a dummy label,
    and then saves it to a new file.

    Args:
        input_file (str): The path to the input CSV file.
        output_file (str): The path to the output CSV file.
    """
    try:
        # Read the input CSV file into a pandas DataFrame
        df = pd.read_csv(input_file)

        # --- 1. Copy SOPInstanceUID ---
        # This is implicitly handled as we are working with the existing DataFrame.

        # --- 2. Randomly assign splits (60% train, 20% validation, 20% test) ---
        total_rows = len(df)
        split_percentages = {'Train': 0.6, 'Validation': 0.2, 'Test': 0.2}

        # Calculate the number of rows for each split
        train_count = int(total_rows * split_percentages['Train'])
        validation_count = int(total_rows * split_percentages['Validation'])
        # Assign the rest to test to handle any rounding issues
        test_count = total_rows - train_count - validation_count

        # Create the list of split labels
        splits = ['Train'] * train_count + ['Validation'] * validation_count + ['Test'] * test_count

        # Shuffle splits with a fixed seed for reproducibility
        np.random.seed(42)
        np.random.shuffle(splits)
        df['Split'] = splits

        # --- 3. Add a dummy labelName ---
        df['labelName'] = "1"

        # --- 4. Copy adjusted_landmarks ---
        # This is also implicitly handled by reading it from the input and keeping it.

        # Reorder columns to match the desired output format
        output_df = df[['SOPInstanceUID', 'Split', 'labelName', 'adjusted_landmarks']]

        # Save the processed DataFrame to the output CSV file
        output_df.to_csv(output_file, index=False)

        print(f"Successfully processed '{input_file}' and saved the output to '{output_file}'.")
        print("\nSplit distribution:")
        print(df['Split'].value_counts(normalize=True))


    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- ---
# Usage
# --- ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV file by assigning random splits and a dummy label.")
    parser.add_argument("input_csv_file", help="Path to the input CSV file.")
    parser.add_argument("output_csv_file", help="Path to the output CSV file.")
    args = parser.parse_args()

    process_csv(args.input_csv_file, args.output_csv_file)

