# utils/dataloader.py

import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import json

class Dataset(Dataset):
    def __init__(self, dataframe, base_image_dir, landmarks_config): # landmarks_config from main config
        """
        Args:
            dataframe (DataFrame): DataFrame containing the data paths and adjusted landmarks.
            base_image_dir (string): Directory with all the images as .npy files.
            landmarks_config (list): List of landmark definitions from the config.
        """
        self.dataframe = dataframe
        self.base_image_dir = base_image_dir
        self.landmarks_config = landmarks_config

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_name = str(row['SOPInstanceUID']) + '.npy' # Ensure SOPInstanceUID is string
        img_path = os.path.join(self.base_image_dir, img_name)
        
        try:
            # Load the image data from the .npy file
            loaded_image_np = np.load(img_path)
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_path}")
            return None, None # Propagate None to be handled by DataLoader if using a custom collate_fn
            
        # Process the loaded numpy array to ensure it's a single-channel (grayscale) image
        if loaded_image_np.ndim == 3:
            if loaded_image_np.shape[2] == 3:  # Assuming (H, W, 3) for RGB color images
                # Convert to grayscale using the luminance formula: Y = 0.299R + 0.587G + 0.114B
                processed_image_np = 0.2989 * loaded_image_np[:, :, 0] + \
                                     0.5870 * loaded_image_np[:, :, 1] + \
                                     0.1140 * loaded_image_np[:, :, 2]
            elif loaded_image_np.shape[2] == 1:  # Assuming (H, W, 1) for grayscale images with a channel dimension
                processed_image_np = loaded_image_np.squeeze(axis=2) # Remove the channel dimension -> (H, W)
            else:
                raise ValueError(f"Unsupported channel size for 3D image: {loaded_image_np.shape} in file {img_name}")
        elif loaded_image_np.ndim == 2:  # Assuming (H, W) for grayscale images
            processed_image_np = loaded_image_np
        else:
            raise ValueError(f"Unsupported image dimensions: {loaded_image_np.ndim} for image {img_name}")

        # Ensure the numpy array is C-contiguous for PyTorch conversion if it's not already.
        # This can prevent potential warnings or errors with torch.from_numpy.
        if not processed_image_np.flags['C_CONTIGUOUS']:
            processed_image_np = np.ascontiguousarray(processed_image_np)

        # Convert the processed numpy array to a PyTorch tensor
        image_tensor = torch.from_numpy(processed_image_np).float()  # Shape becomes (H, W)
        
        # Add the channel dimension: (H, W) -> (1, H, W)
        image_tensor = image_tensor.unsqueeze(0)
            
        image_height, image_width = processed_image_np.shape[:2] # Use shape of processed image
        
        adjusted_landmarks_str = row['adjusted_landmarks'] 
        
        try:
            adjusted_landmarks_dict = json.loads(adjusted_landmarks_str)
        except json.JSONDecodeError:
            print(f"Error decoding JSON for landmarks: {adjusted_landmarks_str} for SOPInstanceUID: {row['SOPInstanceUID']}")
            return None, None

        coordinates = self.parse_coordinates(adjusted_landmarks_dict, image_width, image_height)
        coordinates = torch.tensor(coordinates, dtype=torch.float)
        
        return image_tensor, coordinates # image_tensor should now be (1, H, W)

    def parse_coordinates(self, landmarks_dict, width, height):
        parsed_coords = []
        for landmark_info in self.landmarks_config:
            landmark_name = landmark_info['name']
            if landmark_name in landmarks_dict:
                point = landmarks_dict[landmark_name]
                for coord_name in landmark_info['coordinates']: # e.g., "x", "y"
                    if coord_name in point:
                        # Normalize coordinates
                        if coord_name == 'x':
                            parsed_coords.append(point[coord_name] / width)
                        elif coord_name == 'y':
                            parsed_coords.append(point[coord_name] / height)
                        else: # Should not happen if config is correct
                            parsed_coords.append(0.0) 
                    else: # Coordinate not found in data
                        print(f"Warning: Coordinate '{coord_name}' not found for landmark '{landmark_name}'. Appending 0.0.")
                        parsed_coords.append(0.0) # Or handle error more strictly
            else: # Landmark not found in data
                print(f"Warning: Landmark '{landmark_name}' not found in image's landmarks. Appending 0.0 for its coordinates.")
                for _ in landmark_info['coordinates']:
                    parsed_coords.append(0.0) # Or handle error
        return parsed_coords

def create_dataloaders(unified_df, config, return_test_df=False, base_image_dir=None):
    # Filter based on the new config key
    label_name_to_filter = config.get('data_filter_labelName', None)
    if label_name_to_filter:
        # Ensure 'labelName' column exists
        if 'labelName' in unified_df.columns:
            df = unified_df[unified_df['labelName'] == label_name_to_filter].copy() # Use .copy() to avoid SettingWithCopyWarning
            if df.empty:
                print(f"Warning: No data found for labelName '{label_name_to_filter}'. Check your CSV and config.")
        else:
            print(f"Warning: 'labelName' column not found in unified_df. Cannot filter by '{label_name_to_filter}'. Using all data.")
            df = unified_df.copy()
    else: # If no filter key, use all data (or Pectoralis for backward compatibility if needed)
        print("Warning: 'data_filter_labelName' not in config. Attempting to filter for 'Pectoralis' for legacy reasons, or using all data.")
        if 'labelName' in unified_df.columns and 'Pectoralis' in unified_df['labelName'].unique():
             df = unified_df[unified_df['labelName'] == 'Pectoralis'].copy()
        else:
            df = unified_df.copy() # Fallback to all data if 'Pectoralis' is not applicable

    if df.empty:
        raise ValueError("DataFrame is empty after filtering. Cannot create dataloaders.")

    datasets = {}
    test_df_split = None # Renamed to avoid conflict
    
    landmarks_config_from_main = config.get('landmarks', [])
    if not landmarks_config_from_main:
        raise ValueError("'landmarks' definition is missing in the configuration file.")

    for split in ['Train', 'Validation', 'Test']:
        # Ensure 'Split' column exists
        if 'Split' not in df.columns:
            raise ValueError(f"'Split' column not found in DataFrame. Available columns: {df.columns.tolist()}")
        
        split_df = df[df['Split'] == split]
        if split_df.empty:
            print(f"Warning: No data for split '{split}' after filtering.")
            # Decide how to handle: skip this split, or raise error
            # For now, we'll create an empty dataset which might lead to issues in DataLoader
            # A better approach is to ensure all splits have data or handle it in the training loop.
        
        if split == 'Test':
            test_df_split = split_df # Store the test DataFrame portion
        if base_image_dir is None:
            datasets[split] = Dataset(split_df, config['base_image_dir'], landmarks_config_from_main)
        else:
            # Use provided base_image_dir if available
            datasets[split] = Dataset(split_df, base_image_dir, landmarks_config_from_main)

    # Filter out None items that might have occurred due to missing files/bad JSON in Dataset.__getitem__
    # This requires a custom collate_fn. A simpler approach is to ensure data integrity beforehand.
    # For now, assuming data is clean or __getitem__ raises an error that stops execution.
    
    dataloaders = {
        x: DataLoader(
            datasets[x], 
            batch_size=config['batch_size'], 
            shuffle=(x == 'Train'), 
            num_workers=0 # Set to 0 for simplicity, adjust as needed
        ) for x in datasets.keys() if len(datasets[x]) > 0 # Only create DataLoader if dataset is not empty
    }
    
    if not dataloaders.get('Test') and return_test_df:
        print("Warning: Test dataloader could not be created (e.g. no test data). test_df_split might be empty.")


    if return_test_df:
        return dataloaders, test_df_split
    else:
        return dataloaders


def preprocess_data(config):
    """Loads and merges split and details dataframes."""
    try:
        split_df = pd.read_csv(config['split_file'])
        details_df = pd.read_csv(config['details_file'])
    except FileNotFoundError as e:
        print(f"Error: CSV file not found. {e}")
        raise
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        raise

    # Ensure 'SOPInstanceUID' exists in both dataframes for merging
    if 'SOPInstanceUID' not in split_df.columns or 'SOPInstanceUID' not in details_df.columns:
        raise ValueError("Missing 'SOPInstanceUID' column in one of the CSV files, cannot merge.")

    # Merge the DataFrames
    unified_df = pd.merge(split_df, details_df, on='SOPInstanceUID', how='left', suffixes=('_split', '_details'))
    
    # Check if 'adjusted_landmarks_split' or 'adjusted_landmarks_details' exists and prioritize or rename
    # Based on your CSVs, 'adjusted_landmarks' comes from unified_data_with_splits.csv (split_df)
    # and transformation_details.csv also has an 'adjusted_landmarks' column.
    # pd.merge will create 'adjusted_landmarks_x' and 'adjusted_landmarks_y' if both have it.
    # Your original code uses 'adjusted_landmarks' directly after merge.
    # Let's assume the one from split_df is what you primarily used if there's a conflict.
    if 'adjusted_landmarks_split' in unified_df.columns:
         # If `_details` also exists and you need to choose or combine, add logic here
        unified_df['adjusted_landmarks'] = unified_df['adjusted_landmarks_split']
    elif 'adjusted_landmarks' not in unified_df.columns: # If no suffix and 'adjusted_landmarks' is missing
        raise ValueError("'adjusted_landmarks' column is missing after merge. Check input CSVs and merge logic.")

    return unified_df
