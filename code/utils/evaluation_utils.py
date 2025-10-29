# utils/evaluation_utils.py
import json
from math import atan2, degrees # Keep if general angle calculations are ever needed
import numpy as np
import pandas as pd
import torch
from utils.models import UNet, RAUNet, CRAUNet, ResNeXt50 # Ensure all models are correctly imported

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def load_model(model_type, model_path, device, out_features): # Added out_features
    """Load and return the specified model type onto the device."""
    if model_type == "UNet":
        model = UNet(in_channels=1, out_features=out_features).to(device)
    elif model_type == "RAUNet":
        model = RAUNet(in_channels=1, out_features=out_features).to(device)
    elif model_type == "CRAUNet":
        model = CRAUNet(in_channels=1, out_features=out_features).to(device)
    elif model_type == "ResNeXt50":
        model = ResNeXt50(in_channels=1, out_features=out_features).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        raise
    except RuntimeError as e:
        print(f"Error loading model state_dict: {e}")
        print("This might be due to a mismatch in 'out_features' between the saved model and the current model configuration.")
        raise
    model.eval()
    return model

def preprocess_data(config): # This version of preprocess_data seems simpler, ensure it's what you need for eval_utils context
    """Basic preprocess_data for eval_utils if needed, otherwise dataloader's version is more complete."""
    try:
        split_df = pd.read_csv(config['split_file'])
        details_df = pd.read_csv(config['details_file'])
    except FileNotFoundError as e:
        print(f"Error reading CSV for preprocessing in eval_utils: {e}")
        raise
    unified_df = pd.merge(split_df, details_df, on='SOPInstanceUID', how='left', suffixes=('_split', '_details'))
    # Handle potential column name conflicts like in dataloader.preprocess_data if necessary
    if 'adjusted_landmarks_split' in unified_df.columns:
        unified_df['adjusted_landmarks'] = unified_df['adjusted_landmarks_split']
    elif 'adjusted_landmarks' not in unified_df.columns:
        print("Warning in eval_utils.preprocess_data: 'adjusted_landmarks' column missing after merge.")
    return unified_df

# calculate_perpendicular_endpoint - REMOVE (mammography specific)
# is_point_inside_image - Keep (generic utility)
# is_point_inside_image_with_threshold - Keep (generic utility, but its specific use for PNL quality is removed)
# calculate_side_specific_angle - REMOVE (mammography specific)

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def parse_pixel_spacing(spacing_str):
    if pd.isna(spacing_str): # Handle missing pixel spacing
        print("Warning: Pixel spacing is NaN. Returning default (1.0, 1.0). MM distances will be equivalent to pixel distances.")
        return 1.0, 1.0
    parts = str(spacing_str).split('\\')
    if len(parts) == 2:
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            print(f"Warning: Could not parse pixel spacing '{spacing_str}'. Returning default (1.0, 1.0).")
            return 1.0, 1.0
    else: # Handle cases where format might be different or missing
        print(f"Warning: Unexpected pixel spacing format '{spacing_str}'. Expected 'X\\Y'. Returning default (1.0, 1.0).")
        return 1.0, 1.0 # Default if format is unexpected

def calculate_mm_distance(orig_point_coords, pred_point_coords, df_row):
    """
    Calculate the millimeter distance between predicted and original landmark points.
    orig_point_coords, pred_point_coords: tuples/lists like (x, y)
    df_row: pandas Series containing 'PixelSpacing' or an 'adjusted_pixel_spacing' column.
            The original code used 'adjusted_pixel_spacing'. I'll look for that first.
    """
    pixel_spacing_col_names = ['adjusted_pixel_spacing', 'PixelSpacing'] # Add other potential names
    spacing_str = None
    for col_name in pixel_spacing_col_names:
        if col_name in df_row and not pd.isna(df_row[col_name]):
            spacing_str = df_row[col_name]
            break
    
    if spacing_str is None:
        # print(f"Warning: No pixel spacing information found in df_row for SOP {df_row.get('SOPInstanceUID', 'N/A')}. MM distance will be pixel distance.")
        spacing_x, spacing_y = 1.0, 1.0 # Default to 1.0 if no spacing info
    else:
        spacing_x, spacing_y = parse_pixel_spacing(spacing_str)

    pixel_dist = euclidean_distance(orig_point_coords[0], orig_point_coords[1], 
                                    pred_point_coords[0], pred_point_coords[1])
    
    # Use average of spacing_x and spacing_y for conversion
    # This assumes isotropic pixels if only one value is effectively used,
    # or that the difference between spacing_x and spacing_y is handled by this averaging.
    mm_distance = pixel_dist * ((spacing_x + spacing_y) / 2.0)
    return mm_distance


# calculate_sensitivity_specificity:
# This function's current implementation is for binary classification (Good/Bad).
# For point detection, sensitivity/specificity would typically be defined based on
# whether a predicted point is within a certain tolerance radius of the ground truth.
# You'll need to redefine this if you want to use it for point detection tasks.
# For now, it can be kept if you have another binary classification aspect,
# or removed if purely doing point regression.
# The old use was tied to the perpendicular line quality.

def calculate_sensitivity_specificity_classification(predictions_binary, truths_binary):
    """
    Calculate sensitivity and specificity for binary classification.
    Assumes predictions_binary and truths_binary contain 0s and 1s.
    Sensitivity (Recall or True Positive Rate) for class 1.
    Specificity (True Negative Rate) for class 0.
    Let's define positive class = 1 (e.g., "Good"), negative class = 0 (e.g., "Bad").
    """
    if not isinstance(predictions_binary, np.ndarray): predictions_binary = np.array(predictions_binary)
    if not isinstance(truths_binary, np.ndarray): truths_binary = np.array(truths_binary)

    tp = np.sum((predictions_binary == 1) & (truths_binary == 1)) # True Positives (correctly predicted as 1)
    fn = np.sum((predictions_binary == 0) & (truths_binary == 1)) # False Negatives (predicted as 0, actually 1)
    tn = np.sum((predictions_binary == 0) & (truths_binary == 0)) # True Negatives (correctly predicted as 0)
    fp = np.sum((predictions_binary == 1) & (truths_binary == 0)) # False Positives (predicted as 1, actually 0)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    # accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return sensitivity, specificity