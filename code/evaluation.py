# evaluation.py
import torch
import numpy as np
import pandas as pd
import argparse
import json # For load_config if not already imported by evaluation_utils
from utils.dataloader import create_dataloaders, preprocess_data as preprocess_data_dl # Alias if names conflict
from utils.evaluation_utils import (
    load_config as load_eval_config, # Alias if names conflict
    preprocess_data as preprocess_data_eval_utils, # Alias if names conflict
    calculate_mm_distance, 
    euclidean_distance,
    load_model
    # Removed: calculate_perpendicular_endpoint, is_point_inside_image_with_threshold, calculate_side_specific_angle
    # calculate_sensitivity_specificity might need to be re-defined or used carefully
)

# Helper to get landmark names from config
def get_landmark_names_from_config(landmarks_config):
    return [lm['name'] for lm in landmarks_config]

def get_num_coords_per_landmark(landmarks_config):
    if not landmarks_config: return 0
    return len(landmarks_config[0].get('coordinates', [])) # Assumes all landmarks have same num_coords

def evaluate_model(config, model, test_loader, test_df): # Renamed from evaluate_and_label
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    landmarks_config = config.get('landmarks', [])
    if not landmarks_config:
        raise ValueError("Landmark configuration missing in eval_config.")
        
    landmark_names = get_landmark_names_from_config(landmarks_config)
    num_coords_per_lm = get_num_coords_per_landmark(landmarks_config) # Should be 2 for (x,y)

    # Initialize dictionaries to store distances for each landmark
    pixel_distances = {name: [] for name in landmark_names}
    mm_distances = {name: [] for name in landmark_names}
    
    evaluation_results = []

    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            if batch_data is None or (isinstance(batch_data, (list, tuple)) and (batch_data[0] is None or batch_data[1] is None)):
                print(f"Skipping problematic batch {i} in evaluation due to loading error.")
                continue
            
            image, original_landmarks_normalized = batch_data
            image = image.to(device)
            
            # Model predictions are normalized (0-1 range)
            predictions_normalized = model(image).cpu().numpy() # Shape: (batch_size, num_total_coords)
            original_landmarks_normalized = original_landmarks_normalized.numpy() # Shape: (batch_size, num_total_coords)

            img_height, img_width = image.shape[2], image.shape[3] # Get from image tensor

            # Process each item in the batch
            for batch_idx in range(predictions_normalized.shape[0]):
                current_preds_norm = predictions_normalized[batch_idx]
                current_orig_norm = original_landmarks_normalized[batch_idx]

                # Rescale to image dimensions
                # Reshape to (num_landmarks, num_coords_per_lm)
                preds_rescaled = np.reshape(current_preds_norm, (-1, num_coords_per_lm))
                orig_rescaled = np.reshape(current_orig_norm, (-1, num_coords_per_lm))
                
                # Apply scaling: x by width, y by height
                scaling_vector = np.array([img_width, img_height] * (preds_rescaled.shape[0] // (num_coords_per_lm // num_coords_per_lm))).reshape(-1, num_coords_per_lm) # Handles if num_coords_per_lm changes from 2
                if num_coords_per_lm == 2: # Common case for (x,y)
                    scaling_vector = np.array([img_width, img_height])
                else: # General case, might need adjustment based on coordinate order
                    temp_scale = []
                    for lm_idx in range(len(landmarks_config)):
                        for coord_idx, coord_name in enumerate(landmarks_config[lm_idx]['coordinates']):
                            temp_scale.append(img_width if coord_name.lower() == 'x' else img_height) # Simple x/y assumption
                    scaling_vector = np.array(temp_scale).reshape(-1, num_coords_per_lm)


                preds_img_coords = preds_rescaled * scaling_vector
                orig_img_coords = orig_rescaled * scaling_vector

                # Get corresponding row from test_df for this item
                # This assumes batch_size=1 or careful indexing if batch_size > 1
                # For batch_size > 1, test_loader should not shuffle, and 'i' needs to be mapped to df index
                # If batch_size is always 1 for eval (as in config), then i is the direct index in test_df
                if test_df.shape[0] <= (i * config['batch_size'] + batch_idx) :
                    print(f"Warning: Index out of bounds for test_df. Skipping result for image index {i * config['batch_size'] + batch_idx}")
                    continue
                df_row = test_df.iloc[i * config['batch_size'] + batch_idx]


                result_row = {
                    "image_index": i * config['batch_size'] + batch_idx + 1,
                    "study_uid": df_row.get('StudyInstanceUID', 'N/A'),
                    "sop_uid": df_row.get('SOPInstanceUID', 'N/A'),
                }

                for lm_idx, lm_name in enumerate(landmark_names):
                    pred_pt = preds_img_coords[lm_idx] # (x,y)
                    orig_pt = orig_img_coords[lm_idx] # (x,y)

                    px_dist = euclidean_distance(orig_pt[0], orig_pt[1], pred_pt[0], pred_pt[1])
                    pixel_distances[lm_name].append(px_dist)
                    
                    # Calculate mm distance - calculate_mm_distance expects (pt1, pt2, df_row)
                    # pt1, pt2 should be the (x,y) tuples/lists
                    mm_dist = calculate_mm_distance(orig_pt, pred_pt, df_row)
                    mm_distances[lm_name].append(mm_dist)

                    result_row[f"{lm_name}_pred_x"] = pred_pt[0]
                    result_row[f"{lm_name}_pred_y"] = pred_pt[1]
                    result_row[f"{lm_name}_orig_x"] = orig_pt[0]
                    result_row[f"{lm_name}_orig_y"] = orig_pt[1]
                    result_row[f"{lm_name}_pixel_dist"] = px_dist
                    result_row[f"{lm_name}_mm_dist"] = mm_dist
                
                evaluation_results.append(result_row)

    results_df = pd.DataFrame(evaluation_results)
    
    # Dynamically create columns for saving
    base_cols = ["image_index", "study_uid", "sop_uid"]
    landmark_cols = []
    for lm_name in landmark_names:
        landmark_cols.extend([
            f"{lm_name}_pred_x", f"{lm_name}_pred_y",
            f"{lm_name}_orig_x", f"{lm_name}_orig_y",
            f"{lm_name}_pixel_dist", f"{lm_name}_mm_dist"
        ])
    final_cols = base_cols + landmark_cols
    results_df = results_df[final_cols]
    results_df.to_csv('evaluation_results_generic.csv', index=False)
    print("Evaluation results saved to evaluation_results_generic.csv")

    # Summary Statistics
    summary_metrics = []
    for lm_name in landmark_names:
        if pixel_distances[lm_name]: # Check if list is not empty
            mean_px_dist = np.mean(pixel_distances[lm_name])
            std_px_dist = np.std(pixel_distances[lm_name])
            median_px_dist = np.median(pixel_distances[lm_name])
            summary_metrics.append({
                "Landmark": lm_name,
                "Metric": "Pixel Distance",
                "Mean": mean_px_dist, "Std": std_px_dist, "Median": median_px_dist
            })
        if mm_distances[lm_name]:
            mean_mm_dist = np.mean(mm_distances[lm_name])
            std_mm_dist = np.std(mm_distances[lm_name])
            median_mm_dist = np.median(mm_distances[lm_name])
            summary_metrics.append({
                "Landmark": lm_name,
                "Metric": "MM Distance",
                "Mean": mean_mm_dist, "Std": std_mm_dist, "Median": median_mm_dist
            })
            
    summary_df = pd.DataFrame(summary_metrics)
    summary_df.to_csv('evaluation_summary_stats_generic.csv', index=False)
    print("Evaluation summary stats saved to evaluation_summary_stats_generic.csv")


def main(eval_config_path):
    # Use aliased load_config and preprocess_data to avoid naming conflicts
    config = load_eval_config(eval_config_path) # From evaluation_utils
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use preprocess_data from dataloader.py as it's more comprehensive for merging
    # The one in evaluation_utils.py might be simpler or intended for different purpose
    unified_df = preprocess_data_dl(config) # From dataloader.py
    
    # Create dataloaders; ensure return_test_df=True
    # create_dataloaders expects the main config structure for 'landmarks', 'batch_size', etc.
    # eval_config should have these.
    dataloaders, test_df = create_dataloaders(unified_df, config, return_test_df=True) 
    
    if 'Test' not in dataloaders or test_df is None or test_df.empty:
        print("Error: Test data loader or test_df is not available or empty. Cannot proceed with evaluation.")
        return

    test_loader = dataloaders['Test']

    # Determine out_features for model loading
    out_features = 0
    if 'landmarks' in config:
        for landmark in config['landmarks']:
            out_features += len(landmark.get('coordinates', []))
    if out_features == 0:
        raise ValueError("Cannot determine out_features for model loading from eval_config.")

    # Use the generic load_model from evaluation_utils, ensuring it takes out_features
    model = load_model(config['model_type'], config['best_model_path'], device, out_features=out_features)
    
    evaluate_model(config, model, test_loader, test_df)

def parse_args():
    parser = argparse.ArgumentParser(description="Model Evaluation Script")
    parser.add_argument('--config', type=str, required=True, help="Path to the evaluation configuration JSON file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.config) # Pass the path to main, load_eval_config will handle it