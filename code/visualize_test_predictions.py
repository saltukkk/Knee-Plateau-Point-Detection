# visualize_test_predictions.py
import torch
import numpy as np
import matplotlib.pyplot as plt
# from utils.models import UNet, RAUNet, CRAUNet, ResNeXt50 # Not needed if load_model is used
from utils.dataloader import create_dataloaders, preprocess_data # Keep preprocess_data if used directly here
from utils.evaluation_utils import load_model # Use this
import os
import pandas as pd
import matplotlib.patches as patches # Keep if you want to draw markers/boxes
import json # For loading the main config

# Helper (same as in evaluation.py, consider putting in a shared util if used often)
def get_landmark_names_from_config(landmarks_config):
    return [lm['name'] for lm in landmarks_config]

def get_num_coords_per_landmark(landmarks_config):
    if not landmarks_config: return 0
    return len(landmarks_config[0].get('coordinates', []))


def visualize_prediction(image_numpy, predicted_coords_img, original_coords_img, landmark_names, title="Prediction vs Original"):
    """
    Visualizes a single image with predicted and original landmarks.
    image_numpy: (C, H, W) or (H, W)
    predicted_coords_img, original_coords_img: numpy arrays of shape (num_landmarks, 2) for (x,y)
    landmark_names: list of landmark names
    """
    if image_numpy.shape[0] == 1: # Grayscale with channel dim
        plt.imshow(image_numpy.squeeze(0), cmap='gray')
    else: # Assuming (H,W)
        plt.imshow(image_numpy, cmap='gray')

    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k'] # Cycle through colors for multiple landmarks

    for i, name in enumerate(landmark_names):
        pred_x, pred_y = predicted_coords_img[i]
        orig_x, orig_y = original_coords_img[i]
        
        color_idx = i % len(colors)
        # Predicted
        plt.scatter(pred_x, pred_y, s=30, marker='x', c=colors[color_idx], label=f'Pred {name}' if i==0 else None)
        # Original
        plt.scatter(orig_x, orig_y, s=30, marker='o', facecolors='none', edgecolors=colors[color_idx], label=f'Orig {name}' if i==0 else None)
        # Optional: Add text labels for points
        # plt.text(pred_x + 5, pred_y + 5, name, color=colors[color_idx], fontsize=9)


    # Remove mammography-specific drawing (lines, PNL)
    # If you want to draw bounding boxes or other generic markers, add logic here.
    # Example: bounding box around first landmark if needed
    # if landmark_names:
    #     bbox_size = 20
    #     pred_box = patches.Rectangle((predicted_coords_img[0,0]-bbox_size/2, predicted_coords_img[0,1]-bbox_size/2),
    #                                    bbox_size, bbox_size, linewidth=1, edgecolor='r', facecolor='none', linestyle='--')
    #     plt.gca().add_patch(pred_box)

    plt.legend()
    plt.axis('off') # Keep axis for reference during debugging if needed: plt.axis('on')
    # plt.tight_layout() # Can sometimes cause issues with savefig bbox_inches='tight'

    predictions_dir = 'predictions_visualization' # Changed dir name
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Sanitize title for filename
    safe_title = "".join(c if c.isalnum() else "_" for c in title)
    plt.savefig(os.path.join(predictions_dir, f"{safe_title}.png"), dpi=300, bbox_inches='tight')
    plt.close()


def test_and_visualize_model(config, model, test_loader): # Renamed from test_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) # Ensure model is on the right device
    model.eval()
    
    landmarks_config = config.get('landmarks', [])
    if not landmarks_config:
        raise ValueError("Landmark configuration missing in config for visualization.")
    landmark_names = get_landmark_names_from_config(landmarks_config)
    num_coords_per_lm = get_num_coords_per_landmark(landmarks_config)

    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            if i >= 20:  # Limit number of visualizations
                print("Reached visualization limit (20 images).")
                break

            if batch_data is None or (isinstance(batch_data, (list, tuple)) and (batch_data[0] is None or batch_data[1] is None)):
                print(f"Skipping problematic batch {i} in visualization due to loading error.")
                continue
            
            image_tensor, original_landmarks_normalized = batch_data
            image_tensor = image_tensor.to(device) # Send image to device for model
            
            predictions_normalized = model(image_tensor).cpu().numpy() # (batch_size, num_total_coords)
            original_landmarks_normalized = original_landmarks_normalized.numpy() # (batch_size, num_total_coords)

            # Process each item in the batch (assuming batch_size=1 for visualization is common)
            for batch_idx in range(predictions_normalized.shape[0]):
                if i * config.get('batch_size',1) + batch_idx >= 20: break # Overall limit

                current_preds_norm = predictions_normalized[batch_idx]
                current_orig_norm = original_landmarks_normalized[batch_idx]
                
                # Image for visualization (from CPU tensor, first item in batch)
                img_for_viz_numpy = image_tensor[batch_idx].cpu().numpy() # (C, H, W)
                img_height, img_width = img_for_viz_numpy.shape[1], img_for_viz_numpy.shape[2]

                # Rescale normalized coordinates to image dimensions
                preds_rescaled = np.reshape(current_preds_norm, (-1, num_coords_per_lm))
                orig_rescaled = np.reshape(current_orig_norm, (-1, num_coords_per_lm))
                
                scaling_vector = np.array([img_width, img_height]) # Assuming (x,y)
                
                preds_img_coords = preds_rescaled * scaling_vector
                orig_img_coords = orig_rescaled * scaling_vector
                
                sop_uid_info = "" # Placeholder if SOPInstanceUID is not easily available from loader
                # If you have SOPInstanceUIDs in your test_loader or can map 'i' to it:
                # sop_uid_info = f"_SOP_{test_df.iloc[i]['SOPInstanceUID']}" # If test_df is accessible

                visualize_prediction(
                    img_for_viz_numpy, 
                    preds_img_coords, 
                    orig_img_coords, 
                    landmark_names,
                    title=f"Test_Sample_{i * config.get('batch_size',1) + batch_idx + 1}{sop_uid_info}"
                )
            if i * config.get('batch_size',1) + batch_idx >= 19 : break


def main_visualize(config_path): # Renamed from main
    # Load the main configuration file used for training/evaluation
    # because it contains all necessary paths and landmark definitions.
    with open(config_path, 'r') as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Preprocess and merge data (using dataloader's preprocess_data)
    unified_df = preprocess_data(config) # config has 'split_file', 'details_file'

    # Create dataloaders (config has 'landmarks', 'batch_size', 'base_image_dir', 'data_filter_labelName')
    dataloaders, test_df = create_dataloaders(unified_df, config, return_test_df=True)
    
    if 'Test' not in dataloaders or not dataloaders['Test']:
        print("Error: Test dataloader not found or empty. Cannot visualize.")
        return
    test_loader = dataloaders['Test']

    # Determine out_features for model loading
    out_features = 0
    if 'landmarks' in config:
        for landmark in config['landmarks']:
            out_features += len(landmark.get('coordinates', []))
    if out_features == 0:
        raise ValueError("Cannot determine out_features for model loading from config.")

    # Load the model using the utility from evaluation_utils
    model = load_model(config['model_type'], config['best_model_path'], device, out_features=out_features)
    
    test_and_visualize_model(config, model, test_loader)
    print("Visualization finished. Check the 'predictions_visualization' directory.")


if __name__ == "__main__":
    # This script should ideally use the main training config or an eval config that mirrors it.
    # The example_eval_config.json might be too minimal.
    # Let's assume we pass a path to a comprehensive config like example_config.json
    import argparse
    parser = argparse.ArgumentParser(description="Visualize model predictions on test set.")
    parser.add_argument('--config', type=str, required=True, 
                        help="Path to the JSON configuration file (e.g., training config).")
    args = parser.parse_args()
    
    main_visualize(args.config)

    # Example of how you might call it if not using argparse (for testing in an IDE):
    # mock_config_path = 'path_to_your/example_config.json' 
    # main_visualize(mock_config_path)