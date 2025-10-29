# test_dataloader.py
import sys
import os

# This adds the parent directory of 'current_script_dir' to Python's search path.
# If your_script.py is in /app/current_script_dir/, this adds /app to sys.path.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root_dir)

# test_loss_function.py
from utils.loss import MultifacetedLoss
import torch

import json
def load_test_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

config = load_test_config('../configs/example_config.json')


print("Testing Loss Function...")
try:
    criterion = MultifacetedLoss(
        config_landmarks=config['landmarks'],
        loss_wing_w=config['loss_wing_w'],
        loss_wing_epsilon=config['loss_wing_epsilon'],
        landmark_loss_weights=config['landmark_loss_weights']
    )
    print("MultifacetedLoss instantiated successfully.")

    # Determine out_features from config
    out_features = 0
    for landmark in config['landmarks']:
        out_features += len(landmark.get('coordinates', []))

    # Create dummy predictions and targets
    # Shape: (batch_size, out_features)
    # For "Notch", out_features = 2. Example batch_size = 4.
    batch_size = 4
    dummy_predictions = torch.rand(batch_size, out_features)  # e.g., (4, 2)
    dummy_targets = torch.rand(batch_size, out_features)    # e.g., (4, 2)
    print(f"Dummy predictions shape: {dummy_predictions.shape}")
    print(f"Dummy targets shape: {dummy_targets.shape}")


    loss = criterion(dummy_predictions, dummy_targets)
    print(f"Calculated loss: {loss.item()}") # Should be a scalar

    if isinstance(loss, torch.Tensor) and loss.numel() == 1:
        print("Loss calculation successful and returns a scalar tensor.")
    else:
        print(f"Error: Loss is not a scalar tensor. Shape: {loss.shape if isinstance(loss, torch.Tensor) else type(loss)}")

except Exception as e:
    print(f"Error during loss function testing: {e}")
    # raise
