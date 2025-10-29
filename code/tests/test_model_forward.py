import sys
import os

# This adds the parent directory of 'current_script_dir' to Python's search path.
# If your_script.py is in /app/current_script_dir/, this adds /app to sys.path.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root_dir)
# test_model_forward.py


from utils.models import UNet # Or your chosen model
import torch

import json
def load_test_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

config = load_test_config('../configs/example_config.json')

out_features = 0
if 'landmarks' in config:
    for landmark in config['landmarks']:
        out_features += len(landmark.get('coordinates', []))
if out_features == 0:
    raise ValueError("out_features is 0. Check landmark config.")

print(f"Calculated out_features: {out_features}")

try:
    model = UNet(in_channels=1, out_features=out_features) # Example with UNet
    model.eval() # Set to evaluation mode for a simple forward pass test
    print(f"Model {config.get('model_type', 'UNet')} instantiated successfully.")

    # Create a dummy input tensor (batch_size, channels, height, width)
    # Assuming your processed images are 512x512. Adjust if different.
    dummy_input = torch.randn(1, 1, 512, 512)
    print(f"Dummy input tensor shape: {dummy_input.shape}")

    with torch.no_grad(): # No need to track gradients for this test
        output = model(dummy_input)
    print(f"Model output shape: {output.shape}") # Should be (1, out_features)

    expected_shape = (1, out_features)
    if output.shape == expected_shape:
        print("Model forward pass successful and output shape is correct.")
    else:
        print(f"Error: Model output shape is {output.shape}, expected {expected_shape}.")

except Exception as e:
    print(f"Error during model instantiation or forward pass: {e}")
