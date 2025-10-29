# test_dataloader.py
import sys
import os

# This adds the parent directory of 'current_script_dir' to Python's search path.
# If your_script.py is in /app/current_script_dir/, this adds /app to sys.path.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root_dir)

# test_training_step.py
from utils.dataloader import create_dataloaders, preprocess_data
from utils.models import UNet # Or your chosen model
from utils.loss import MultifacetedLoss
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR

import json
def load_test_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

config = load_test_config('../configs/example_config.json')


print("Testing Training Step...")

# 1. Setup: Data, Model, Criterion, Optimizer
try:
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    unified_df = preprocess_data(config)
    # Use a small subset for faster testing if needed, ensure 'Split' column exists and has 'Train'
    # For a single step test, we want at least one batch from 'Train'
    train_df_for_test = unified_df[unified_df['Split'] == 'Train'].head(config['batch_size'] * 2).copy()
    if train_df_for_test.empty:
         print("Warning: No 'Train' data in the first few samples of unified_df. Adjust subset or use full df.")
         # Fallback to using the full df for dataloader creation if subset is bad
         dataloaders = create_dataloaders(unified_df, config, base_image_dir="/app/data/images/") # Ensure base_image_dir is set correctly
    else:
         # Create a temporary minimal config for dataloader for this specific small df
         temp_config_for_loader = config.copy()
         # We are directly passing a df that should only contain 'Train' items for the 'Train' loader
         # The create_dataloaders function will further filter by 'Split'.
         # To make this simple, ensure train_df_for_test has 'Split' column correctly set.
         dataloaders = create_dataloaders(train_df_for_test, temp_config_for_loader, base_image_dir="/app/data/images/") # Ensure base_image_dir is set correctly


    if 'Train' not in dataloaders or dataloaders['Train'] is None or len(dataloaders['Train']) == 0:
        print("Train Dataloader is empty or None. Cannot perform training step test.")
        exit()
    train_loader = dataloaders['Train']

    out_features = sum(len(lm.get('coordinates', [])) for lm in config['landmarks'])
    model = UNet(in_channels=1, out_features=out_features).to(device) # Example
    criterion = MultifacetedLoss(
        config_landmarks=config['landmarks'],
        loss_wing_w=config['loss_wing_w'],
        loss_wing_epsilon=config['loss_wing_epsilon'],
        landmark_loss_weights=config['landmark_loss_weights']
    ).to(device)
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    # Scheduler setup (optional for single step test, but good to include if complex)
    # scheduler = CyclicLR(optimizer, base_lr=config['base_lr'], max_lr=config['max_lr'],
    #                      step_size_down=config.get('step_size_down', 2000), mode='triangular', cycle_momentum=False)

    print("Setup complete. Fetching one batch...")
    model.train() # Set model to training mode

    # Fetch one batch
    images, landmarks = next(iter(train_loader))
    if images is None or landmarks is None: # From our collate_fn
        print("Fetched batch is None. Dataloader might be empty or all items in first batch were problematic.")
        exit()

    images, landmarks = images.to(device), landmarks.to(device)
    print(f"Batch fetched: Images shape {images.shape}, Landmarks shape {landmarks.shape}")

    # 2. Perform a few training steps on this single batch
    num_test_steps = 3
    print(f"Performing {num_test_steps} training steps on the fetched batch...")
    for step in range(num_test_steps):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, landmarks)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Step {step+1}: NaN or Inf loss detected: {loss.item()}. Stopping test.")
            break

        loss.backward()
        optimizer.step()
        # scheduler.step() # If using scheduler per step

        print(f"  Step {step+1}/{num_test_steps} - Loss: {loss.item():.4f}")
        if step > 0 and loss.item() > previous_loss and abs(loss.item() - previous_loss) > 1e-5 : # Check if loss isn't decreasing
             print(f"  Warning: Loss increased from {previous_loss:.4f} to {loss.item():.4f}")
        previous_loss = loss.item()


    print("Single batch training step test complete.")

except Exception as e:
    print(f"Error during training step test: {e}")
    # raise
