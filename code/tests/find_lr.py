# find_lr.py
import argparse
import json
import torch
import os
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- REUSING YOUR EXISTING MODULES ---
# We import all the necessary components from your own files.
from utils.dataloader import preprocess_data, create_dataloaders
from utils.models import UNet, RAUNet, CRAUNet, ResNeXt50
from utils.loss import MultifacetedLoss

def load_config(config_path):
    """Loads config and sets the device. Copied from your main.py."""
    with open(config_path, 'r') as file:
        config = json.load(file)
    config['device'] = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    return config

# --- MODIFICATION 1: THE CORE LR FINDER LOGIC ---
# This function replaces the Trainer class for this specific test. It contains the
# loop that increases the learning rate at every single batch.
def lr_find_epoch(model, loader, optimizer, criterion, start_lr, end_lr, num_steps, device):
    """
    Performs one epoch of training while linearly increasing the learning rate.
    This is the heart of the LR Range Test.
    """
    model.train()
    
    # Lists to store the learning rates and corresponding losses
    lrs = []
    losses = []
    
    # We calculate a multiplier for an exponential increase in LR.
    # This gives us evenly spaced points on a logarithmic scale.
    lr_multiplier = (end_lr / start_lr) ** (1 / (num_steps - 1))
    current_lr = start_lr
    
    # Smoothed loss helps in visualizing a clearer trend.
    avg_loss = 0.0
    beta = 0.98  # Smoothing factor

    # Wrap the loader with tqdm for a progress bar
    progress_bar = tqdm(loader, desc='LR Range Test', leave=True)
    
    for batch_idx, data in enumerate(progress_bar):
        # Your dataloader might return None for bad data, so we handle it.
        if data is None or (isinstance(data, (list, tuple)) and (data[0] is None or data[1] is None)):
            print(f"Skipping problematic batch {batch_idx}.")
            continue
            
        images, landmarks = data
        images, landmarks = images.to(device), landmarks.to(device)
        
        # --- Manually set the learning rate for this batch ---
        # This is the key part of the test.
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Standard training step
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, landmarks)
        
        # Stop if loss explodes to prevent running a useless test
        if torch.isnan(loss) or loss.item() > 4 * (losses[0] if losses else loss.item()):
             print(f"Loss exploded at batch {batch_idx}. Stopping test.")
             break
        
        loss.backward()
        optimizer.step()

        # --- Log results and update LR for the next batch ---
        lrs.append(current_lr)
        
        # Compute smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta**(batch_idx + 1))
        losses.append(smoothed_loss)
        
        progress_bar.set_postfix(loss=smoothed_loss, lr=current_lr)
        
        # Update the learning rate for the next iteration
        current_lr *= lr_multiplier

    return lrs, losses


def main(config):
    """
    Main function to set up and run the LR Range Test.
    This is an adapted version of your original main.py.
    """
    print(f"--- Starting Learning Rate Range Test ---")
    print(f"Using device: {config['device']}")

    # --- MODIFICATION 2: SETUP IS ALMOST THE SAME, BUT NO VALIDATOR/TRAINER ---
    # We reuse your exact data loading and model instantiation logic.
    unified_df = preprocess_data(config)
    # We only need the training data for this test.
    dataloaders = create_dataloaders(unified_df, config, base_image_dir=config.get("base_image_dir", "./images"))
    train_loader = dataloaders['Train']

    out_features = sum(len(lm.get('coordinates', [])) for lm in config['landmarks'])
    print(f"Number of output features: {out_features}")

    # Instantiate the model based on the configuration
    model_map = {"UNet": UNet, "RAUNet": RAUNet, "CRAUNet": CRAUNet, "ResNeXt50": ResNeXt50}
    if config['model_type'] in model_map:
        model = model_map[config['model_type']](in_channels=1, out_features=out_features)
    else:
        raise ValueError(f"Invalid model type specified: {config['model_type']}")
    
    model = model.to(config['device'])
    
    # Instantiate your loss function directly
    criterion = MultifacetedLoss(
        config_landmarks=config['landmarks'],
        loss_wing_w=config['loss_wing_w'],
        loss_wing_epsilon=config['loss_wing_epsilon'],
        landmark_loss_weights=config['landmark_loss_weights']
    ).to(config['device'])

    # --- MODIFICATION 3: OPTIMIZER SETUP ---
    # We initialize the optimizer with a placeholder LR, as it will be
    # set manually in the loop. CRUCIALLY, WE DO NOT CREATE THE SCHEDULER.
    start_lr = 1e-7 # The test will start here
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)

    # Define test parameters
    end_lr = 1.0   # A high value to ensure we see the loss explode
    num_steps = len(train_loader) # Number of batches in one epoch

    # --- MODIFICATION 4: EXECUTION AND PLOTTING ---
    # Run the test
    lrs, losses = lr_find_epoch(model, train_loader, optimizer, criterion, start_lr, end_lr, num_steps, config['device'])

    # Plot the results
    print("Plotting results...")
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Smoothed Loss")
    plt.title("Learning Rate Range Test")
    plt.grid(True, which="both", ls="--")
    plt.xticks([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    # Save the plot to a file
    plot_filename = "lr_range_test_plot.png"
    plt.savefig(plot_filename)
    print(f"--- LR Range Test Finished. Plot saved to {plot_filename} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Learning Rate Range Test.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration JSON file')
    args = parser.parse_args()
    config_data = load_config(args.config)
    main(config_data)
