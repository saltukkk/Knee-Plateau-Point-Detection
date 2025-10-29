# main.py
import argparse
import json
import pandas as pd
import torch
import os

from utils.dataloader import create_dataloaders, preprocess_data
from utils.train import Trainer
from utils.validate import Validator
from utils.models import UNet, RAUNet, CRAUNet, ResNeXt50 # GuptaModel is not in models.py, ensure it's defined or remove if not used

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    config['device'] = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    return config

def main(config):
    print(f"Using device: {config['device']}")

    unified_df = preprocess_data(config) # preprocess_data in dataloader.py uses config['split_file'] and config['details_file']
    dataloaders = create_dataloaders(unified_df, config, base_image_dir="./images")

    train_loader, val_loader = dataloaders['Train'], dataloaders['Validation']

    # Determine the number of output features based on the landmarks configuration
    out_features = 0
    if 'landmarks' in config:
        for landmark in config['landmarks']:
            out_features += len(landmark.get('coordinates', []))
    else:
        # Fallback or error for old configurations (optional)
        # For Pectoralis (2 points * 2 coords) + Nipple (1 point * 2 coords) = 6
        out_features_map_legacy = { 'pectoralis': 4, 'nipple': 2, 'all': 6 }
        if config.get('target_task') in out_features_map_legacy: # Support old config if target_task exists
            out_features = out_features_map_legacy[config['target_task']]
        else:
            raise ValueError("Invalid landmark configuration in config file.")
    
    if out_features == 0:
        raise ValueError("No landmarks defined, out_features is 0.")

    print(f"Number of output features: {out_features}")

    # Instantiate the model based on the configuration
    if config['model_type'] == "UNet":
        model = UNet(in_channels=1, out_features=out_features)
    elif config['model_type'] == "RAUNet":
        model = RAUNet(in_channels=1, out_features=out_features)
    elif config['model_type'] == "CRAUNet":
        model = CRAUNet(in_channels=1, out_features=out_features)
    elif config['model_type'] == "ResNeXt50":
        model = ResNeXt50(in_channels=1, out_features=out_features)
    else:
        raise ValueError("Invalid model type specified")
    
    model = model.to(config['device'])
    
    trainer = Trainer(config, train_loader, model)
    validator = Validator(config, val_loader, model)

    best_model_state = None
    best_val_loss = float('inf')
    metrics = []

    for epoch in range(config['num_epochs']):
        train_loss = trainer.train(epoch)
        validation_loss, is_best = validator.validate()
        current_lr = trainer.scheduler.get_last_lr()[0] # Safe if scheduler is always used

        metrics.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'validation_loss': validation_loss,
            'learning_rate': current_lr
        }) 

        if is_best:
            best_model_state = model.state_dict()
            best_val_loss = validation_loss
            print(f"New best validation loss: {best_val_loss:.4f} at epoch {epoch+1}")


    if best_model_state:
        os.makedirs(os.path.dirname(config['best_model_path']), exist_ok=True)
        torch.save(best_model_state, config['best_model_path'])
        print(f"Best model saved to {config['best_model_path']}, Validation Loss: {best_val_loss}")

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv('training_metrics.csv', index=False)
    print("Training metrics saved to training_metrics.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training experiments with different configurations")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration JSON file')
    
    args = parser.parse_args()
    config_data = load_config(args.config)
    main(config_data)
