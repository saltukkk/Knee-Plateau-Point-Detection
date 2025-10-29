# validate.py
import torch
from tqdm import tqdm
from .loss import MultifacetedLoss # Assuming MultifacetedLoss is in the same directory or correctly pathed

class Validator:
    def __init__(self, config, val_loader, model):
        self.config = config
        self.val_loader = val_loader
        self.device = config.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        
        # Updated MultifacetedLoss instantiation
        self.criterion = MultifacetedLoss(
            config_landmarks=config['landmarks'],
            loss_wing_w=config['loss_wing_w'],
            loss_wing_epsilon=config['loss_wing_epsilon'],
            landmark_loss_weights=config['landmark_loss_weights']
        ).to(self.device)
        self.best_val_loss = float('inf')

    def validate(self):
        self.model.eval()
        total_loss_epoch = 0.0
        if not self.val_loader: # Handle case where val_loader might be None if no validation data
            print("Warning: Validation loader not available. Skipping validation.")
            return float('inf'), False # Return high loss, not best

        progress_bar = tqdm(self.val_loader, desc='Validation', leave=False)
        
        with torch.no_grad():
            for batch_idx, data in enumerate(progress_bar):
                if data is None or (isinstance(data, (list, tuple)) and (data[0] is None or data[1] is None)):
                    print(f"Skipping problematic batch {batch_idx} in validation due to loading error.")
                    continue
                
                images, landmarks = data
                images, landmarks = images.to(self.device), landmarks.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, landmarks)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN or Inf loss detected during validation at batch {batch_idx}.")
                    # Decide how to handle, e.g. add a very large number to total_loss or skip
                    # For now, let's skip adding it to prevent propagation of NaN/Inf
                    continue
                
                total_loss_epoch += loss.item()
                progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss_epoch / len(self.val_loader) if self.val_loader and len(self.val_loader) > 0 else float('inf')
        
        is_best = avg_loss < self.best_val_loss
        if is_best and avg_loss != float('inf'): # only update if not inf
            self.best_val_loss = avg_loss
        
        print(f'Validation Avg Loss: {avg_loss:.4f}')
        return avg_loss, is_best