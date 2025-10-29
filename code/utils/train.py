# train.py
import torch
from tqdm import tqdm
from .loss import MultifacetedLoss # Assuming MultifacetedLoss is in the same directory or correctly pathed
from torch.optim.lr_scheduler import CyclicLR

class Trainer:
    def __init__(self, config, train_loader, model=None):
        self.config = config
        self.train_loader = train_loader
        self.device = config.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device) if model else None # Ensure model is on device

        if not self.model:
            raise ValueError("Model not provided to Trainer.")

        # Updated MultifacetedLoss instantiation
        self.criterion = MultifacetedLoss(
            config_landmarks=config['landmarks'], # Pass landmark definitions
            loss_wing_w=config['loss_wing_w'],
            loss_wing_epsilon=config['loss_wing_epsilon'],
            landmark_loss_weights=config['landmark_loss_weights']
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.scheduler = CyclicLR(self.optimizer, 
                                  base_lr=config['base_lr'], 
                                  max_lr=config['max_lr'],
                                  step_size_down=config.get('step_size_down', 2000), # Provide default if missing
                                  mode='triangular',
                                  cycle_momentum=False)

    def train(self, epoch):
        self.model.train()
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]}', leave=False)
        total_loss_epoch = 0.0
        
        for batch_idx, data in enumerate(progress_bar):
            if data is None or (isinstance(data, (list, tuple)) and (data[0] is None or data[1] is None)):
                print(f"Skipping problematic batch {batch_idx} in epoch {epoch+1} due to loading error.")
                continue # Skip batch if data loading failed
            
            images, landmarks = data
            images, landmarks = images.to(self.device), landmarks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, landmarks)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf loss detected at epoch {epoch+1}, batch {batch_idx}. Skipping backward pass for this batch.")
                # Potentially log more details about inputs/outputs here
                continue

            loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # According to PyTorch docs, CyclicLR should usually be stepped each batch.
            self.scheduler.step() 
            
            total_loss_epoch += loss.item()
            progress_bar.set_postfix(loss=loss.item(), lr=self.scheduler.get_last_lr()[0])
        
        avg_loss = total_loss_epoch / len(self.train_loader) if len(self.train_loader) > 0 else 0
        # The scheduler step here (end of epoch) depends on the specific CyclicLR mode and intent.
        # If step_size_up/down are in terms of batches, then stepping per batch is correct.
        # If they are in terms of epochs, then scheduler.step() should be at the end of the epoch.
        # Given 'step_size_down', it's likely batch-wise.
        # self.scheduler.step() # Potentially remove if only stepping per batch
        print(f"Epoch {epoch+1} Train Avg Loss: {avg_loss:.4f}, Current LR: {self.scheduler.get_last_lr()[0]:.1e}")
        return avg_loss