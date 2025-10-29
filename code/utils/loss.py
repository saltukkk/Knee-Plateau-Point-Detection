# utils/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class WingLoss(nn.Module):
    def __init__(self, w=3.0, epsilon=1.5):
        super(WingLoss, self).__init__()
        self.w = torch.tensor(w, dtype=torch.float32) 
        self.epsilon = torch.tensor(epsilon, dtype=torch.float32)
        self.C = self.w - self.w * torch.log(1.0 + self.w / self.epsilon)

    def forward(self, prediction, target):
        y_abs_diff = torch.abs(prediction - target)
        
        # Condition for applying Wing Loss's non-linear part or linear part
        # Ensure device consistency if self.w, self.epsilon, self.C are not on the same device as y_abs_diff
        if self.w.device != y_abs_diff.device:
            self.w = self.w.to(y_abs_diff.device)
            self.epsilon = self.epsilon.to(y_abs_diff.device)
            self.C = self.C.to(y_abs_diff.device)
            
        loss = torch.where(
            y_abs_diff < self.w,
            self.w * torch.log(1.0 + y_abs_diff / self.epsilon),
            y_abs_diff - self.C
        )
        
        # prediction/target shape: (batch_size, num_total_coordinates)
        # num_total_coordinates = num_landmarks * 2 (for x, y)
        # We want loss per landmark, then mean over batch.
        
        # Reshape to (batch_size, num_landmarks, 2)
        loss_reshaped = loss.view(loss.size(0), -1, 2) 
        
        # Mean loss across x, y for each landmark: (batch_size, num_landmarks)
        loss_per_landmark_in_batch = torch.mean(loss_reshaped, dim=2)
        
        # Mean loss across all landmarks in the batch: (batch_size)
        # This returns a loss value for each item in the batch.
        # The final reduction (e.g., mean over batch) is typically done outside the loss function,
        # or the loss function returns a scalar. Let's return mean over landmarks for now.
        return loss_per_landmark_in_batch # Shape: (batch_size, num_landmarks)


class MultifacetedLoss(nn.Module):
    def __init__(self, config_landmarks, loss_wing_w=3.0, loss_wing_epsilon=1.5, landmark_loss_weights=None):
        super(MultifacetedLoss, self).__init__()
        self.wing_loss_fn = WingLoss(loss_wing_w, loss_wing_epsilon)
        self.config_landmarks = config_landmarks # e.g., [{"name": "Notch", "coordinates": ["x", "y"]}]
        self.landmark_loss_weights = landmark_loss_weights if landmark_loss_weights else {}

        if not self.config_landmarks:
            raise ValueError("Landmarks must be defined in the configuration for MultifacetedLoss.")

    def forward(self, prediction, target):
        # prediction & target shape: (batch_size, num_total_coordinates)
        # num_total_coordinates = sum(len(lm['coordinates']) for lm in self.config_landmarks)
        
        # WingLoss returns loss per landmark: (batch_size, num_landmarks)
        all_landmarks_wing_loss = self.wing_loss_fn(prediction, target)
        
        total_weighted_loss = torch.zeros(prediction.size(0), device=prediction.device) # (batch_size)
        
        # For detailed printing (optional)
        # loss_details_str = []

        for i, landmark_info in enumerate(self.config_landmarks):
            landmark_name = landmark_info['name']
            weight = self.landmark_loss_weights.get(landmark_name, 1.0) # Default weight 1.0 if not specified
            
            current_landmark_loss = all_landmarks_wing_loss[:, i] # Loss for the i-th landmark for all items in batch
            weighted_landmark_loss = weight * current_landmark_loss
            total_weighted_loss += weighted_landmark_loss
            
            # loss_details_str.append(f"{landmark_name}_loss: {current_landmark_loss.mean().item() * weight:.5f}")

        # print(" * ".join(loss_details_str) + f" \nTotal batch mean loss: {total_weighted_loss.mean().item():.5f}")
        
        # Return the mean loss over the batch
        return total_weighted_loss.mean()