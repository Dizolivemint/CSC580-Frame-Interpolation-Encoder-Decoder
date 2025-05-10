import torch
import torch.nn as nn
import pandas as pd
from model_training.model_torch import EncoderDecoder
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure
import numpy as np
from config import normalize_input, get_input_fields
from utils.path_utils import resolve_path
import os

class PhysicsTrajectoryDataset(Dataset):
    def __init__(self, filepath, physics_type):
        self.df = pd.read_pickle(filepath)
        self.physics_type = physics_type
        
        # Detect the number of time steps from the first sample
        sample_trajectory = self.df.iloc[0]['trajectory']
        self.time_steps = len(sample_trajectory)
        
        # Sanity check
        if not all(len(traj) == self.time_steps for traj in self.df['trajectory']):
            raise ValueError("Not all trajectories have the same number of time steps!")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fields = get_input_fields(self.physics_type)
        inputs = torch.tensor(normalize_input(self.physics_type, *[row[f] for f in fields]), dtype=torch.float32)
        
        trajectory_np = np.array(row['trajectory'])
        trajectory = torch.from_numpy(trajectory_np).float()
        

        return inputs, trajectory

def compute_ssim(pred, target, device):
    """
    Computes SSIM between predicted and target tensors.
    If the input is 1D (H=1), attempts to squarify.
    Falls back to MSE-only if squarify fails.
    """
    B, C, H, W = pred.shape
    try:
        if H == 1:
            # 1D sequence → squarify to [B, C, S, S]
            pred_sq = squarify_1d_sequence(pred)
            target_sq = squarify_1d_sequence(target)
            return structural_similarity_index_measure(pred_sq, target_sq, data_range=1.0)
        else:
            # Already 2D image-like
            return structural_similarity_index_measure(pred, target, data_range=1.0)
    except ValueError:
        # Fallback: SSIM not computable
        return torch.tensor(0.0, device=device)
      
def squarify_1d_sequence(x):
    """Reshape 1D sequence into square 2D frames for SSIM."""
    B, C, _, T = x.shape
    S = int(T ** 0.5)
    if S * S != T:
        raise ValueError(f"T={T} must be a perfect square for squarify, got {T}.")
    return x.view(B, C, S, S)

def bce_dot_loss(pred, target, device):
    """
    Compute binary cross entropy loss between predicted and target dot-maps.
    Assumes pred and target have shape [B, T, H, W] with values in [0, 1].
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    # Compute per-pixel weights: 6.0 where target=1, 1.0 where target=0
    weight = (target * 10.0 + 1.0)  # weight = 610 where dot is, 1 elsewhere
    loss = nn.functional.binary_cross_entropy(pred, target, weight=weight, reduction='mean')
    return loss
  
def train_model(
    physics_type, 
    hidden_size=128, 
    lr=0.002, 
    epochs=20, 
    early_stopping=False, 
    patience=5, 
    batch_size=256, 
    clip_grad=1.0
):
    dataset_path = resolve_path(f"{physics_type}_data.pkl")
    model_path = resolve_path(f"{physics_type}_model.pth", write_mode=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset and DataLoader
    dataset = PhysicsTrajectoryDataset(dataset_path, physics_type)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Inspect shape of sample
    sample_input, sample_target = dataset[0]
    timesteps, coord_dims = sample_target.shape
    assert coord_dims == 2, "Expected target shape [T, 2] for (x, y) coordinates"

    model = EncoderDecoder(
        input_dim=sample_input.shape[0],
        hidden_size=hidden_size,
        output_seq_len=timesteps,
        output_shape=None  # Not used in coordinate mode
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    loss_fn = nn.MSELoss()
    best_loss = float('inf')
    wait = 0
    losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for batch_inputs, batch_targets in progress_bar:
            batch_inputs = batch_inputs.to(device)            # [B, F]
            batch_targets = batch_targets.to(device)          # [B, T, 2]

            optimizer.zero_grad()
            outputs = model(batch_inputs)                     # [B, T, 2]
            loss = loss_fn(outputs, batch_targets)

            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        scheduler.step(avg_loss)

        # Debug: check one sample trajectory
        # if epoch % 5 == 0:
        #     with torch.no_grad():
        #         coords = outputs[0].detach().cpu().numpy()
        #         print(f"🔍 Predicted trajectory sample (epoch {epoch+1}):")
        #         for t, (x, y) in enumerate(coords[:5]):
        #             print(f"t={t}: (x={x:.3f}, y={y:.3f})")

        # Early Stopping Logic
        if early_stopping:
            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"⏹️ Early stopping at epoch {epoch+1}")
                    break

    # Save model
    torch.save({
        'model_state': model.state_dict(),
        'input_dim': sample_input.shape[0],
        'output_seq_len': timesteps,
    }, model_path)
    print(f"✅ Model saved to {model_path}")
    return f"✅ Trained and saved {physics_type} model to {model_path}", losses

if __name__ == "__main__":
    train_model("ball_motion")
