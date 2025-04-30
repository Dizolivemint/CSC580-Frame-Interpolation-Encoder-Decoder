import torch
import torch.nn as nn
import pandas as pd
from model_training.model_torch import EncoderDecoder
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure
import numpy as np

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
        if self.physics_type == "ball_motion":
            inputs = torch.tensor([row['mass'], row['angle'], row['friction']], dtype=torch.float32)
        elif self.physics_type == "camera_motion":
            inputs = torch.tensor([row['initial_velocity'], row['acceleration']], dtype=torch.float32)
        else:
            raise ValueError(f"Unknown physics type: {self.physics_type}")
        
        trajectory_np = np.array(row['trajectory'])  # shape: (T, H, W)
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

def hybrid_loss(pred, target, alpha=0.8, device='cuda'):
    mse_loss = nn.MSELoss()

    def format_tensor(x):
        if x.ndim == 2:
            # [B, T] -> [B, 1, 1, T]
            x = x.unsqueeze(1).unsqueeze(2)
        elif x.ndim == 3:
            if x.shape[1] == 1:
                # [B, 1, T] -> [B, 1, 1, T]
                x = x.view(x.shape[0], 1, 1, x.shape[-1])
            else:
                # [B, T, F] -> [B, 1, 1, T] after permute
                x = x.permute(0, 2, 1).unsqueeze(2)
        elif x.ndim == 4:
            pass  # [B, 1, H, W] — good
        else:
            raise ValueError(f"Unexpected shape: {x.shape}")
        return x

    pred = format_tensor(pred).to(device)
    target = format_tensor(target).to(device)

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    # Default fallback
    ssim_val = torch.tensor(0.0, device=device)

    # Decide whether to use SSIM
    B, C, H, W = pred.shape
    if H == 1:
        # 1D sequence
        try:
            pred_sq = squarify_1d_sequence(pred)
            target_sq = squarify_1d_sequence(target)
            ssim_val = compute_ssim(pred_sq, target_sq, device)
        except ValueError:
            # Cannot squarify — fallback to MSE-only loss
            pass
    else:
        # Already 2D frames (H > 1)
        ssim_val = compute_ssim(pred, target, device)


    return alpha * mse_loss(pred, target) + (1 - alpha) * (1 - ssim_val)
  
def train_model(
    physics_type, 
    hidden_size=64, 
    lr=0.002, 
    epochs=20, 
    early_stopping=False, 
    patience=5, 
    batch_size=256, 
    clip_grad=1.0
):
    data_path = f"data/{physics_type}_data.pkl"
    model_path = f"model_training/{physics_type}_model.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset and DataLoader
    dataset = PhysicsTrajectoryDataset(data_path, physics_type)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Model
    example_input, example_target = next(iter(dataloader))
    input_size = example_input.shape[1]
    timesteps, height, width = example_target.shape[1:]
    
    output_dim = example_target.shape[2] if example_target.ndim == 3 else 1

    model = EncoderDecoder(input_dim=input_size, hidden_size=hidden_size, output_shape=(height, width), output_seq_len=timesteps)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Early Stopping Setup
    best_loss = float('inf')
    wait = 0
    losses = []

    # Training Loop
    for epoch in range(epochs):
        total_loss = 0
        model.train()

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for batch_inputs, batch_targets in progress_bar:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

            optimizer.zero_grad()
            outputs = model(batch_inputs)

            loss = hybrid_loss(outputs, batch_targets, device=device)
            loss.backward()

            # Gradient Clipping (for stability)
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        scheduler.step(avg_loss)

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
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")
    return f"✅ Trained and saved {physics_type} model to {model_path}", losses

if __name__ == "__main__":
    train_model("ball_motion")
