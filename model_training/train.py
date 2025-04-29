import torch
import torch.nn as nn
import pandas as pd
from model_training.model_torch import EncoderDecoder

def train_model(physics_type, hidden_size=64, lr=0.001, epochs=20, early_stopping=False, patience=5):
    data_path = f"data/{physics_type}_data.pkl"
    model_path = f"model_training/{physics_type}_model.pth"
    
    # Load dataset
    df = pd.read_pickle(data_path)
    
    if physics_type == "ball_motion":
        X = torch.tensor(df[['mass', 'angle', 'friction']].values, dtype=torch.float32)
    elif physics_type == "camera_motion":
        X = torch.tensor(df[['initial_velocity', 'acceleration']].values, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown physics type: {physics_type}")
    
    y = torch.tensor(df['trajectory'].tolist(), dtype=torch.float32)

    # Model
    model = EncoderDecoder(output_seq_len=y.shape[1], hidden_size=hidden_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Early Stopping
    best_loss = float('inf')
    wait = 0
    losses = []
    
    # Training Loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        losses.append(current_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {current_loss:.6f}")
        
        # Early Stopping Logic
        if early_stopping:
            if current_loss < best_loss - 1e-6:  # small improvement threshold
                best_loss = current_loss
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
