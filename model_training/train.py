import torch
import torch.nn as nn
import pandas as pd
from model_training.model_torch import EncoderDecoder

def train_model(physics_type, hidden_size=64, lr=0.001, epochs=20):
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

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")
