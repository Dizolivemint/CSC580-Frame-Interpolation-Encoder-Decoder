import torch
import torch.nn as nn
import pandas as pd
from model_torch import EncoderDecoder

# Load dataset
train_df = pd.read_pickle('data/training_data.pkl')
X = torch.tensor(train_df[['mass', 'angle', 'friction']].values, dtype=torch.float32)
y = torch.tensor(train_df['trajectory'].tolist(), dtype=torch.float32)

# Model
model = EncoderDecoder(output_seq_len=y.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 20
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

# Save model
torch.save(model.state_dict(), 'model_training/model.pth')
print("\u2705 Model saved to model_training/model.pth")