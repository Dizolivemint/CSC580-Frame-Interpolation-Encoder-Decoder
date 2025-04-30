import torch
import torch.nn as nn
from torchtyping import TensorType
from typeguard import typechecked

class EncoderDecoder(nn.Module):
    def __init__(self, input_dim=3, hidden_size=64, output_seq_len=10, output_shape=(64, 64)):
        """
        Encoder-Decoder model for sequence-to-sequence prediction.

        Args:
            input_dim (int): Dimension of input features (e.g., [mass, angle, friction])
            hidden_size (int): Size of hidden representation
            output_seq_len (int): Number of time steps (T)
            output_shape (tuple): Output frame shape (H, W)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.output_seq_len = output_seq_len # T
        self.output_shape = output_shape  # (H, W)
        H, W = self.output_shape
        T = self.output_seq_len
        self.output_dim = H * W  # H * W
        
        print(f"Decoder output_dim = {self.output_dim}")  # should print 4096

        # --- Encoder ---
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # --- Decoder ---
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Map LSTM output → flattened image
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.output_dim)  # Map each time step to a frame
        )

    @typechecked
    def forward(self, inputs: TensorType["B", "F"]) -> TensorType["B", "T", "H", "W"]:
        """
        Args:
            inputs: [B, F] → input features

        Returns:
            output: [B, T, H, W]
        """
        B = inputs.shape[0]
        T = self.output_seq_len
        H, W = self.output_shape

        encoded = self.encoder(inputs)  # [B, hidden]
        decoder_input = encoded.unsqueeze(1).repeat(1, T, 1)  # [B, T, hidden]

        lstm_out, _ = self.decoder_lstm(decoder_input)  # [B, T, hidden]

        # Flatten for Sequential block
        lstm_out_flat = lstm_out.reshape(B * T, -1)              # [B*T, hidden]
        frame_flat = self.output_layer(lstm_out_flat)          # [B*T, H*W]
        
        expected = B * T * H * W
        actual = frame_flat.numel()

        assert actual == expected, "Mismatch in total output elements!"
    
        output = frame_flat.view(B, T, H, W)                   # [B, T, H, W]

        return output