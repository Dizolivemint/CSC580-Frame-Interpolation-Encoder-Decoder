import torch.nn as nn
from torchtyping import TensorType
from typeguard import typechecked
import torch

class EncoderDecoder(nn.Module):
    def __init__(self, input_dim=3, hidden_size=128, output_seq_len=10, output_shape=None):
        """
        Encoder-Decoder model for sequence-to-sequence prediction.

        Args:
            input_dim (int): Dimension of input features (e.g., [mass, angle, friction])
            hidden_size (int): Size of hidden representation
            output_seq_len (int): Number of time steps (T)
            output_shape (tuple, optional): Output frame shape (H, W) for image prediction mode.
                                          If None, model operates in coordinate prediction mode.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.output_seq_len = output_seq_len # T
        self.output_shape = output_shape  # (H, W) or None for coordinate mode
        self.is_coordinate_mode = output_shape is None

        # --- Encoder ---
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # --- Decoder LSTM ---
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size + 1,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=0.3
        )

        # --- Output Layer ---
        if self.is_coordinate_mode:
            # For coordinate prediction: output [x, y] coordinates
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 2),  # Predict [x, y] in normalized coordinates (0 to 1)
                nn.Sigmoid()
            )
        else:
            # For image prediction: output frames
            H, W = self.output_shape
            self.spatial_decoder_input_size = 8  # assumes hidden_size can reshape to [C, 8, 8]
            self.channels = hidden_size // (self.spatial_decoder_input_size ** 2)
            assert self.channels * self.spatial_decoder_input_size ** 2 == hidden_size, "hidden_size must be divisible by 64"
            
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_size, H * W),
                nn.Sigmoid()
            )

    @typechecked
    def forward(self, inputs: TensorType["B", "F"]) -> TensorType["B", "T", "H", "W"]:
        """
        Forward pass for both coordinate and image prediction modes.

        Args:
            inputs: [B, F] → input features (normalized mass, angle, friction)

        Returns:
            If coordinate mode: [B, T, 2] → sequence of (x,y) coordinates
            If image mode: [B, T, H, W] → sequence of predicted frames
        """
        B = inputs.shape[0]
        T = self.output_seq_len

        encoded = self.encoder(inputs)                  # [B, hidden]
        decoder_input = encoded.unsqueeze(1).repeat(1, T, 1)  # [B, T, hidden]
        timesteps = torch.linspace(0, 1, T, device=inputs.device).unsqueeze(0).repeat(B, 1).unsqueeze(-1)  # [B, T, 1]
        decoder_input = torch.cat([encoded.unsqueeze(1).repeat(1, T, 1), timesteps], dim=-1)  # [B, T, hidden + 1]

        lstm_out, _ = self.decoder_lstm(decoder_input)  # [B, T, hidden]

        if self.is_coordinate_mode:
            # Coordinate prediction mode
            output = self.output_layer(lstm_out)  # [B, T, 2]
        else:
            # Image prediction mode
            H, W = self.output_shape
            output = self.output_layer(lstm_out)  # [B, T, H*W]
            output = output.view(B, T, H, W)  # Reshape to [B, T, H, W]

        return output
