import torch
import torch.nn as nn

class EncoderDecoder(nn.Module):
    def __init__(self, input_dim=3, hidden_size=64, output_seq_len=10, output_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU()
        )
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, 2)
        self.output_seq_len = output_seq_len

    def forward(self, inputs):
        encoded = self.encoder(inputs)
        decoder_input = encoded.unsqueeze(1).repeat(1, self.output_seq_len, 1)
        decoded, _ = self.decoder_lstm(decoder_input)
        outputs = self.output_layer(decoded)
        return outputs