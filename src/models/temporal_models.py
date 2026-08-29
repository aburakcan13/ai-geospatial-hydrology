"""
Spatio-Temporal ConvLSTM for Surface Water Dynamics & Flood Hydro-Forecasting
AI-Driven Geospatial Hydrology (Springer Nature)
"""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x: torch.Tensor, hidden_state: tuple) -> tuple:
        h_prev, c_prev = hidden_state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(gates, self.hidden_channels, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class HydroConvLSTM(nn.Module):
    def __init__(self, in_channels: int = 4, hidden_channels: int = 32, num_layers: int = 2):
        """Sequentially processes historical satellite & meteorological sequences (T, B, C, H, W)."""
        super().__init__()
        self.hidden_channels = hidden_channels
        self.cell = ConvLSTMCell(in_channels, hidden_channels)
        self.head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        seq_len, b, _, h, w = sequence.shape
        device = sequence.device
        h_t = torch.zeros(b, self.hidden_channels, h, w, device=device)
        c_t = torch.zeros(b, self.hidden_channels, h, w, device=device)

        for t in range(seq_len):
            h_t, c_t = self.cell(sequence[t], (h_t, c_t))

        return torch.sigmoid(self.head(h_t))
