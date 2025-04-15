import torch
from torch import nn


class AudioClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.conv_section = nn.Sequential(  # [B, C, 256, 1024]
            ConvBlock(1, 32, kernel_size=7, padding=3),
            nn.MaxPool2d(kernel_size=2),  # [B, C, 128, 512]
            ConvBlock(32, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=(2, 1)),  # [B, C, 64, 512]
            ConvBlock(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),  # [B, C, 32, 256]
            ConvBlock(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=(2, 1)),  # [B, C, 16, 256]
            ConvBlock(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),  # [B, C, 8, 128]
            ConvBlock(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=(2, 1)),  # [B, C, 4, 128]
            ConvBlock(128, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),  # [B, C, 2, 64]
            ConvBlock(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=(2, 1)),  # [B, C, 1, 64]
        )

        self.pre_transformer_norm = nn.LayerNorm(256)

        self.pe = PositionalEncoding(embed_dim=256)

        self.enc_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(self.enc_layer, num_layers=4)

        self.linear = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.Dropout(0.1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, tensor):
        tensor = self.conv_section(tensor).squeeze(2).permute(0, 2, 1)
        tensor = self.pre_transformer_norm(tensor)
        tensor = self.pe(tensor)
        tensor = self.transformer(tensor).permute(0, 2, 1)
        tensor = self.linear(tensor)
        return tensor


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dropout=0.1):
        super().__init__()
        self.conv_seq = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.Dropout(dropout),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
        )
        self.res_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, tensor):
        res_tensor = self.res_conv(tensor) if self.res_conv else tensor
        tensor = self.conv_seq(tensor)
        tensor = self.norm(res_tensor + tensor)
        tensor = self.act(tensor)
        return tensor


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=256):
        super().__init__()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])
        pe = pe.unsqueeze(0)

        self.register_buffer("pe_buffer", pe, persistent=False)

        self.pe_scale = nn.Parameter(torch.tensor([0.2]))

    def forward(self, tensor):
        tensor = tensor + self.pe_scale * self.pe_buffer[:, : tensor.size(1), :]
        return tensor
