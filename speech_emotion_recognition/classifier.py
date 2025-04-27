import torch
from torch import nn


class AudioClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.conv_section = nn.Sequential(  # [B, C, 256, 1024]
            nn.Conv2d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 1)),  # [B, C, 64, 1024]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # [B, C, 32, 512]
            nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 1)),  # [B, C, 8, 512]
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # [B, C, 4, 256]
            nn.Dropout2d(0.2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 1)),  # [B, C, 1, 256]
        )

        self.pre_transformer_norm = nn.LayerNorm(512)

        self.pe = PositionalEncoding(embed_dim=512)

        self.enc_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=4,
            dim_feedforward=1024,
            dropout=0.2,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            self.enc_layer,
            num_layers=4,
            norm=nn.LayerNorm(512),
        )

        self.linear = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Softmax(),
        )

    def forward(self, tensor):
        tensor = self.conv_section(tensor).squeeze(2).transpose(1, 2)
        tensor = self.pre_transformer_norm(tensor)
        tensor = self.pe(tensor)
        tensor = self.transformer(tensor).transpose(1, 2)
        tensor = self.linear(tensor)
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
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe_buffer", pe, persistent=False)

        self.pe_scale = nn.Parameter(torch.tensor([0.1]))

    def forward(self, tensor):
        tensor = tensor + self.pe_scale * self.pe_buffer[:, : tensor.size(1), :]
        return tensor
