from torch import nn


class AudioClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.conv_section = nn.Sequential(
            ConvBlock(1, 16, kernel_size=3, stride=1, padding=1),  # [B, 16, 256, 1024]
            ConvBlock(16, 16, kernel_size=3, stride=1, padding=1),  # [B, 16, 256, 1024]
            ConvBlock(16, 32, kernel_size=3, stride=2, padding=1),  # [B, 32, 128, 512]
            ConvBlock(32, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, 128, 512]
            ConvBlock(32, 64, kernel_size=3, stride=2, padding=1),  # [B, 64, 64, 256]
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),  # [B, 64, 64, 256]
            ConvBlock(64, 128, kernel_size=3, stride=2, padding=1),  # [B, 128, 32, 128]
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1),  # [B, 128, 32,128]
            ConvBlock(128, 256, kernel_size=3, stride=2, padding=1),  # [B, 256, 16, 64]
            nn.MaxPool2d(kernel_size=(16, 1)),  # [B, 256, 1, 64]
        )

        self.att_section = nn.Sequential(
            AttentionBlock(embed_dim=256, num_heads=4),
            AttentionBlock(embed_dim=256, num_heads=4),
            nn.AdaptiveAvgPool1d(1),
        )

        self.linear = nn.Linear(64, num_classes)

    def forward(self, tensor):
        tensor = self.conv_section(tensor).squeeze(2).transpose(1, 2)  # [B, 256, 64]
        tensor = self.att_section(tensor).squeeze(2)  # [B, 64]
        tensor = self.linear(tensor)
        return tensor


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.act = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, tensor):
        tensor = self.conv(tensor)
        tensor = self.act(tensor)
        tensor = self.norm(tensor)
        return tensor


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)

        self.W_Q = nn.Linear(embed_dim, embed_dim)
        self.W_K = nn.Linear(embed_dim, embed_dim)
        self.W_V = nn.Linear(embed_dim, embed_dim)
        self.multihead_att = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
        )

        self.act = nn.ReLU()
        self.norm2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, tensor):
        tensor = tensor + (
            self.multihead_att.forward(
                self.W_Q(tensor),
                self.W_K(tensor),
                self.W_V(tensor),
            )[0]
        )

        tensor = tensor + self.linear(self.norm2(tensor))

        return tensor
