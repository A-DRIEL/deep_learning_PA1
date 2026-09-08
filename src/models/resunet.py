import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # projeção do atalho quando o número de canais muda
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResUNet(nn.Module):
    """
    ResUNet: encoder-decoder simétrico com blocos residuais e
    skip connections (concatenação), no estilo U-Net.

    Entrada: (B, in_channels, H, W)
    Saída: (B, num_classes, H, W) -- logits (sem sigmoid/softmax)
    """

    def __init__(self, in_channels=3, num_classes=3, base_channels=32):
        super().__init__()
        c = base_channels

        # --- encoder ---
        self.enc1 = ResidualBlock(in_channels, c)
        self.enc2 = ResidualBlock(c, c * 2)
        self.enc3 = ResidualBlock(c * 2, c * 4)
        self.enc4 = ResidualBlock(c * 4, c * 8)

        self.pool = nn.MaxPool2d(2)

        # --- bottleneck ---
        self.bottleneck = ResidualBlock(c * 8, c * 16)

        # --- decoder ---
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = ResidualBlock(c * 16, c * 8)  # concat: c*8 (up) + c*8 (skip)

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = ResidualBlock(c * 8, c * 4)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = ResidualBlock(c * 4, c * 2)

        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = ResidualBlock(c * 2, c)

        self.out_conv = nn.Conv2d(c, num_classes, 1)

    def forward(self, x):
        # encoder, guardando saídas para as skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        # decoder, concatenando com o nível correspondente do encoder
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)