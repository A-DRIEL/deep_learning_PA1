import torch
import torch.nn as nn

from src.models.resunet import ResidualBlock


class ResUNetPoolIndices(nn.Module):
    """
    Mesmo encoder (ResidualBlock) do ResUNet, mas a recuperação de
    resolução usa POOL INDICES (estilo SegNet) em vez de skip
    connections.

    IMPORTANTE: MaxUnpool2d exige que o tensor sendo despoolado tenha
    o MESMO número de canais do tensor que gerou os índices (os
    índices são "por canal"). Por isso, toda mudança de canais
    acontece ANTES do unpool correspondente, nunca depois -- cada
    unpool(x, idxN) só é chamado quando x já está com o número de
    canais que foi poolado naquele nível.
    """

    def __init__(self, in_channels=3, num_classes=1, base_channels=32):
        super().__init__()
        c = base_channels

        self.enc1 = ResidualBlock(in_channels, c)       # -> c
        self.enc2 = ResidualBlock(c, c * 2)              # -> c*2
        self.enc3 = ResidualBlock(c * 2, c * 4)          # -> c*4
        self.enc4 = ResidualBlock(c * 4, c * 8)          # -> c*8

        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)

        # bottleneck expande (capacidade extra), depois reduz de volta
        # para c*8 ANTES do primeiro unpool, pra bater com idx4
        self.bottleneck = ResidualBlock(c * 8, c * 16)
        self.bottleneck_reduce = ResidualBlock(c * 16, c * 8)

        self.dec4 = ResidualBlock(c * 8, c * 4)  # roda DEPOIS do unpool com idx4
        self.dec3 = ResidualBlock(c * 4, c * 2)  # roda DEPOIS do unpool com idx3
        self.dec2 = ResidualBlock(c * 2, c)      # roda DEPOIS do unpool com idx2
        self.dec1 = ResidualBlock(c, c)          # roda DEPOIS do unpool com idx1

        self.out_conv = nn.Conv2d(c, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1, idx1 = self.pool(e1)          # idx1: c canais

        e2 = self.enc2(p1)
        p2, idx2 = self.pool(e2)          # idx2: c*2 canais

        e3 = self.enc3(p2)
        p3, idx3 = self.pool(e3)          # idx3: c*4 canais

        e4 = self.enc4(p3)
        p4, idx4 = self.pool(e4)          # idx4: c*8 canais

        b = self.bottleneck(p4)                    # c*16 canais
        b = self.bottleneck_reduce(b)               # volta pra c*8 <- bate com idx4

        d4 = self.unpool(b, idx4, output_size=e4.shape)   # c*8 canais
        d4 = self.dec4(d4)                                 # reduz pra c*4 <- bate com idx3

        d3 = self.unpool(d4, idx3, output_size=e3.shape)  # c*4 canais
        d3 = self.dec3(d3)                                 # reduz pra c*2 <- bate com idx2

        d2 = self.unpool(d3, idx2, output_size=e2.shape)  # c*2 canais
        d2 = self.dec2(d2)                                 # reduz pra c <- bate com idx1

        d1 = self.unpool(d2, idx1, output_size=e1.shape)  # c canais
        d1 = self.dec1(d1)

        return self.out_conv(d1)