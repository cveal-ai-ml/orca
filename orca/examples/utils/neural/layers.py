"""
Author: Charlie
Purpose: Custom network layers / blocks
"""


import torch
import torch.nn as nn


class Residual(nn.Module):

    def __init__(self, block):

        super().__init__()

        self.block = block

    def forward(self, x):

        return x + self.block(x)


class Upsample(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=2, padd=1, out_padd=1):

        super().__init__()

        self.arch = nn.ConvTranspose2d(in_channels, out_channels,
                                       kernel_size, stride, padd, out_padd)

    def forward(self, x):

        return self.arch(x)


class D_block(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padd=1, out_padd=0):

        super().__init__()

        layers = [nn.ConvTranspose2d(in_channels, out_channels,
                                     kernel_size, stride, padd, out_padd),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU()]

        self.block = nn.Sequential(*layers)

    def forward(self, x):

        return self.block(x)


class Decoder(nn.Module):

    def __init__(self, data_shape, in_channels, using_standard=0):

        super().__init__()

        channels, max_size, _, = data_shape

        # Calculate: Number Decoder Blocks
        # - Assuming a construction from [N, C, 1, 1] shape

        f_size = 1
        num_blocks = 0
        while f_size <= max_size:
            f_size = f_size * 2
            num_blocks = num_blocks + 1

        num_blocks = num_blocks - 1

        # Construct: Network Decoder

        out_channels = in_channels

        layers = []
        for i in range(num_blocks):

            if i == (num_blocks - 1):
                out_channels = channels
                kernel_size, stride, padd, out_padd = 3, 2, 1, 1
                final = nn.ConvTranspose2d(in_channels, out_channels,
                                           kernel_size, stride, padd, out_padd)
                layers.append(final)

            else:

                layers.append(Residual(D_block(in_channels=out_channels,
                                               out_channels=out_channels)))

                out_channels = out_channels // 2

                layers.append(Upsample(in_channels, out_channels))

            in_channels = in_channels // 2

        if using_standard:
            layers.append(nn.Tanh())
        else:
            layers.append(nn.Sigmoid())

        self.arch = nn.Sequential(*layers)

    def forward(self, x):

        return self.arch(x)


class Downsample(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=2, padd=1):

        super().__init__()

        self.arch = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padd)

    def forward(self, x):

        return self.arch(x)


class E_block(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padd=1):

        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels,
                            kernel_size, stride, padd),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(),

                  nn.Conv2d(out_channels, out_channels,
                            kernel_size, stride, padd),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU()]

        self.block = nn.Sequential(*layers)

    def forward(self, x):

        return self.block(x)


class Encoder(nn.Module):

    def __init__(self, data_shape, out_channels=64,
                 min_channels=5, max_channels=1024):

        super().__init__()

        in_channels, pool_kernel_size, _ = data_shape

        # Calculate: Pooling Kernel Size & Number Encoder Blocks
        # - Pooling size starts as spatial shape and reduces
        # - Pooling is invoked as global average pooling

        num_blocks = 1
        while pool_kernel_size >= min_channels:
            pool_kernel_size = pool_kernel_size // 2
            num_blocks = num_blocks + 1
        num_blocks = num_blocks + 1

        # Construct: Network Encoder

        layers = []
        for i in range(num_blocks):

            # - Convolutional layer for the first encoder layer

            if i == 0:
                kernel_size, stride, padd = 7, 1, 3
                layers.append(nn.Conv2d(in_channels, out_channels,
                                        kernel_size, stride, padd))

            # - Average pooling for the final encoder layer

            elif i == (num_blocks - 1):
                layers.append(nn.AvgPool2d(pool_kernel_size))

            # - Otherwise, residual block with convolutional downsampling

            else:

                layers.append(Residual(E_block(in_channels=out_channels,
                                               out_channels=out_channels)))

                out_channels *= 2
                if out_channels >= max_channels:
                    out_channels = max_channels

                layers.append(Downsample(in_channels, out_channels))

            in_channels = out_channels

        self.arch = nn.Sequential(*layers)
        self.output_features = out_channels

    def forward(self, x):

        return self.arch(x)


if __name__ == "__main__":

    x = torch.rand(100, 3, 64, 64)

    encoder = Encoder((3, 64, 64))

    from IPython import embed
    embed()
    exit()

    decoder = Decoder((3, 64, 64), 1024)

    features = encoder(x)
    recons = decoder(features)

    print(features.shape, recons.shape)
