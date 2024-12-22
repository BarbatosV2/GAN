import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
'''
# Define the Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = 64 // 4
        self.l1 = nn.Sequential(nn.Linear(100, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),  # Transposed convolution
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [spectral_norm(nn.Conv2d(in_filters, out_filters, 3, 2, 1)), 
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            return block

        self.model = nn.Sequential(
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        self.skip_connection = spectral_norm(nn.Conv2d(3, 128, 1, stride=16))  # Downsample input image for skip connection

        ds_size = 64 // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        skip = self.skip_connection(img).view(img.shape[0], -1)  # Skip connection
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out + skip)  # Combine skip connection and feature map
        return validity

'''

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        return x + self.block(x)
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = 64 // 4
        self.l1 = nn.Sequential(nn.Linear(100, 128 * self.init_size ** 2))

        self.res_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        out = self.res_blocks(out)
        img = self.conv_blocks(out)
        return img

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [spectral_norm(nn.Conv2d(in_filters, out_filters, 3, 2, 1)), 
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            return block

        self.model = nn.Sequential(
            *discriminator_block(3, 16, bn=False),
            ResidualBlock(16),
            *discriminator_block(16, 32),
            ResidualBlock(32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        self.skip_connection = spectral_norm(nn.Conv2d(3, 128, 1, stride=16))  # Downsample input image for skip connection

        ds_size = 64 // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        skip = self.skip_connection(img).view(img.shape[0], -1)  # Skip connection
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out + skip)  # Combine skip connection and feature map
        return validity
