# """ Parts of the U-Net model """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
#             nn.BatchNorm2d(mid_channels,track_running_stats=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True),
#             nn.BatchNorm2d(out_channels,track_running_stats=False),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)


# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.avgpool_conv = nn.Sequential(
#             nn.AvgPool2d(2),
#             nn.Dropout(p=0.1),
#             DoubleConv(in_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.avgpool_conv(x)


# class Up(nn.Module):
#     """Upscaling then double conv"""

#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()
#         # if bilinear, use the normal convolutions to reduce the number of channels
#         self.bilinear = bilinear
#         if bilinear:
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels)

#     def forward(self, x1, x2):

#         if self.bilinear:
#             x1 = F.interpolate(x1, size=x2.size()[2:], mode='bicubic', align_corners=True)
#         else:
#             x1 = self.up(x1)
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)


# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.outconv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1,bias=True),
#         )

#     def forward(self, x):
#         return self.outconv(x)

#unet_parts.py V2

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    (Conv2d ¡ú BatchNorm2d ¡ú ReLU ¡ú Dropout2d) ¡Á 2
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, drop_prob=0.2):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # first conv block
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop_prob),

            # second conv block
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop_prob),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with AvgPool ¡ú Dropout ¡ú DoubleConv"""

    def __init__(self, in_channels, out_channels, drop_prob=0.1):
        super().__init__()
        self.avgpool_conv = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Dropout2d(p=drop_prob),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.avgpool_conv(x)


class Up(nn.Module):
    """Upscaling then DoubleConv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear

        # if using bilinear upsampling, reduce channel count first
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                           kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1: lower-resolution feature map, x2: skip connection
        x1 = self.up(x1)
        # pad if needed (not usually necessary with power-of-two spatial dims)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffY or diffX:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX//2,
                            diffY // 2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1¡Á1 convolution to map to desired number of output channels"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        return self.conv(x)
