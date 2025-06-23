# import torch
# import torch.nn as nn
# from unet_parts import DoubleConv, Down, Up, OutConv
# import torch.nn.functional as F

# class UNet(nn.Module):
#     # def __init__(self, in_channels: int = 1, base_channels: int = 64, bilinear: bool = True):
#     #     super(UNet, self).__init__()
#     def __init__(self, in_channels, base_channels=64, bilinear=True, H=64, W=64):
#         super().__init__()
#         self.bilinear = bilinear

#         self.inc   = DoubleConv(in_channels, base_channels)
#         self.down1 = Down(base_channels, base_channels * 2)
#         self.down2 = Down(base_channels * 2, base_channels * 4)
#         self.down3 = Down(base_channels * 4, base_channels * 8)
#         factor     = 2 if bilinear else 1
#         self.down4 = Down(base_channels * 8, (base_channels * 16) // factor)
#         self.up1   = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
#         self.up2   = Up(base_channels * 8,  base_channels * 4 // factor, bilinear)
#         self.up3   = Up(base_channels * 4,  base_channels * 2 // factor, bilinear)
#         self.up4   = Up(base_channels * 2,  base_channels,            bilinear)
#         self.outc = nn.Conv2d(base_channels, H*W, kernel_size=1)
#         self.H, self.W = H, W


#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x  = self.up1(x5, x4)
#         x  = self.up2(x,  x3)
#         x  = self.up3(x,  x2)
#         x  = self.up4(x,  x1)
#         # logits = self.outc(x)
#         # return logits
#         x = self.outc(x)                        # (batch, H*W, H, W)
#         x = F.adaptive_avg_pool2d(x, (1,1))     # (batch, H*W, 1, 1)
#         return x.view(x.size(0), -1)           # (batch, H*W)

#unet_model.py V2
import torch
import torch.nn as nn
import torch.nn.functional as F
from unet_parts import DoubleConv, Down, Up

class UNet(nn.Module):
    """
    A U-Net that predicts range-bin logits and velocity-bin logits separately.
    Input:  (batch, in_channels, H, W)
    Output: (rlogits, vlogits) of shapes (batch, H) and (batch, W)
    """
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 128,  # wider network
        bilinear: bool = True,
        H: int = 64,
        W: int = 64
    ):
        super().__init__()
        self.bilinear = bilinear
        self.H, self.W = H, W

        # Encoder
        self.inc   = DoubleConv(in_channels,       base_channels)
        self.down1 = Down(base_channels,           base_channels * 2)
        self.down2 = Down(base_channels * 2,       base_channels * 4)
        self.down3 = Down(base_channels * 4,       base_channels * 8)
        factor     = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8,
                          (base_channels * 16) // factor)

        # Decoder
        self.up1   = Up(base_channels * 16,        base_channels * 8  // factor, bilinear)
        self.up2   = Up(base_channels * 8,         base_channels * 4  // factor, bilinear)
        self.up3   = Up(base_channels * 4,         base_channels * 2  // factor, bilinear)
        self.up4   = Up(base_channels * 2,         base_channels            , bilinear)

        # Two separate 1¡Á1 conv heads
        self.out_r = nn.Conv2d(base_channels, H, kernel_size=1)
        self.out_v = nn.Conv2d(base_channels, W, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        feat = self.up4(x,  x1)

        # Range head: output shape (B, H, H, W)
        rmap    = self.out_r(feat)
        # average over velocity dim (W), then spatial range dim (H) ¡ú (B, H)
        rlogits = rmap.mean(dim=3).mean(dim=2)

        # Velocity head: output shape (B, W, H, W)
        vmap    = self.out_v(feat)
        # average over range dim (H), then spatial velocity dim (W) ¡ú (B, W)
        vlogits = vmap.mean(dim=2).mean(dim=2)

        return rlogits, vlogits
