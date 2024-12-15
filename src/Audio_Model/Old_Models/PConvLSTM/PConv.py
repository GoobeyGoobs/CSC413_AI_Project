import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SconvBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.LeakyReLU(negative_slope=0.02)
        self.conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        x = self.bn(x)
        x = self.act(x)
        x = self.conv(x)
        return x

class BconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BconvBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.LeakyReLU(negative_slope=0.02)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        x = self.bn(x)
        x = self.act(x)
        x = self.conv(x)
        return x

class Stem(nn.Module):
    # As previously designed
    def __init__(self, in_channels=1):
        super(Stem, self).__init__()
        self.sconv1 = SconvBlock(in_channels, 32, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.sconv2 = SconvBlock(32, 16, stride=2)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=3, padding=1, bias=False)

    def forward(self, x):
        # x: (batch, 1, time, 48)
        x = self.sconv1(x)  # -> (batch, 32, time, 48) reduced by next ops
        x = self.pool1(x)   # stride=2 in time/height dimension
        x = self.sconv2(x)  # additional reduction
        x = self.conv3(x)
        return x


class PconvBlock(nn.Module):
    def __init__(self, in_channels):
        super(PconvBlock, self).__init__()
        self.sconv = SconvBlock(in_channels, 64, stride=2)
        self.bconv = BconvBlock(in_channels, 32, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # After concat: channels = 64 + 32 + in_channels
        total_out = 64 + 32 + in_channels

        # Projection if channel mismatch
        self.channel_proj = None
        if total_out != in_channels:
            self.channel_proj = nn.Conv2d(total_out, in_channels, kernel_size=1, bias=False)

        # Downsample to match spatial dimensions if needed
        # Since we are using stride=2 in the branches, we must also downsample x.
        self.spatial_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=2, bias=False)

    def forward(self, x):
        out_s = self.sconv(x)  # (batch,64,H/2,W/2)
        out_b = self.bconv(x)  # (batch,32,H/2,W/2)
        out_p = self.pool(x)  # (batch,in_channels,H/2,W/2)

        concat_out = torch.cat([out_s, out_b, out_p], dim=1)  # (batch, 64+32+in_channels, H/2, W/2)

        if self.channel_proj is not None:
            concat_out = self.channel_proj(concat_out)  # (batch,in_channels,H/2,W/2)

        # Downsample x to match spatial dims
        x = self.spatial_proj(x)  # (batch,in_channels,H/2,W/2)

        out = x + concat_out
        return out


# class PconvBlock(nn.Module):
#     def __init__(self, in_channels):
#         super(PconvBlock, self).__init__()
#         self.sconv = SconvBlock(in_channels, 64, stride=2)
#         self.bconv = BconvBlock(in_channels, 32, stride=2)
#         self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         total_out = 64 + 32 + in_channels
#         if total_out != in_channels:
#             self.proj = nn.Conv2d(total_out, in_channels, kernel_size=1, bias=False)
#         else:
#             self.proj = None
#
#     def forward(self, x):
#         out_s = self.sconv(x)
#         out_b = self.bconv(x)
#         out_p = self.pool(x)
#         concat_out = torch.cat([out_s, out_b, out_p], dim=1)
#
#         if self.proj is not None:
#             concat_out = self.proj(concat_out)
#
#         out = x + concat_out
#         return out

class StemPlusPConvNet(nn.Module):
    def __init__(self, in_channels=1):
        super(StemPlusPConvNet, self).__init__()
        self.stem = Stem(in_channels=in_channels)
        self.pconv = PconvBlock(in_channels=16)

    def forward(self, x):
        # x: (batch, 1, time, 48)
        x = self.stem(x)   # -> (batch,16,H',W')
        x = self.pconv(x)  # -> (batch,16,H'',W'')
        return x
