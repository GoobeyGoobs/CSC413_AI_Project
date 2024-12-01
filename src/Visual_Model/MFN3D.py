from torch.nn import Linear, Conv3d, BatchNorm1d, BatchNorm3d, PReLU, Sequential, Module, Dropout
import torch
import torch.nn as nn


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv3d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding,
                           bias=False)
        self.bn = BatchNorm3d(out_c)
        self.prelu = PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv3d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding,
                           bias=False)
        self.bn = BatchNorm3d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Depth_Wise(Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1))
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


NON_LINEARITY = {
    'ReLU': nn.ReLU(inplace=True),
    'Swish': Swish(),
}


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class CoordAtt3D(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt3D, self).__init__()
        self.pool_d = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv3d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(mip)
        self.relu = h_swish()

        self.conv_d = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_h = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, d, h, w = x.size()
        x_d = self.pool_d(x)
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)

        x_d = x_d.permute(0, 1, 3, 4, 2)  # Move D to last dimension
        x_h = x_h.permute(0, 1, 2, 4, 3)  # Move H to last dimension

        y = torch.cat([x_d, x_h, x_w], dim=-1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)

        x_d, x_h, x_w = torch.split(y, [d, h, w], dim=-1)

        x_d = x_d.permute(0, 1, 4, 2, 3)
        x_h = x_h.permute(0, 1, 2, 4, 3)

        x_d = self.conv_d(x_d).sigmoid()
        x_h = self.conv_h(x_h).sigmoid()
        x_w = self.conv_w(x_w).sigmoid()

        x_d = x_d.expand(-1, -1, d, h, w)
        x_h = x_h.expand(-1, -1, d, h, w)
        x_w = x_w.expand(-1, -1, d, h, w)

        y = identity * x_d * x_h * x_w

        return y


class MDConv(Module):
    def __init__(self, channels, kernel_sizes, split_out_channels, stride):
        super(MDConv, self).__init__()
        self.num_groups = len(kernel_sizes)
        self.split_channels = split_out_channels
        self.mixed_depthwise_conv = nn.ModuleList()
        for i in range(self.num_groups):
            kernel_size = kernel_sizes[i]
            self.mixed_depthwise_conv.append(Conv3d(
                self.split_channels[i],
                self.split_channels[i],
                kernel_size=kernel_size,
                stride=stride,
                padding=tuple(k // 2 for k in kernel_size),
                groups=self.split_channels[i],
                bias=False
            ))
        self.bn = BatchNorm3d(channels)
        self.prelu = PReLU(channels)

    def forward(self, x):
        if self.num_groups == 1:
            x = self.mixed_depthwise_conv[0](x)
        else:
            x_split = torch.split(x, self.split_channels, dim=1)
            x = [conv(t) for conv, t in zip(self.mixed_depthwise_conv, x_split)]
            x = torch.cat(x, dim=1)

        x = self.bn(x)
        x = self.prelu(x)
        return x


class Mix_Depth_Wise(Module):
    def __init__(self, in_c, out_c, residual=False, stride=(2, 2, 2), groups=1,
                 kernel_sizes=[(3, 3, 3), (5, 5, 5), (7, 7, 7)], split_out_channels=[64, 32, 32]):
        super(Mix_Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1))
        self.conv_dw = MDConv(channels=groups, kernel_sizes=kernel_sizes, split_out_channels=split_out_channels,
                              stride=stride)
        self.CA = CoordAtt3D(groups, groups)
        self.project = Linear_block(groups, out_c, kernel=(1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1))
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.CA(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(Module):
    def __init__(self, c, num_block, groups, stride=(1, 1, 1), kernel=(3, 3, 3), padding=(1, 1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(
                Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class Mix_Residual(Module):
    def __init__(self, c, num_block, groups, stride=(1, 1, 1),
                 kernel_sizes=[(3, 3, 3), (5, 5, 5)], split_out_channels=[64, 64]):
        super(Mix_Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(
                Mix_Depth_Wise(c, c, residual=True, stride=stride, groups=groups,
                               kernel_sizes=kernel_sizes, split_out_channels=split_out_channels))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class MixedFeatureNet(Module):
    def __init__(self, embedding_size=256, out_d=2, out_h=7, out_w=7):
        super(MixedFeatureNet, self).__init__()
        # Input size: L x 112 x 112
        self.conv1 = Conv_block(1, 32, kernel=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        # L x 56 x 56
        self.conv2_dw = Conv_block(32, 32, kernel=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=32)
        self.conv_23 = Mix_Depth_Wise(32, 32, stride=(1, 2, 2), groups=64,
                                      kernel_sizes=[(3, 3, 3), (5, 5, 5), (7, 7, 7)],
                                      split_out_channels=[32, 16, 16])

        # L x 28 x 28
        self.conv_3 = Mix_Residual(32, num_block=3, groups=64, stride=(1, 1, 1),
                                   kernel_sizes=[(3, 3, 3), (5, 5, 5)],
                                   split_out_channels=[48, 16])
        self.conv_34 = Mix_Depth_Wise(32, 64, stride=(1, 2, 2), groups=128,
                                      kernel_sizes=[(3, 3, 3), (5, 5, 5), (7, 7, 7)],
                                      split_out_channels=[64, 32, 32])

        # L x 14 x 14
        self.conv_4 = Mix_Residual(64, num_block=4, groups=128, stride=(1, 1, 1),
                                   kernel_sizes=[(3, 3, 3), (5, 5, 5)],
                                   split_out_channels=[96, 32])
        self.conv_45 = Mix_Depth_Wise(64, 128, stride=(1, 2, 2), groups=256,
                                      kernel_sizes=[(3, 3, 3), (5, 5, 5), (7, 7, 7), (9, 9, 9)],
                                      split_out_channels=[64, 64, 64, 64])
        # L x 7 x 7
        self.conv_5 = Mix_Residual(128, num_block=2, groups=256, stride=(1, 1, 1),
                                   kernel_sizes=[(3, 3, 3), (5, 5, 5), (7, 7, 7)],
                                   split_out_channels=[86, 84, 86])
        self.conv_6_sep = Conv_block(128, 256, kernel=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv_6_dw = Linear_block(256, 256, groups=256, kernel=(out_d, out_h, out_w), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(256, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)
        self.dropout = Dropout(0.7)

    def forward(self, x):
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        out = self.linear(out)
        out = self.bn(out)

        return l2_norm(out)
