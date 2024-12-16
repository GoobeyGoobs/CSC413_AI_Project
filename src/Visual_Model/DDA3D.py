from torch import nn
import torch  
from torch.nn import Module
from src.Visual_Model.MFN3D import MixedFeatureNet


class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = nn.Conv3d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


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


class CoordAtt3D(nn.Module):
    def __init__(self, inp, oup, groups=16):
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

        x_d = x_d.permute(0, 1, 3, 4, 2)  
        x_h = x_h.permute(0, 1, 2, 4, 3)  

        y = torch.cat([x_d, x_h, x_w], dim=4)  
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)

        x_d, x_h, x_w = torch.split(y, [d, h, w], dim=4)

        x_d = x_d.permute(0, 1, 4, 2, 3)  
        x_h = x_h.permute(0, 1, 2, 4, 3)  

        x_d = self.conv_d(x_d).sigmoid()
        x_h = self.conv_h(x_h).sigmoid()
        x_w = self.conv_w(x_w).sigmoid()

        x_d = x_d.expand(-1, -1, d, h, w)
        x_h = x_h.expand(-1, -1, d, h, w)
        x_w = x_w.expand(-1, -1, d, h, w)

        y = x_d * x_h * x_w

        return y


class CoordAttHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.CoordAtt = CoordAtt3D(128, 128)

    def forward(self, x):
        ca = self.CoordAtt(x)
        return ca


class DDAMNet(nn.Module):
    def __init__(self, num_class=8, num_head=2, dropout=0.0):
        super(DDAMNet, self).__init__()

        net = MixedFeatureNet() 

        self.features = nn.Sequential(*list(net.children())[:-4])  
        self.num_head = num_head
        for i in range(int(num_head)):
            setattr(self, "cat_head%d" % (i), CoordAttHead())

        
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = Flatten()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_class)
        self.bn = nn.BatchNorm1d(num_class)

    def forward(self, x):
        x = self.features(x)
        x = self.dropout(x)
        heads = []

        for i in range(self.num_head):
            heads.append(getattr(self, "cat_head%d" % i)(x))
        head_out = heads

        y = heads[0]

        for i in range(1, self.num_head):
            y = torch.max(y, heads[i])

        y = x * y
        y = self.pool(y)
        y = self.flatten(y)
        y = self.dropout(y)
        out = self.fc(y)
        return out, x, head_out
