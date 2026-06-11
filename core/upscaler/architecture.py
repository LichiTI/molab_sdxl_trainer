try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None

def make_layer(basic_block, num_basic_block, **kwargs):
    if nn is None: 
        return None
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwargs))
    return nn.Sequential(*layers)

if nn is not None:
    Module = nn.Module
else:
    Module = object

class ResidualDenseBlock(Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        if nn is None: raise ImportError("PyTorch not found")
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x

class RRDB(Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, num_feat, num_grow_ch=32):
        if nn is None: raise ImportError("PyTorch not found")
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class RRDBNet(Module):
    def __init__(self, in_nc=3, out_nc=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4):
        if nn is None: raise ImportError("PyTorch not found")
        super(RRDBNet, self).__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(in_nc, num_feat, 3, 1, 1)
        self.RRDB_trunk = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.trunk_conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # Upsampling
        self.upconv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.HRconv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, out_nc, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        if self.scale == 4:
            fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
            
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        return out
