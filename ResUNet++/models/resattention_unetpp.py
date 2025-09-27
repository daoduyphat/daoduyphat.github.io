import torch
import torch.nn as nn
import torch.nn.functional as F

# block residual
class ResidualConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        #cai nay de skip khong loi
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.skip is not None:
            x = self.skip(x)
        return F.relu(out + x)


# attention block, ep size
class AttentionBlock(nn.Module):
    def __init__(self, g_ch, x_ch):
        super().__init__()
        self.g_conv = nn.Conv2d(g_ch, x_ch, 1)
        self.x_conv = nn.Conv2d(x_ch, x_ch, 1)
        self.psi = nn.Conv2d(x_ch, 1, 1)

    def forward(self, g, x):
        # ep cho g khop voi x
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)

        g1 = self.g_conv(g)
        x1 = self.x_conv(x)
        out = F.relu(g1 + x1)
        out = torch.sigmoid(self.psi(out))
        return x * out


#model
class ResAttentionUNetPP(nn.Module):
    def __init__(self, num_classes=1, input_channels=3):
        super().__init__()
        f = [64, 128, 256, 512, 1024]

        # encoder
        self.conv0_0 = ResidualConv(input_channels, f[0])
        self.conv1_0 = ResidualConv(f[0], f[1])
        self.conv2_0 = ResidualConv(f[1], f[2])
        self.conv3_0 = ResidualConv(f[2], f[3])
        self.conv4_0 = ResidualConv(f[3], f[4])
        self.pool = nn.MaxPool2d(2)

        self.reduce1 = nn.Conv2d(f[1], f[0], 1)
        self.reduce2 = nn.Conv2d(f[2], f[1], 1)
        self.reduce3 = nn.Conv2d(f[3], f[2], 1)
        self.reduce4 = nn.Conv2d(f[4], f[3], 1)

        #decoder unet++
        self.conv0_1 = ResidualConv(f[0]*2, f[0])
        self.conv1_1 = ResidualConv(f[1]*2, f[1])
        self.conv2_1 = ResidualConv(f[2]*2, f[2])
        self.conv3_1 = ResidualConv(f[3]*2, f[3])

        self.conv0_2 = ResidualConv(f[0]*3, f[0])
        self.conv1_2 = ResidualConv(f[1]*3, f[1])
        self.conv2_2 = ResidualConv(f[2]*3, f[2])

        self.conv0_3 = ResidualConv(f[0]*4, f[0])
        self.conv1_3 = ResidualConv(f[1]*4, f[1])

        self.conv0_4 = ResidualConv(f[0]*5, f[0])

        #attention
        self.att3 = AttentionBlock(f[4], f[3])
        self.att2 = AttentionBlock(f[3], f[2])
        self.att1 = AttentionBlock(f[2], f[1])
        self.att0 = AttentionBlock(f[1], f[0])

        self.final = nn.Conv2d(f[0], num_classes, 1)

    def forward(self, x):
        #encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        #decoder unet++
        x0_1 = self.conv0_1(torch.cat([x0_0, F.interpolate(self.reduce1(x1_0), x0_0.shape[2:])], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, F.interpolate(self.reduce2(x2_0), x1_0.shape[2:])], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, F.interpolate(self.reduce3(x3_0), x2_0.shape[2:])], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, F.interpolate(self.reduce4(x4_0), x3_0.shape[2:])], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, F.interpolate(self.reduce1(x1_1), x0_0.shape[2:])], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, F.interpolate(self.reduce2(x2_1), x1_0.shape[2:])], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, F.interpolate(self.reduce3(x3_1), x2_0.shape[2:])], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, F.interpolate(self.reduce1(x1_2), x0_0.shape[2:])], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, F.interpolate(self.reduce2(x2_2), x1_0.shape[2:])], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, F.interpolate(self.reduce1(x1_3), x0_0.shape[2:])], 1))

        #attention
        g3 = self.att3(x4_0, x3_1)
        g2 = self.att2(g3, x2_2)
        g1 = self.att1(g2, x1_3)
        g0 = self.att0(g1, x0_4)

        out = self.final(g0)
        return out


if __name__ == "__main__":
    model = ResAttentionUNetPP(num_classes=1, input_channels=3)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print("output:", y.shape)
