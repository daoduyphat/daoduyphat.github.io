import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """2 convolution liem tiep: conv2d =>batchnorm2d => relu => conv2d => batchnorm2d -> relu"""
    #giup trich xuat dac trung hieu qua hon
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x): return self.conv(x)
#unet++
class NestedUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        #so luong filter cho tung tang
        nb_filter=[64,128,256,512,1024]
        #encoder
        self.conv0_0=DoubleConv(in_ch,nb_filter[0])
        self.conv1_0=DoubleConv(nb_filter[0],nb_filter[1])
        self.conv2_0=DoubleConv(nb_filter[1],nb_filter[2])
        self.conv3_0=DoubleConv(nb_filter[2],nb_filter[3])
        self.conv4_0=DoubleConv(nb_filter[3],nb_filter[4])

        self.up1_0=nn.ConvTranspose2d(nb_filter[1],nb_filter[0],2,2)
        self.up2_0=nn.ConvTranspose2d(nb_filter[2],nb_filter[1],2,2)
        self.up3_0=nn.ConvTranspose2d(nb_filter[3],nb_filter[2],2,2)
        self.up4_0=nn.ConvTranspose2d(nb_filter[4],nb_filter[3],2,2)

        self.conv0_1=DoubleConv(nb_filter[0]*2,nb_filter[0])
        self.conv1_1=DoubleConv(nb_filter[1]*2,nb_filter[1])
        self.conv2_1=DoubleConv(nb_filter[2]*2,nb_filter[2])
        self.conv3_1=DoubleConv(nb_filter[3]*2,nb_filter[3])

        self.conv0_2=DoubleConv(nb_filter[0]*3,nb_filter[0])
        self.conv1_2=DoubleConv(nb_filter[1]*3,nb_filter[1])
        self.conv2_2=DoubleConv(nb_filter[2]*3,nb_filter[2])

        self.conv0_3=DoubleConv(nb_filter[0]*4,nb_filter[0])
        self.conv1_3=DoubleConv(nb_filter[1]*4,nb_filter[1])

        self.conv0_4=DoubleConv(nb_filter[0]*5,nb_filter[0])
        #output
        self.final=nn.Conv2d(nb_filter[0],out_ch,1)
        #maxpooling dung trong encoder
        self.pool=nn.MaxPool2d(2,2)

    def forward(self,x):
        x0_0=self.conv0_0(x)
        x1_0=self.conv1_0(self.pool(x0_0))
        x0_1=self.conv0_1(torch.cat([x0_0,self.up1_0(x1_0)],1))
        x2_0=self.conv2_0(self.pool(x1_0))
        x1_1=self.conv1_1(torch.cat([x1_0,self.up2_0(x2_0)],1))
        x0_2=self.conv0_2(torch.cat([x0_0,x0_1,self.up1_0(x1_1)],1))
        x3_0=self.conv3_0(self.pool(x2_0))
        x2_1=self.conv2_1(torch.cat([x2_0,self.up3_0(x3_0)],1))
        x1_2=self.conv1_2(torch.cat([x1_0,x1_1,self.up2_0(x2_1)],1))
        x0_3=self.conv0_3(torch.cat([x0_0,x0_1,x0_2,self.up1_0(x1_2)],1))
        x4_0=self.conv4_0(self.pool(x3_0))
        x3_1=self.conv3_1(torch.cat([x3_0,self.up4_0(x4_0)],1))
        x2_2=self.conv2_2(torch.cat([x2_0,x2_1,self.up3_0(x3_1)],1))
        x1_3=self.conv1_3(torch.cat([x1_0,x1_1,x1_2,self.up2_0(x2_2)],1))
        x0_4=self.conv0_4(torch.cat([x0_0,x0_1,x0_2,x0_3,self.up1_0(x1_3)],1))
        #dung sigmoid de output trong [0,1]
        return torch.sigmoid(self.final(x0_4))
