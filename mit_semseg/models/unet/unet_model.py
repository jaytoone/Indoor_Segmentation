""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *
from torch.autograd import Variable

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        # self.outc = OutConv(1472, n_classes)

    def forward(self, x):

        segSize = x.size()[2:]

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # x_list = [x1, x2, x3, x4, x5]
        # concat_list = list()
        # for x_ in x_list:
        #   x_ = nn.functional.interpolate(
        #             x_, size=segSize, mode='bilinear', align_corners=False)
        #   # print('x_.shape :', x_.shape)
        #   concat_list.append(x_)
        
        ### with torch.no_grad(): # grad가 없다. -> backprop와 부합하지 않는다.
        #   concat_out = torch.cat(concat_list, 1)
        # # print('concat_out.shape :', concat_out.shape)

        #   x = nn.Conv2d(concat_out.shape[1], 2, kernel_size=1).cuda()(concat_out)  # 1472
        #   x = nn.functional.log_softmax(x, dim=1)
        # # x = self.outc(concat_out)
        # del concat_out
        # torch.cuda.empty_cache()
        # return x

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
