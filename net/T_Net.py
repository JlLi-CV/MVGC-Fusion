import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from glob import glob
from torchvision.utils import save_image
from scipy.io import savemat

def sobel_kernel(channels):
    # 返回 shape [out_c, in_c, 3, 3] 的 Sobel kernel
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32) / 8.0
    sobel_y = sobel_x.t()
    kernel_x = sobel_x.expand(channels, channels, 3, 3).clone()
    kernel_y = sobel_y.expand(channels, channels, 3, 3).clone()
    # kernel_x = sobel_x.view(1, 1, 3, 3).expand(-1, channels, -1, -1).clone()
    # kernel_y = sobel_y.view(1, 1, 3, 3).expand(-1, channels, -1, -1).clone()
    return kernel_x, kernel_y

class SobelGrad(nn.Module):
    def __init__(self, channels):
        super().__init__()
        kx, ky = sobel_kernel(channels)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)
        self.pad = nn.ReflectionPad2d(1)

    def forward(self, x):
        x = self.pad(x)
        gx = nn.functional.conv2d(x, self.kx, padding=0)
        gy = nn.functional.conv2d(x, self.ky, padding=0)
        return gx, gy

class DWConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # self.bn = nn.BatchNorm2d(out_channels)
        # self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        # x = self.bn(x)
        # return self.act(x)
        return x

class TransferModel(nn.Module):
    def __init__(self , channels , msi_channels):
        super().__init__()
        # self.dim = 32
        # self.reduce_dim = nn.Conv2d(channels, 32, kernel_size=1)
        # self.bn_reduce = nn.BatchNorm2d(32)


        # self.layer1 = nn.Conv2d(32, 64, kernel_size=3, padding=1, groups=4)
        # self.bn1 = nn.BatchNorm2d(64)
        #
        # self.layer2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=4)
        # self.bn2 = nn.BatchNorm2d(64)
        #
        # self.layer3 = nn.Conv2d(64, 32, kernel_size=3, padding=1, groups=4)
        # self.bn3 = nn.BatchNorm2d(32)
        #
        #
        # self.layer4 = nn.Conv2d(32 + 32, 32, kernel_size=3, padding=1)
        # self.bn4 = nn.BatchNorm2d(32)
        #
        #
        # self.layer5 = nn.Conv2d(channels + 32, 16, kernel_size=3, padding=1)
        # self.bn5 = nn.BatchNorm2d(16)
        #
        # self.layer6 = nn.Conv2d(16, msi_channels, kernel_size=3, padding=1)


        ##### Pavia 数据集
        self.dim = 64
        self.reduce_dim = nn.Conv2d(channels, self.dim, kernel_size=1)
        self.bn_reduce = nn.BatchNorm2d(self.dim)


        self.layer1 = DWConvBlock(self.dim, self.dim)
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.layer2 = DWConvBlock(self.dim, self.dim)
        self.bn2 = nn.BatchNorm2d(self.dim)
        self.layer3 = DWConvBlock(self.dim, self.dim)
        self.bn3 = nn.BatchNorm2d(self.dim)

        self.layer4 = nn.Conv2d(self.dim * 2, self.dim , kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(self.dim)

        self.layer5 = nn.Conv2d(channels + self.dim, 32, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        self.layer6 = nn.Conv2d(32, msi_channels, kernel_size=3, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):

        x_origin = x

        
        x_reduce = nn.functional.leaky_relu(self.bn_reduce(self.reduce_dim(x)), 0.2)  # [B, 32, H, W]


        conv1 = nn.functional.leaky_relu(self.bn1(self.layer1(x_reduce)), 0.2)  # [B, 64, H, W]
        conv2 = self.bn2(self.layer2(conv1)) + conv1  
        conv2 = nn.functional.leaky_relu(conv2, 0.2)  # [B, 64, H, W]
        conv3 = nn.functional.leaky_relu(self.bn3(self.layer3(conv2)), 0.2)  # [B, 32, H, W]

        concat1 = torch.cat([x_reduce, conv3], dim=1)  # [B, 64, H, W]
        conv4 = nn.functional.leaky_relu(self.bn4(self.layer4(concat1)), 0.2)  # [B, 32, H, W]

        
        concat2 = torch.cat([x_origin, conv4], dim=1)  # [B, 160, H, W]
        conv5 = nn.functional.leaky_relu(self.bn5(self.layer5(concat2)), 0.2)  # [B, 16, H, W]

        conv6 = torch.tanh(self.layer6(conv5)) * 2.0
        return conv6



class TModel(nn.Module):
    def __init__(self, hsi_channels , msi_channels):
        super().__init__()
        self.transfer = TransferModel(hsi_channels , msi_channels)
        self.sobel_hsi = SobelGrad(hsi_channels)
        self.sobel_msi = SobelGrad(msi_channels)
        self.msi_grad_reduce = nn.Conv2d(msi_channels, 1, kernel_size=1)

    # def forward(self, hsi , msi):
    def forward(self, hsi , msi):

        _, C, H, W = hsi.shape

        half_h = H // 2
        half_w = W // 2

        # split_hsi:
        
        LT_hsi = hsi[:, :, :half_h, :half_w]
        RT_hsi = hsi[:, :, :half_h, half_w:]
        LB_hsi = hsi[:, :, half_h:, :half_w]
        RB_hsi = hsi[:, :, half_h:, half_w:]

       

        # split_msi:
        
        LT_msi = msi[:, :, :half_h, :half_w]
        RT_msi = msi[:, :, :half_h, half_w:]
        LB_msi = msi[:, :, half_h:, :half_w]
        RB_msi = msi[:, :, half_h:, half_w:]

    


        gx_hsi, gy_hsi = self.sobel_hsi(hsi)
        gx_msi, gy_msi = self.sobel_msi(msi)

     

        ##### MSI sub_view
        gx_LT_msi , gy_LT_msi = self.sobel_msi(LT_msi)
        gx_RT_msi , gy_RT_msi = self.sobel_msi(RT_msi)
        gx_LB_msi , gy_LB_msi = self.sobel_msi(LB_msi)
        gx_RB_msi , gy_RB_msi = self.sobel_msi(RB_msi)

        ##### HSI sub_view
        gx_LT_hsi , gy_LT_hsi= self.sobel_hsi(LT_hsi)
        gx_RT_hsi , gy_RT_hsi= self.sobel_hsi(RT_hsi)
        gx_LB_hsi , gy_LB_hsi= self.sobel_hsi(LB_hsi)
        gx_RB_hsi , gy_RB_hsi= self.sobel_hsi(RB_hsi)

        gx_pred = self.transfer(gx_hsi)
        gy_pred = self.transfer(gy_hsi)

     
        

        ##### x-direction
        gx_LT_pred = self.transfer(gx_LT_hsi)
        gx_RT_pred = self.transfer(gx_RT_hsi)
        gx_LB_pred = self.transfer(gx_LB_hsi)
        gx_RB_pred = self.transfer(gx_RB_hsi)

        ##### y-direction
        gy_LT_pred = self.transfer(gy_LT_hsi)
        gy_RT_pred = self.transfer(gy_RT_hsi)
        gy_LB_pred = self.transfer(gy_LB_hsi)
        gy_RB_pred = self.transfer(gy_RB_hsi)

        return [gx_pred, gx_msi, gy_pred, gy_msi] , \
            [gx_LT_pred ,gx_LT_msi, gx_RT_pred ,gx_RT_msi , gx_LB_pred , gx_LB_msi, gx_RB_pred , gx_RB_msi ], \
            [gy_LT_pred, gy_LT_msi , gy_RT_pred, gy_RT_msi , gy_LB_pred, gy_LB_msi ,gy_RB_pred, gy_RB_msi]




if __name__ == '__main__':
    # input_1 = torch.randn((2 , 31 , 256 , 256))
    # print('size: ' , input_1.shape)
    # t_net = TransferModel(31)
    # output = t_net(input_1)
    # print(output.shape)

    hsi = torch.randn((4 , 31 , 128 , 128))
    msi = torch.randn((4 , 3 , 128 , 128))
    t_net = TModel(31 , 3)
    out = t_net(hsi , msi)
    print(out[1][0].shape , out[1][1].shape)