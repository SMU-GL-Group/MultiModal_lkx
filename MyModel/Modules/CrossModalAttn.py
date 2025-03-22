#coding=utf-8
"""
多模态多尺度方法：Cross-modal attention, Cross Attention的改进版，

[1] Cross-modal attention for multi-modal image registration
"""
import torch.nn as nn
import torch

class CrossModalAttn(nn.Module):
    def __init__(self,ch_in=32, ch_y=16):
        super(CrossModalAttn, self).__init__()
        self.q = nn.Conv2d(in_channels=ch_in,out_channels=ch_y,kernel_size=(1,1))
        self.k = nn.Conv2d(in_channels=ch_in,out_channels=ch_y,kernel_size=(1,1))
        self.v = nn.Conv2d(in_channels=ch_in,out_channels=ch_y,kernel_size=(1,1))
        self.y = nn.Conv2d(in_channels=ch_y,out_channels=ch_in,kernel_size=(1,1))

    def forward(self,c,p):
        q = self.q(c)
        k = self.k(p)
        v = self.v(p)
        y = q * k
        y = y.softmax(dim=-1)
        y = y * v
        y = self.y(y)
        z = torch.cat((y,c),dim=1)  # [bs,2*ch_in,h,w]

        return z


