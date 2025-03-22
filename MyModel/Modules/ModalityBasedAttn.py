#coding=utf-8
"""
Modality-based Attention[1]的复现。

[1] Multiple instance convolutional neural network with modality-based attention and contextual multi-instance learning
pooling layer for effective differentiation between borderline and malignant epithelial ovarian tumors
"""
import torch.nn as nn
import torch


class ModalityBasedAttn(nn.Module):
    def __init__(self, channel, reduction=2):
        """
        :param channel: 输入的向量的通道维度
        :param reduction: 通道维度减小的倍数
        """
        super(ModalityBasedAttn, self).__init__()
        self.GP = nn.AdaptiveAvgPool2d(1)
        self.FC1 = nn.Linear(channel, channel//reduction, bias=False)
        self.FC2 = nn.Linear(channel//reduction, channel, bias=False)
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):  # x=[bs,C=c*self.n_modal, h, w]
        bs, C, h, w = x.size()
        y = self.GP(x).view(bs,C)  # [bs, c*n_modal]
        y = self.FC1(y)  # [bs, c*n_modal//reduction]
        y = self.ReLU(y)
        y = self.FC2(y)
        y_out = self.Sigmoid(y).view(bs,C,1,1)  # [bs, c*n_modal,1,1]

        out = x * y_out.expand_as(x)  # [bs,c*n_modal, h, w] 通道级特征响应的重新校准

        return out
