#coding=utf-8
"""
基于光镜、荧光、电镜的简单多模态融合。使用vgg16编码器。
"""
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torchvision import transforms
from torchvision.models import swin_transformer, Swin_B_Weights, vgg16, VGG16_Weights
from MyModel.Modules.MyCrossScaleAttn import CrossAttention

# ======================= 预测最终结果的MLP ======================= #
class MLP(nn.Module):
    def __init__(self, num_i, num_h, num_o):
        super(MLP, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear1 = torch.nn.Linear(num_i, num_h)  # in_feature, out_feature
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(num_h, num_o)  # 2个隐层
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.n_modal = len(self.args.modal.split('+')) # 模态数量
        self.TEM_encoder = nn.Sequential(*list(vgg16(weights=VGG16_Weights).children())[:-1], nn.AdaptiveAvgPool2d(1))
        self.OM_encoder = nn.Sequential(*list(vgg16(weights=VGG16_Weights).children())[:-1], nn.AdaptiveAvgPool2d(1))
        self.IM_encoder = nn.Sequential(*list(vgg16(weights=VGG16_Weights).children())[:-1], nn.AdaptiveAvgPool2d(1))

        basic_dim = 512
        # MLP
        self.MLP = MLP(num_i=basic_dim * self.n_modal,
                       num_h=128, num_o=args.num_cls)

        #self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, TEM_tensor, OM_tensor, IM_tensor):
        x_TEM = self.TEM_encoder(TEM_tensor)  # [bs,basic_dim,1,1]
        x_OM = self.OM_encoder(OM_tensor)
        x_IM = self.IM_encoder(IM_tensor)

        # ********模态融合******** #
        x_cat = torch.cat([x_OM,x_IM, x_TEM], dim=1)
        out = self.MLP(x_cat)

        return out # [bs,1,num_cls]
