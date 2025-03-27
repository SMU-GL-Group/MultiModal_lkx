# coding=utf-8
"""
基于光镜、荧光、电镜的简单多模态融合。使用CrossFormer编码器。
"""
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
# from mmcls.models import build_backbone # 支持不同输入尺寸
from torchvision.models import swin_transformer, Swin_B_Weights
from MyModel.Modules.MyCrossScaleAttn import CrossAttention
from MyModel.CrossFormer.build import build_model


# ======================= 预测最终结果的MLP ======================= #
class MLP(nn.Module):
    def __init__(self, num_i, num_h, num_o):
        super(MLP, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = torch.nn.Linear(num_i, num_h)  # in_feature, out_feature
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(num_h, num_o)  # 2个隐层
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        #x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x

# ======================= 输出每个分支预测结果的fc ======================= #
class FC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FC, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.relu(x)

        return x


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        basic_dim = 768
        self.n_modal = len(args.DATA.MODAL.split('+'))  # 模态数量
        encoder = build_model(config=args)
        encoder.state_dict(torch.load('/public/longkaixing/CrossModalScale/MyModel/CrossFormer/crossformer-b.pth'))
        encoder.head = nn.Identity()
        self.TEM_encoder = encoder
        self.OM_encoder = encoder
        self.IM_encoder = encoder

        # MLP
        self.MLP = MLP(num_i=basic_dim * self.n_modal,
                       num_h=128, num_o=args.DATA.NUM_CLS)

        self.t = transforms.ToPILImage()

    def forward(self, TEM_tensor, OM_tensor, IM_tensor):
        x_TEM = self.TEM_encoder(TEM_tensor)  # [bs,7,7,768]

        x_OM = self.OM_encoder(OM_tensor)
        x_IM = self.IM_encoder(IM_tensor)
        # ********模态融合******** #
        x_cat = torch.cat([x_TEM, x_OM, x_IM],dim=-1)
        out = self.MLP(x_cat)
        # [bs,1,num_cls]
        return out
