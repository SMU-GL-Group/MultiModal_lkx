#coding=utf-8
"""
基于光镜、荧光、电镜的简单多模态融合。使用ViT-B编码器。
"""
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torchvision import transforms
from torchvision.models import ViT_B_32_Weights
from MyModel.torchvision_ViT import vit_b_32 # 用自己修改过分类头的ViT

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
        #x = self.avgpool(x)
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
        backbone = vit_b_32(weights=ViT_B_32_Weights)  # 虽然初始化中有heads，但forward中不参与（方便参数对应上）
        # backbone.heads = zero()
        self.TEM_encoder = backbone
        self.OM_encoder = backbone
        self.IM_encoder = backbone

        basic_dim = 768
        # MLP
        self.MLP = MLP(num_i=basic_dim * self.n_modal,
                       num_h=128, num_o=args.num_cls)

        #self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, TEM_tensor, OM_tensor, IM_tensor):
        x_TEM = self.TEM_encoder(TEM_tensor)  # [bs,50,basic_dim]
        x_OM = self.OM_encoder(OM_tensor)
        x_IM = self.IM_encoder(IM_tensor)

        # ********模态融合******** #
        x_cat = torch.cat([x_OM,x_IM, x_TEM], dim=-1)  # [bs,50,basic_dim*n]
        out = self.MLP(x_cat[:,0])  # [bs,basic_dim*n]

        return out # [bs,1,num_cls]
