#coding=utf-8
"""
各模态使用自注意力机制,电镜固定一张
"""
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
# from mmcls.models import build_backbone # 支持不同输入尺寸
from torchvision.models import swin_transformer, Swin_B_Weights
from MyModel.Modules.MySelfAttn import SelfAttention

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
        self.args = args
        self.n_modal = len(self.args.modal.split('+')) # 模态数量
        self.TEM_encoder = nn.Sequential(*list(swin_transformer.swin_b().children())[:-4])
        self.OM_encoder = nn.Sequential(*list(swin_transformer.swin_b().children())[:-4])
        self.IM_encoder = nn.Sequential(*list(swin_transformer.swin_b().children())[:-4])

        # MLP
        self.MLP = MLP(num_i=1024 * self.n_modal,
                       num_h=128, num_o=args.num_cls)

        self.SelfAttn = SelfAttention(dim=1024, num_heads=4)

        # 各模态fc分支
        self.TEM_fc = FC(in_channels=1024, out_channels=args.num_cls)
        self.OM_fc = FC(in_channels=1024, out_channels=args.num_cls)
        self.IM_fc = FC(in_channels=1024, out_channels=args.num_cls)

        self.t = transforms.ToPILImage()

    def forward(self, TEM_tensor, OM_tensor, IM_tensor):
        x_TEM = self.TEM_encoder(TEM_tensor)
        x_OM = self.OM_encoder(OM_tensor)
        x_IM = self.IM_encoder(IM_tensor)
        #********各模态使用SelfAttn********#
        bs, h, w, C = x_TEM.shape
        attn_TEM = self.SelfAttn(input=x_TEM.view(bs,-1,C))
        x_TEM = x_TEM * attn_TEM.view(bs,h,w,C)
        TEM_pred = self.TEM_fc(x_TEM.permute(0, 3, 1, 2))

        attn_OM = self.SelfAttn(input=x_OM.view(bs,-1,C))
        x_OM = x_OM * attn_OM.view(bs,h,w,C)
        OM_pred = self.OM_fc(x_OM.permute(0, 3, 1, 2))

        attn_IM = self.SelfAttn(input=x_IM.view(bs,-1,C))
        x_IM = x_IM * attn_IM.view(bs,h,w,C)
        IM_pred = self.IM_fc(x_IM.permute(0, 3, 1, 2))
        #********直接拼接********#
        x = torch.cat([x_TEM, x_OM, x_IM], dim=-1)
        out = self.MLP(x.permute(0,3,1,2)) # [bs,C,H,W]

        return out, OM_pred, IM_pred, TEM_pred
