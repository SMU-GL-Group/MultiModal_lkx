#coding=utf-8
"""
各模态使用双向互交叉注意力机制 [1],电镜固定一张

[1] Multimodal attention-based deep learning for Alzheimer’s disease diagnosis
"""
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
# from mmcls.models import build_backbone # 支持不同输入尺寸
from torchvision.models import swin_transformer, Swin_B_Weights

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
        self.TEM_encoder = nn.Sequential(*list(swin_transformer.swin_b(weights=Swin_B_Weights).children())[:-4])
        self.OM_encoder = nn.Sequential(*list(swin_transformer.swin_b(weights=Swin_B_Weights).children())[:-4])
        self.IM_encoder = nn.Sequential(*list(swin_transformer.swin_b(weights=Swin_B_Weights).children())[:-4])

        basic_dim = 1024
        # MLP
        self.MLP = MLP(num_i=basic_dim * self.n_modal * 2,
                       num_h=128, num_o=args.num_cls)
        self.CrossModalAttn = nn.MultiheadAttention(embed_dim=basic_dim, num_heads=1, batch_first=True)  # [bs,h*w, C]

        # 各模态fc分支
        self.TEM_fc = FC(in_channels=basic_dim*2, out_channels=args.num_cls)
        self.OM_fc = FC(in_channels=basic_dim*2, out_channels=args.num_cls)
        self.IM_fc = FC(in_channels=basic_dim*2, out_channels=args.num_cls)

    def forward(self, TEM_tensor, OM_tensor, IM_tensor):
        x_TEM = self.TEM_encoder(TEM_tensor)  # [bs,h,w,C]
        x_OM = self.OM_encoder(OM_tensor)
        x_IM = self.IM_encoder(IM_tensor)
        #********各模态使用SelfAttn********#
        bs, h, w, C = x_TEM.shape
        x_TEM = x_TEM.view(bs,-1,C)
        x_OM = x_OM.view(bs,-1,C)
        x_IM = x_IM.view(bs,-1,C)
        # 第二个输出是注意力权重矩阵
        x_TEMc_OM,_ = self.CrossModalAttn(query=x_TEM, key=x_OM, value=x_OM)  # [bs,h*w,C]
        x_TEMc_IM,_ = self.CrossModalAttn(query=x_TEM, key=x_IM, value=x_IM)

        x_OMc_TEM,_ = self.CrossModalAttn(query=x_OM, key=x_TEM, value=x_TEM)
        x_OMc_IM,_ = self.CrossModalAttn(query=x_OM, key=x_IM, value=x_IM)

        x_IMc_TEM,_ = self.CrossModalAttn(query=x_IM, key=x_TEM, value=x_TEM)
        x_IMc_OM,_ = self.CrossModalAttn(query=x_IM, key=x_OM, value=x_OM)

        OM = torch.cat((x_TEMc_OM, x_IMc_OM), dim=-1).view(bs,h,w,-1) # [bs,h,w,2C]
        IM = torch.cat((x_TEMc_IM, x_OMc_IM), dim=-1).view(bs,h,w,-1)
        TEM = torch.cat((x_OMc_TEM, x_IMc_TEM), dim=-1).view(bs,h,w,-1)
        OM_pred = self.OM_fc(OM.permute(0,3,1,2))  # ->[bs,C,h,w]
        IM_pred = self.IM_fc(IM.permute(0,3,1,2))
        TEM_pred = self.TEM_fc(TEM.permute(0,3,1,2))
        #********直接拼接********#
        x = torch.cat([x_TEMc_OM, x_TEMc_IM, x_OMc_TEM, x_OMc_IM, x_IMc_TEM, x_IMc_OM], dim=-1).view(bs,h,w,-1)
        out = self.MLP(x.permute(0,3,1,2)) # [bs,C,H,W]

        return out, OM_pred, IM_pred, TEM_pred
