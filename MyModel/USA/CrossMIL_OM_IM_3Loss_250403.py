# coding=utf-8
"""
OM、IM简单特征融合，使用加权损失。
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
        self.n_modal = len(self.args.modal.split('+'))  # 模态数量
        self.OM_encoder = nn.Sequential(*list(swin_transformer.swin_b(weights=Swin_B_Weights).children())[:-4])
        self.IM_encoder = nn.Sequential(*list(swin_transformer.swin_b(weights=Swin_B_Weights).children())[:-4])

        basic_dim = 1024
        # MLP
        self.MLP = MLP(num_i=basic_dim * self.n_modal,
                       num_h=128, num_o=args.num_cls)

        # 各模态fc分支
        self.OM_fc = FC(in_channels=basic_dim, out_channels=args.num_cls)
        self.IM_fc = FC(in_channels=basic_dim, out_channels=args.num_cls)

        self.t = transforms.ToPILImage()

    def forward(self, bag_tensor, OM_tensor, IM_tensor):
        batch_output, OM_output, IM_output = [],[],[]  # 一个batch的输出
        # 先对电镜图像进行特征提取 [1,bs,图像数,C,H,W ]
        for bs in range(OM_tensor.shape[0]):  # 第bs个batch
            x_OM = self.OM_encoder(torch.unsqueeze(OM_tensor[bs], dim=0))
            x_IM = self.IM_encoder(torch.unsqueeze(IM_tensor[bs], dim=0))

            # ********CrossAttn融合******** #
            OM_pred = self.OM_fc(x_OM.permute(0, 3, 1, 2))
            OM_output.append(OM_pred)

            IM_pred = self.IM_fc(x_IM.permute(0, 3, 1, 2))
            IM_output.append(IM_pred)
            # ********模态融合******** #
            x_cat = torch.cat([x_OM, x_IM], dim=-1)
            out = self.MLP(x_cat.permute(0, 3, 1, 2))
            batch_output.append(out)
        # [bs,1,num_cls]
        return torch.stack(batch_output, dim=0), torch.stack(OM_output, dim=0), torch.stack(IM_output, dim=0)
