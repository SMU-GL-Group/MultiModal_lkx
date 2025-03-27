#coding=utf-8
"""
基于光镜、荧光、电镜的简单多模态融合。使用InceptionV3编码器，不使用辅助分类器aux_logits=False。
"""
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torchvision import transforms
from torchvision.models import Inception3, Inception_V3_Weights
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
        #x = self.avgpool(x)
        #x = torch.flatten(x, 1)
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
        backbone = Inception3(init_weights=Inception_V3_Weights, aux_logits=False)
        # 移除最后的池化层和全连接层，因为我们只需要特征提取
        backbone.avgpool = nn.Identity()
        backbone.fc = nn.Dropout(p=0.3)
        self.TEM_encoder = backbone
        self.OM_encoder = backbone
        self.IM_encoder = backbone

        self.basic_dim = 2048
        # 特征降维
        self.TEM_conv = nn.Conv2d(self.basic_dim, 1024, kernel_size=(1,1))
        self.OM_conv = nn.Conv2d(self.basic_dim, 1024, kernel_size=(1,1))
        self.IM_conv = nn.Conv2d(self.basic_dim, 1024, kernel_size=(1,1))
        # MLP
        self.MLP = MLP(num_i=1024 * self.n_modal,
                       num_h=256, num_o=args.num_cls)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, TEM_tensor, OM_tensor, IM_tensor):
        bs, c, h, w = TEM_tensor.size()
        x_TEM = self.TEM_encoder(TEM_tensor).view(bs, self.basic_dim, 8, 8) # [bs,basic_dim,8,8], 推理时没有logits的结果
        x_TEM = self.TEM_conv(x_TEM)  # [bs,1024,8,8]
        x_TEM = torch.flatten(self.avgpool(x_TEM),1) # [bs,1024,1,1]

        x_OM = self.OM_encoder(OM_tensor).view(bs, self.basic_dim, 8, 8)
        x_OM = self.OM_conv(x_OM)
        x_OM = torch.flatten(self.avgpool(x_OM), 1)

        x_IM = self.IM_encoder(IM_tensor).view(bs, self.basic_dim, 8, 8)
        x_IM = self.IM_conv(x_IM)
        x_IM = torch.flatten(self.avgpool(x_IM), 1)

        # ********模态融合******** #
        x_cat = torch.cat([x_OM,x_IM, x_TEM], dim=1)
        out = self.MLP(x_cat)

        return out # [bs,1,num_cls]
