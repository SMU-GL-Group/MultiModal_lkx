# coding=utf-8
"""
使用Cross-modal Attention[1]，进行TEM与OM、TEM与IM的互加权，再将结果拼接在一起。

[1] Cross-modal attention for multi-modal image registration
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
from MyModel.Modules.CrossModalAttn import CrossModalAttn


basic_dim = 1024  # 特征提取器提取出的特征维度

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
        self.TEM_encoder = nn.Sequential(*list(swin_transformer.swin_b(weights=Swin_B_Weights).children())[:-4])
        self.OM_encoder = nn.Sequential(*list(swin_transformer.swin_b(Swin_B_Weights).children())[:-4])
        self.IM_encoder = nn.Sequential(*list(swin_transformer.swin_b(Swin_B_Weights).children())[:-4])

        # MLP
        self.MLP = MLP(num_i=basic_dim * 2,
                       num_h=128, num_o=args.num_cls)

        self.CrossAttn = CrossModalAttn(ch_in=basic_dim,ch_y=int(basic_dim//2))
        self.BN = nn.BatchNorm2d(num_features=basic_dim*2)
        self.conv1 = nn.Conv2d(in_channels=basic_dim*2, out_channels=basic_dim*2,kernel_size=(1,1))
        self.conv2 = nn.Conv2d(in_channels=basic_dim*2, out_channels=basic_dim*2,kernel_size=(1,1))

        # 各模态fc分支
        self.TEM_fc = FC(in_channels=basic_dim, out_channels=args.num_cls)
        self.OM_fc = FC(in_channels=basic_dim, out_channels=args.num_cls)
        self.IM_fc = FC(in_channels=basic_dim, out_channels=args.num_cls)

        self.t = transforms.ToPILImage()

    def forward(self, bag_tensor, OM_tensor, IM_tensor):
        batch_output, OM_output, IM_output, TEM_output = [],[],[],[]  # 一个batch的输出
        # 先对电镜图像进行特征提取
        bag_tuple = torch.split(bag_tensor, 1)  # 先按batch分开 [bs,图像数,C,H,W ]-> bs * [1,图像数,C,H,W]
        for bs in range(len(bag_tuple)):  # 第bs个batch
            for bag in bag_tuple[bs]:  # 第bag个包
                bag = bag[:6, :, :, :]  # 取若干张电镜，太多内存会爆
                TEMs = []  # 一个包内的图像特征
                img_tuple = torch.split(bag, 1)
                for i in img_tuple:
                    # # 查看每张图像
                    # img = self.t(torch.squeeze(i, dim=0))
                    # plt.imshow(img)
                    # plt.show()
                    if not torch.equal(i.cpu(), torch.zeros(i.shape)):  # 跳过空张量
                        feats = self.TEM_encoder(i)  # [1,7,7,1024]
                        TEMs.append(feats)
            x_TEM = (torch.stack(TEMs).sum(dim=0)).permute(0,3,1,2)  # 一个包内的特征求和

            TEM_pred = self.TEM_fc(x_TEM)
            TEM_output.append(TEM_pred)
            x_OM = self.OM_encoder(torch.unsqueeze(OM_tensor[bs], dim=0)).permute(0,3,1,2)
            x_IM = self.IM_encoder(torch.unsqueeze(IM_tensor[bs], dim=0)).permute(0,3,1,2)

            # ********CrossModalAttn融合,电镜分别和光镜、荧光融合******** #
            bs, h, w, C = x_TEM.shape

            TEM_p_OM = self.CrossAttn(c=x_OM,p=x_TEM)
            OM_p_TEM = self.CrossAttn(c=x_TEM,p=x_OM)
            TEM_p_OM = self.BN(TEM_p_OM)
            OM_p_TEM = self.BN(OM_p_TEM)
            TEM_p_OM = self.conv1(TEM_p_OM)
            OM_p_TEM = self.conv2(OM_p_TEM)
            TEM_OM = TEM_p_OM + OM_p_TEM

            TEM_p_IM = self.CrossAttn(c=x_IM, p=x_TEM)
            IM_p_TEM = self.CrossAttn(c=x_TEM, p=x_IM)
            TEM_p_IM = self.BN(TEM_p_IM)
            IM_p_TEM = self.BN(IM_p_TEM)
            TEM_p_IM = self.conv1(TEM_p_IM)
            IM_p_TEM = self.conv2(IM_p_TEM)
            TEM_IM = TEM_p_IM + IM_p_TEM

            OM_pred = self.OM_fc(x_OM.permute(0, 3, 1, 2))
            OM_output.append(OM_pred)

            IM_pred = self.IM_fc(x_IM.permute(0, 3, 1, 2))
            IM_output.append(IM_pred)
            # ********模态融合******** #
            x_cat = torch.cat((TEM_OM, TEM_IM), dim=1)  # [bs,2c,h,w]

            out = self.MLP(x_cat)
            batch_output.append(out)
        # [bs,1,num_cls]
        return torch.stack(batch_output, dim=0), torch.stack(OM_output, dim=0), torch.stack(IM_output, dim=0), torch.stack(TEM_output, dim=0)
