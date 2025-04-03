# coding=utf-8
"""
电镜作为q,加权光镜kv、荧光kv。注意力矩阵相加而非相乘，电镜多实例加上KNN聚类特征。模态拼接，损失函数加权。
（损失下降过慢，40轮大概降至50~60左右，太大了。）
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
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


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
        self.OM_encoder = nn.Sequential(*list(swin_transformer.swin_b(weights=Swin_B_Weights).children())[:-4])
        self.IM_encoder = nn.Sequential(*list(swin_transformer.swin_b(weights=Swin_B_Weights).children())[:-4])

        # MLP
        self.MLP = MLP(num_i=1024 * self.n_modal,
                       num_h=128, num_o=args.num_cls)

        self.CrossAttn = CrossAttention(dim=1024, num_heads=4)

        # 各模态fc分支
        self.TEM_fc = FC(in_channels=1024, out_channels=args.num_cls)
        self.OM_fc = FC(in_channels=1024, out_channels=args.num_cls)
        self.IM_fc = FC(in_channels=1024, out_channels=args.num_cls)

        self.t = transforms.ToPILImage()

        self.kmeans = KMeans(n_clusters=4, random_state=0)  # 沿用国钰师兄的KNN设计，4个类别。

    def forward(self, bag_tensor, OM_tensor, IM_tensor):
        batch_output, OM_output, IM_output, TEM_output = [],[],[],[]  # 一个batch的输出
        # 先对电镜图像进行特征提取 [1,bs,图像数,C,H,W ]
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

                        # -------- 利用KNN对实例特征进行重加权,对图像内的结构特征重新加权 --------
                        feats_flatten = feats.view(-1, feats.shape[-1]).cpu().detach().numpy()
                        self.kmeans.fit(feats_flatten)
                        cluster_labels = self.kmeans.labels_

                        # 计算每个特征向量到其聚类中心的距离
                        cluster_centers = self.kmeans.cluster_centers_
                        distances = euclidean_distances(feats_flatten, cluster_centers[cluster_labels])

                        # 找到每个特征向量到其最近聚类中心的距离
                        min_distances = np.min(distances, axis=1, keepdims=True)

                        # 计算权重（距离的倒数）
                        weights = 1 / (min_distances + 1e-6)  # 加上一个小的常数以避免除以零

                        # 将权重应用到原始特征上
                        weighted_feats = feats_flatten * weights
                        # 将加权特征重新塑形为原始特征图的形状
                        weighted_feats = torch.tensor(weighted_feats.reshape(feats.shape), dtype=feats.dtype,
                                                      device=feats.device)

                        TEMs.append(weighted_feats)

            x_TEM = torch.stack(TEMs).sum(dim=0)  # 一个包内的特征求和

            TEM_pred = self.TEM_fc(x_TEM.permute(0, 3, 1, 2))
            TEM_output.append(TEM_pred)
            x_Modals = [x_TEM]
            x_OM = self.OM_encoder(torch.unsqueeze(OM_tensor[bs], dim=0))
            x_IM = self.IM_encoder(torch.unsqueeze(IM_tensor[bs], dim=0))

            # ********CrossAttn融合******** #
            bs, h, w, C = x_TEM.shape
            attn_TEM_OM = self.CrossAttn(m_q=x_TEM.view(bs, -1, C), m_kv=x_OM.view(bs, -1, C))  # 电镜加权光镜
            x_OM = x_OM + attn_TEM_OM.view(bs, h, w, C)
            OM_pred = self.OM_fc(x_OM.permute(0, 3, 1, 2))
            OM_output.append(OM_pred)
            x_Modals.append(x_OM)

            attn_TEM_IM = self.CrossAttn(m_q=x_TEM.view(bs, -1, C), m_kv=x_IM.view(bs, -1, C))  # 电镜加权荧光
            x_IM = x_IM + attn_TEM_IM.view(bs, h, w, C)
            IM_pred = self.IM_fc(x_IM.permute(0, 3, 1, 2))
            IM_output.append(IM_pred)
            x_Modals.append(x_IM)
            # ********模态融合******** #
            x_cat = x_Modals[0]
            for m in range(1, self.n_modal):
                x_temp = torch.cat((x_cat, x_Modals[m]), dim=-1)
                x_cat = x_temp
            out = self.MLP(x_cat.permute(0, 3, 1, 2))
            batch_output.append(out)
        # [bs,1,num_cls]
        return torch.stack(batch_output, dim=0), torch.stack(OM_output, dim=0), torch.stack(IM_output, dim=0), torch.stack(TEM_output, dim=0)
