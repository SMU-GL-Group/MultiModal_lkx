# coding=utf-8
"""
电镜作为q,加权光镜kv、荧光kv。注意力矩阵相加而非相乘，电镜多实例加上KNN聚类特征。模态拼接，损失函数加权。
对实例之间进行KNN距离计算，剔除掉距离低于平均距离一定阈值的的实例。
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


# -------- 找出中心点 --------
def find_center_point(matrix):
    n = matrix.shape[0]
    total_distances = []

    for i in range(n):
        # 排除对角线元素（即自己到自己的距离）
        distances_to_others = matrix[i, matrix[i] != 0]
        total_distance = np.sum(distances_to_others)
        total_distances.append(total_distance)

    # 找到总距离最小的点作为中心点
    center_index = np.argmin(total_distances)
    return center_index

# -------- 找出离群点 --------
def find_outlier_points(matrix, center_index):
    n = matrix.shape[0]
    average_distances = []

    for i in range(n):
        # 排除对角线元素（即自己到自己的距离）
        distances_to_others = matrix[i, matrix[i] != 0]
        average_distance = np.mean(distances_to_others)
        average_distances.append(average_distance)

    # 计算所有点的平均距离的平均值
    overall_average_distance = np.mean(average_distances)

    # 获取中心点与其他点的距离
    center_distances = matrix[center_index, matrix[center_index] != 0]

    # 找出中心点与其他点的距离大于整体平均距离的点
    outlier_indices = [i for i, dist in enumerate(center_distances) if dist > overall_average_distance]
    outlier_distances = [dist for i, dist in enumerate(center_distances) if dist > overall_average_distance]

    return outlier_indices, outlier_distances, overall_average_distance


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
                        TEMs.append(feats)

                    # -------- 计算KNN距离，剔除距离较远的实例 --------
                    # 在处理完一个包的所有特征后，计算特征之间的距离并剔除异常实例
                    if len(TEMs) > 2:  # 至少有三个实例才能确定谁是距离较远的点
                        # 将特征列表转换为张量
                        feats_tensor = torch.cat(TEMs, dim=0)  # [num_instances, C, H, W]
                        feats_tensor = feats_tensor.view(feats_tensor.size(0), -1)  # 展平为 [num_instances, C*H*W]

                        # 计算特征之间的欧氏距离矩阵
                        distance_matrix = euclidean_distances(
                            feats_tensor.cpu().detach().numpy())  # [num_instances, num_instances]

                        # 找出中心点，根据其它点与中心点的距离是否大于平均距离找出离群点
                        center_index = find_center_point(distance_matrix)
                        outlier_index, outlier_distance, avg_distance = find_outlier_points(distance_matrix, center_index)

                        # 剔除距离大于阈值的实例
                        TEMs = [TEMs[i] for i in range(len(TEMs)) if i not in outlier_index]

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
