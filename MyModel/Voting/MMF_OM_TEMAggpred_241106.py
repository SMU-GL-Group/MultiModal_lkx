#coding=utf-8
"""
电镜每张图像预测一次，求和预测结果（投票）作为病人的最终电镜的预测结果。该结果再与光镜预测结果进行相加。
"""
import torch.nn as nn
import torch
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
        x = self.avgpool(x) # 1,12,8,8 # 特征维度高，一般更适合分类
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
        self.TEM_encoder = nn.Sequential(*list(swin_transformer.swin_b(weights=Swin_B_Weights).children())[:-4])
        self.TEM_MLP = MLP(num_i=1024, num_h=128, num_o=args.num_cls)
        self.OM_encoder = nn.Sequential(*list(swin_transformer.swin_b(weights=Swin_B_Weights).children())[:-4])
        self.OM_MLP = MLP(num_i=1024, num_h=128, num_o=args.num_cls)

        self.t = transforms.ToPILImage()

    def forward(self, bag_tensor, OM_tensor):
        batch_output, OM_output, TEM_output = [],[],[] # 一个batch的输出
        # 先对电镜图像进行特征提取
        bag_tuple = torch.split(bag_tensor, 1) # 先按batch分开 [bs,图像数,C,H,W ]-> bs * [1,图像数,C,H,W]
        for bs in range(len(bag_tuple)): # 第bs个batch
            for bag in bag_tuple[bs]: # 第bag个包
                bag = bag[:6,:,:,:] # 取若干张电镜，太多内存会爆
                Preds = []  # 一个包内所有图像的预测结果
                img_tuple = torch.split(bag, 1)
                for i in img_tuple:
                    # # 查看每张图像
                    # img = self.t(torch.squeeze(i, dim=0))
                    # plt.imshow(img)
                    # plt.show()
                    if not torch.equal(i.cpu(), torch.zeros(i.shape)): # 跳过空张量
                        TEM_feat = self.TEM_encoder(i) # [1,7,7,1024]
                        TEM_pred = self.TEM_MLP(TEM_feat.permute(0,3,1,2)) # 每张TEM图像进行一次预测
                        Preds.append(TEM_pred)
            TEM_preds = torch.stack(Preds).sum(dim=0) # 一个包内所有预测结果求和作为病人的最终预测
            TEM_output.append(TEM_pred)

            OM_feat = self.OM_encoder(torch.unsqueeze(OM_tensor[bs], dim=0))
            OM_pred = self.OM_MLP(OM_feat.permute(0,3,1,2))
            OM_output.append(OM_pred)

            patient_pred = TEM_preds + OM_pred
            batch_output.append(patient_pred)

        return torch.stack(batch_output,dim=0), torch.stack(OM_output, dim=0), torch.stack(TEM_output, dim=0)
