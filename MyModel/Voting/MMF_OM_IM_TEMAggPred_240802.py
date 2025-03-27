#coding=utf-8
"""
电镜每张图像预测一次，求和预测结果（投票）作为病人的最终电镜的预测结果。该结果再与光镜、荧光预测结果进行相加。
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
        self.TEM_encoder = nn.Sequential(*list(swin_transformer.swin_b(weights=Swin_B_Weights).children()))
        self.TEM_encoder[-1] = nn.Linear(in_features=1024, out_features=args.num_cls, bias=True) # 最终的输出从1000类改为num_cls类 # todo 不公平，最后几层还是用到了Swin的权重
        self.OM_encoder = nn.Sequential(*list(swin_transformer.swin_b(weights=Swin_B_Weights).children()))
        self.OM_encoder[-1] = nn.Linear(in_features=1024, out_features=args.num_cls, bias=True)
        self.IM_encoder = nn.Sequential(*list(swin_transformer.swin_b(weights=Swin_B_Weights).children()))
        self.IM_encoder[-1] = nn.Linear(in_features=1024, out_features=args.num_cls, bias=True)

        # MLP
        self.MLP = MLP(num_i=1024,
                       num_h=128, num_o=args.num_cls)

        self.t = transforms.ToPILImage()

    def forward(self, bag_tensor, OM_tensor, IM_tensor):
        batch_output = [] # 一个batch的输出
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
                        output = self.TEM_encoder(i) # [1,7,7,1024]
                        # output = self.MLP(feats.permute(0,3,1,2)) # 每张TEM图像进行一次预测
                        Preds.append(output)
            TEM_pred = torch.stack(Preds).sum(dim=0) # 一个包内所有预测结果求和作为病人的最终预测

            OM_pred = self.OM_encoder(torch.unsqueeze(OM_tensor[bs], dim=0))
            IM_pred = self.IM_encoder(torch.unsqueeze(IM_tensor[bs], dim=0))

            patient_pred = TEM_pred + OM_pred + IM_pred
            batch_output.append(patient_pred)

        return torch.stack(batch_output,dim=0) # [bs,1,num_cls]
