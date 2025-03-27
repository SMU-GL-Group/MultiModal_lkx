#coding=utf-8
"""
OM、IM、TEM均使用MultiScaleAttention[1]提取特征,TEM固定一张图像。

[1] Multi-Modal Retinal Image Classification With Modality-Specific Attention Network
"""
import torch
import torch.nn as nn
from MyModel.Modules.MASN_attention import PAM_Module
from MyModel.MASN_ResNet import BasicBlock,Bottleneck,conv1x1
from torchvision.models import swin_transformer, Swin_B_Weights
from torchvision.models import resnet18, ResNet18_Weights


# ======================= 预测最终结果的MLP ======================= #
class Encoder(nn.Module):
    def __init__(self,):
        super(Encoder, self).__init__()

        # 载入resnet预训练权重
        self.conv1 = resnet18().conv1
        self.bn1 = resnet18().bn1
        self.relu = resnet18().relu
        self.maxpool = resnet18().maxpool

        self.layer1 = resnet18().layer1
        self.layer2 = resnet18().layer2
        self.layer3 = resnet18().layer3
        self.layer4 = resnet18().layer4

        self.avgpool = resnet18().avgpool

        # MSA块
        self.pam_attention = PAM_Module(64)
        self.pam_attention2 = PAM_Module(128)
        self.pam_attention3 = PAM_Module(256)
        self.pam_attention4 = PAM_Module(512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        PA = self.pam_attention(x, x, x)
        # draw_features(PA, 'resnet18_OCT_focal_with_spatial_attention_Resblock1')
        x = self.layer2(PA)
        PA2 = self.pam_attention2(x, x, x)
        x = self.layer3(PA2)
        PA3 = self.pam_attention3(x, x, x)
        x = self.layer4(PA3)
        PA4 = self.pam_attention4(x, x, x)
        x = self.avgpool(PA4)

        return x

# ======================= 预测最终结果的MLP ======================= #
class MLP(nn.Module):
    def __init__(self, num_i, num_h, num_o):
        super(MLP, self).__init__()

        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
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
    def __init__(self, args, block=BasicBlock, layers=[2, 2, 2, 2], zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(Model, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 128
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=2, stride=2, #padding=2,
        #                        bias=False)
        # self.bn1 = norm_layer(self.inplanes)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
        #                                dilate=replace_stride_with_dilation[0])
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
        #                                dilate=replace_stride_with_dilation[1])
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
        #                                dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AvgPool2d(7, stride=1)

        basic_dim = 512
        self.OM_encoder = Encoder()
        self.IM_encoder = Encoder()
        self.TEM_encoder = Encoder()

        self.fc = nn.Linear(basic_dim * block.expansion, args.num_cls)
        self.cls_OM = nn.Sequential(nn.Dropout(p=0.5),nn.Linear(basic_dim, args.num_cls))
        self.cls_IM = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(basic_dim, args.num_cls))
        self.cls_TEM = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(basic_dim, args.num_cls))

        self.args = args
        self.n_modal = len(self.args.modal.split('+'))  # 模态数量

        # MLP
        self.MLP = MLP(num_i=basic_dim * self.n_modal,
                       num_h=128, num_o=args.num_cls)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, TEM_tensor, OM_tensor, IM_tensor):
        x_TEM = self.TEM_encoder(TEM_tensor)
        x_TEM = x_TEM.view(x_TEM.size(0), -1)  # [bs,basic_dim]
        TEM_pred = self.cls_TEM(x_TEM)

        x_IM = self.IM_encoder(IM_tensor)
        x_IM = x_IM.view(x_IM.size(0), -1)
        IM_pred = self.cls_IM(x_IM)

        x_OM = self.OM_encoder(OM_tensor)
        x_OM = x_OM.view(x_OM.size(0),-1)
        OM_pred = self.cls_OM(x_OM) #.squeeze()
        # ********模态融合******** #
        x_cat = torch.cat([x_OM, x_IM, x_TEM], dim=1)
        out = self.MLP(x_cat)

        return out, OM_pred, IM_pred, TEM_pred
