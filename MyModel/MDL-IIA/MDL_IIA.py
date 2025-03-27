"""
MDL-IIA [1] 的pytorch复现。

[1] Predicting breast cancer types on and beyond molecular level in a multi-modal fashion
    https://github.com/Netherlands-Cancer-Institute/Multimodal_attention_DeepLearning/tree/main
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torchvision.models import resnet50, ResNet50_Weights


# 定义瓶颈块(ResNet结构的残差块)
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, with_conv_shortcut=False):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.with_conv_shortcut = with_conv_shortcut
        if self.with_conv_shortcut:
            self.conv_shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )

    def forward(self, x):
        shortcut = x
        out = self.relu(self.bn1(self.conv1(x)))  # [bs,512,16,16]
        out = self.relu(self.bn2(self.conv2(out)))  # [bs,512,8,8]
        out = self.bn3(self.conv3(out))  # [bs,2048,8,8]
        if self.with_conv_shortcut:
            shortcut = self.conv_shortcut(x)
        out += shortcut
        out = self.relu(out)
        return out

# 定义下采样模型
class DownsamplingModel(nn.Module):
    def __init__(self):
        super(DownsamplingModel, self).__init__()
        self.bottleneck1 = BottleneckBlock(1024, 512, stride=2, with_conv_shortcut=True)
        self.bottleneck2 = BottleneckBlock(2048, 512, stride=1)
        self.bottleneck3 = BottleneckBlock(2048, 512, stride=1)

    def forward(self, x):
        x = self.bottleneck1(x)  # [bs,2048,8,8]
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        return x


# self_attention
class self_attention(nn.Module):
    def __init__(self, ch):
        super(self_attention, self).__init__()
        self.channels = ch
        self.filters_f_g = self.channels // 8
        self.filters_h = self.channels

        # Define the convolutional layers
        self.conv_f = nn.Conv2d(self.channels, self.filters_f_g, kernel_size=1, bias=True)
        self.conv_g = nn.Conv2d(self.channels, self.filters_f_g, kernel_size=1, bias=True)
        self.conv_h = nn.Conv2d(self.channels, self.filters_h, kernel_size=1, bias=True)

        # Define the gamma parameter
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, _, height, width = x.size()

        # Apply convolution separately
        f = self.conv_f(x)  # [bs, filters_f_g, h, w]
        g = self.conv_g(x)
        h = self.conv_h(x)  # [bs, filter_h, h, w]

        # Reshape for matrix multiplication
        f = f.view(batch_size, self.filters_f_g, -1)  # [bs, f, N]
        g = g.view(batch_size, self.filters_f_g, -1)  # [bs, g, N]
        h = h.view(batch_size, self.filters_h, -1)  # [bs, h, N]

        # Compute attention map
        s = torch.matmul(g.transpose(1, 2), f)  # [bs, N, N]

        # Apply softmax to get the attention weights
        beta = F.softmax(s, dim=-1)  # attention map

        # Apply attention to h
        o = torch.matmul(beta, h.transpose(1, 2))  # [bs, N, C]

        # Reshape to original input shape
        o = o.view(batch_size, self.filters_h, height, width)  # [bs, C, h, w]

        # Apply gamma scaling and add original input
        x = self.gamma * o + x

        return x

# channel_spatial_attention
class channel_attention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=0.125):
        super(channel_attention,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(num_channels, int(num_channels * reduction_ratio), bias=True)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(int(num_channels * reduction_ratio), num_channels, bias=True)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x).view(x.size(0), -1))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x).view(x.size(0), -1))))
        out = avg_out.view(avg_out.size(0), avg_out.size(1),1,1) + max_out.view(max_out.size(0), max_out.size(1),1,1)
        return torch.sigmoid(out) * x

class spatial_attention(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out) * x

class CSA(nn.Module):
    def __init__(self, num_channels, reduction_ratio=0.5):
        super(CSA, self).__init__()
        self.ChannelAttn = channel_attention(num_channels, reduction_ratio)
        self.SpatialAttn = spatial_attention()

    def forward(self, x):
        channel_refined_feature = self.ChannelAttn(x)
        spatial_attention_feature = self.SpatialAttn(channel_refined_feature)
        refined_feature = channel_refined_feature * spatial_attention_feature

        return refined_feature + x

# Define the model
class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        self.base_model = nn.Sequential(*list(resnet50(weights=ResNet50_Weights).children())[:7])

    def forward(self, x):
        return self.base_model(x)

class MultiModalModel(nn.Module):
    def __init__(self, num_classes=3):
        super(MultiModalModel, self).__init__()
        self.mlo_cc_extractor = ResNetFeatureExtractor()  # shared extractor
        self.us_extractor = ResNetFeatureExtractor()
        self.basic_dim = 1024
        self.downsampling_model = DownsamplingModel()
        self.self_attention = self_attention(ch=self.basic_dim)
        self.self_attention_ccc = self_attention(ch=2048)
        self.CSA = CSA(num_channels=2*self.basic_dim)
        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(3 * 2048, 512)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(512, num_classes)

    def forward(self, mlo, cc, us):
        x_mlo = self.mlo_cc_extractor(mlo)  # [bs,C=basic_dim,H=16,W=16]
        x_cc = self.mlo_cc_extractor(cc)
        x_us = self.us_extractor(us)

        c1 = torch.cat((x_mlo, x_cc), dim=-1)  # [bs,C,H,2W]
        a1 = self.self_attention(c1)
        x11, x12 = torch.split(a1, a1.size(3)//2, dim=-1)  # [bs,1024,H,W],将合并的两个特征原样分开
        bs, c1, h1, w1 = x11.size()
        x11 = x11.view(-1, c1, h1, w1)
        x12 = x12.view(-1, c1, h1, w1)

        x_mlo = self.downsampling_model(x11)  # [bs, 2C, H/2, W/2]
        x_cc = self.downsampling_model(x12)

        a2 = self.self_attention(x_us)  # [bs,1024,16,16]
        x_us = self.downsampling_model(a2)  # [bs,2048,8,8]

        ccc = torch.cat([x_mlo, x_cc, x_us], dim=-1)  # [bs,2C,H/2,3W/2]

        a3 = self.self_attention_ccc(ccc)  # [bs,2048,8,24]
        x61,x62,x63 = torch.split(a3, a3.size(3)//3, dim=-1)  # [bs,2048,8,8]
        # 获取分割后张量的高、宽、通道数
        bs, c6, h6, w6 = x61.size()
        # 使用torch.reshape来重塑张量
        x61 = x61.view(-1, c6, h6, w6)
        x62 = x62.view(-1, c6, h6, w6)
        x63 = x63.view(-1, c6, h6, w6)
        # 使用torch.cat来拼接张量
        c6 = torch.cat([x61, x62, x63], dim=-1)  # [bs,2048,8,24]拼接w维度
        x3 = self.CSA(c6)

        x31, x32, x33 = torch.split(x3, x3.size(3)//3, dim=-1)  # [bs,2048,8,8]
        bs, c3, h3, w3 = x31.size()
        x31 = x31.view(-1, c3, h3, w3)
        x32 = x32.view(-1, c3, h3, w3)
        x33 = x33.view(-1, c3, h3, w3)
        x31 = self.GAP(x31)  # [bs,2048,1,1]
        x32 = self.GAP(x32)
        x33 = self.GAP(x33)
        x = torch.cat([x31,x32,x33],dim=1)
        x = torch.flatten(x,1)
        x = self.fc(x)
        x = self.dropout(x)
        out = self.output(x)

        return out








