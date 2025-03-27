#coding=utf-8
"""
对AGGN/base_model2.py中的Mainnet模型进行适度调整以符合所需输入。
改动：去掉多尺度不同阶段的特征融合的部分，验证是否是这一过程导致MN、LN分类差。
"""
import sys

sys.path.append('../')
sys.path.append('../../')
import torch
import torch.nn as nn
from torch.nn import init

# ======================= 初始化模型参数 ======================= #
def init_weights(model, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                pass
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    model.apply(init_func)

def init_net(model, init_type='normal', init_gain=0.02):
    init_weights(model, init_type, gain=init_gain)


class conv3d_block(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding):
        super(conv3d_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.PReLU(num_parameters=1, init=0.25),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

# ======================= 设置卷积块(含BN-PRelu) ======================= #
class conv2d_block(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding):
        super(conv2d_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.PReLU(num_parameters=1, init=0.25),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv2dx2_block(nn.Module):  # baseblock
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding):
        super(conv2dx2_block, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.PReLU(num_parameters=1, init=0.25),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.PReLU(num_parameters=1, init=0.25),
        )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        return x


class conv1d_block(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding):
        super(conv1d_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.PReLU(num_parameters=1, init=0.25),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Res_block(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding):
        super(Res_block, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.PReLU(num_parameters=1, init=0.25),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.PReLU(num_parameters=1, init=0.25),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(ch_out),
        )
        self.activetion = nn.PReLU(num_parameters=1, init=0.25)

    def forward(self, x):
        x0 = self.conv0(x)
        x = self.conv1(x0)
        x = self.conv2(x)
        x = x0 + x
        x = self.activetion(x)
        return x


class Dense_block(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding):
        super(Dense_block, self).__init__()
        self.grow = ch_out - ch_in
        self.grow_flag = False
        if self.grow == 0:
            self.grow_flag = True
            self.grow = ch_out // 2
            self.ch_in_reduction = nn.Sequential(
                nn.PReLU(num_parameters=1, init=0.25),
                nn.BatchNorm2d(ch_in),
                nn.Conv2d(ch_in, ch_out // 2, kernel_size=1, stride=1, padding=0, bias=True),
            )
        self.conv0 = nn.Sequential(
            nn.PReLU(num_parameters=1, init=0.25),
            nn.BatchNorm2d(ch_in),
            nn.Conv2d(ch_in, self.grow * 4, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.conv1 = nn.Sequential(
            nn.PReLU(num_parameters=1, init=0.25),
            nn.BatchNorm2d(self.grow * 4),
            nn.Conv2d(self.grow * 4, self.grow, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        )

    def forward(self, x):
        if self.grow_flag:
            x0 = self.ch_in_reduction(x)
        else:
            x0 = x
        x = self.conv0(x)
        x = self.conv1(x)
        x = torch.cat((x0, x), 1)
        return x

# ======================= 双域注意力机制模块Dual-domain Attention mechanism module之通道注意力 ======================= #
class SE_block(nn.Module):
    def __init__(self, ch_in, data_size):
        super(SE_block, self).__init__()
        self.reduction_block = adaptive_reduction_block_three_branch(ch_in, ch_in, data_size, [1, 1])
        self.fc1 = nn.Sequential( # filters=8(32/4)
            nn.Conv2d(ch_in, ch_in // 4, kernel_size=1, stride=1, padding=0, bias=True),
            nn.PReLU(num_parameters=1, init=0.25),
        )
        self.fc2 = nn.Sequential( # filters=32
            nn.Conv2d(ch_in // 4, ch_in, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        x = self.reduction_block(x) # 三个分支减少特征图尺寸至[1,1]
        x1 = self.fc1(x)
        # print("x1",np.shape(x1))
        x2 = self.fc2(x1)
        # print("x2",np.shape(x2))
        x = self.sigmoid(x2)
        x = input * x
        return x


class Self_Attn(nn.Module):
    def __init__(self, ch_in):
        super(Self_Attn, self).__init__()
        self.chanel_in = ch_in
        self.activation = nn.PReLU(num_parameters=1, init=0.25)

        self.query_conv = nn.Conv2d(in_channels=ch_in, out_channels=ch_in // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=ch_in, out_channels=ch_in // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=ch_in, out_channels=ch_in, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention

# ======================= 多尺度分支卷积MultiBranch conv的三个分支 ======================= #
class Multiconv_block1(nn.Module):  # 多感受野卷积块-1
    def __init__(self, ch_in, ch_out):
        super(Multiconv_block1, self).__init__()
        # 左侧分支
        self.branch0 = nn.Sequential(
            # ACB非对称卷积块(内含BN-PRelu)
            conv2d_block(ch_in, ch_out // 8 * 4,  kernel_size=(1, 7), stride=1, padding=(0, 3)),
            conv2d_block(ch_out // 8 * 4, ch_out // 8 * 4, (7, 1), 1, (3, 0)),
            conv2d_block(ch_out // 8 * 4, ch_out // 8 * 4, kernel_size=3, stride=2, padding=1),
        )
        # 右侧分支
        self.branch1 = nn.Sequential(
            conv2d_block(ch_in, ch_out // 8 * 1, kernel_size=1, stride=1, padding=0), # filters=8的1x1conv
            nn.MaxPool2d(kernel_size=5, stride=2, padding=2),
        )
        # 右侧分支
        self.branch2 = nn.Sequential(
            # ACB非对称卷积块(内含BN-PRelu)
            conv2d_block(ch_in, ch_out // 8 * 3, kernel_size=(1, 11), stride=1, padding=(0, 5)),
            conv2d_block(ch_out // 8 * 3, ch_out // 8 * 3, (11, 1), 1, (5, 0)),
            conv2d_block(ch_out // 8 * 3, ch_out // 8 * 3, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        x = torch.cat([x0, x1, x2], 1)
        return x


class Multiconv_block2(nn.Module):  # 多感受野卷积块-2
    def __init__(self, ch_in, ch_out):
        super(Multiconv_block2, self).__init__()
        self.branch0 = conv2d_block(ch_in, ch_out // 8 * 4, 3, 2, 1)
        self.branch1 = nn.Sequential(
            conv2d_block(ch_in, ch_out // 8 * 1, 1, 1, 0),
            nn.MaxPool2d(5, stride=2, padding=2),
        )
        self.branch2 = nn.Sequential(
            conv2d_block(ch_in, ch_out // 8 * 3, (1, 7), 1, (0, 3)),
            conv2d_block(ch_out // 8 * 3, ch_out // 8 * 3, (7, 1), 1, (3, 0)),
            conv2d_block(ch_out // 8 * 3, ch_out // 8 * 3, (1, 5), 1, (0, 2)),
            conv2d_block(ch_out // 8 * 3, ch_out // 8 * 3, (5, 1), 1, (2, 0)),
            conv2d_block(ch_out // 8 * 3, ch_out // 8 * 3, 3, 2, 1),
        )
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = torch.cat([x0, x1, x2], 1)
        return x


class Multiconv_block3(nn.Module):  # 多感受野池化块
    def __init__(self, ch_in, ch_out):
        super(Multiconv_block3, self).__init__()
        self.branch0 = nn.Sequential(
            conv2d_block(ch_in, ch_out // 8 * 4, (1, 5), 1, (0, 2)),
            conv2d_block(ch_out // 8 * 4, ch_out // 8 * 4, (5, 1), 1, (2, 0)),
            conv2d_block(ch_out // 8 * 4, ch_out // 8 * 4, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.branch1 = nn.Sequential(
            conv2d_block(ch_in, ch_out // 8 * 3, 1, 1, 0),
            nn.AdaptiveAvgPool2d(4),
            nn.AdaptiveAvgPool2d(1),
        )
        self.branch2 = nn.Sequential(
            conv2d_block(ch_in, ch_out // 8 * 1, 1, 1, 0),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = torch.cat([x0, x1, x2], 1)
        return x


class adaptive_reduction_block(nn.Module):  # 可能没有出现过
    def __init__(self, ch_in, ch_out, data_size, reduction_size):
        super(adaptive_reduction_block, self).__init__()
        _, _, h, w = data_size
        rh, rw = reduction_size
        kh = h - rh + 1
        kw = w - rw + 1
        self.conv = nn.Sequential(
            conv2d_block(ch_in, ch_out, (1, kw), 1, (0, 0)),
            conv2d_block(ch_out, ch_out, (kh, 1), 1, (0, 0)),
            conv2d_block(ch_out, ch_out, 3, 1, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

# ======================= 通道注意力Channel Attention的三个分支 ======================= #
class adaptive_reduction_block_three_branch(nn.Module):  # 多尺度降维块
    def __init__(self, ch_in, ch_out, data_size, reduction_size):
        super(adaptive_reduction_block_three_branch, self).__init__()
        _, _, h, w = data_size
        rh, rw = reduction_size # 减小后的特征图尺寸
        kh = h - rh + 1 # 卷积核大小
        kw = w - rw + 1
        if h <= 7:
            branch1_h = h
        else:
            branch1_h = 7
        if w <= 7:
            branch1_w = w
        else:
            branch1_w = 7
        # 左侧分支 filters=12(32//8*3) ACB非对称卷积
        self.branch0 = nn.Sequential(
            conv2d_block(ch_in, ch_out // 8 * 3, kernel_size=(1, kh), stride=1, padding=(0, 0)),
            conv2d_block(ch_out // 8 * 3, ch_out // 8 * 3, (kw, 1), 1, (0, 0)),  # 拆成两半
            conv2d_block(ch_out // 8 * 3, ch_out // 8 * 3, 3, 1, 1),
        )
        # 中间分支 filters=16(32//8*4) ACB非对称卷积
        self.branch1 = nn.Sequential(
            conv2d_block(ch_in, ch_out // 8 * 4, kernel_size=(1, branch1_w), stride=1, padding=(0, 0)),
            conv2d_block(ch_out // 8 * 4, ch_out // 8 * 4, (branch1_h, 1), 1, (0, 0)),
            conv2d_block(ch_out // 8 * 4, ch_out // 8 * 4, 3, 1, 1),
            nn.AdaptiveAvgPool2d((rh, rw)),
        )
        # 右侧分支 filters=4(32//8*1) 均值池化kernel_size=192,输出特征图尺寸=[rh,rw]
        self.branch2 = nn.Sequential(
            conv2d_block(ch_in, ch_out // 8 * 1, 1, 1, 0),
            nn.AdaptiveAvgPool2d((rh, rw)),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = torch.cat((x0, x1, x2), 1)
        return x


class fusion2d_block(nn.Module):  # 多模态融合块
    def __init__(self, ch_in, ch_out, model_in, datasize=[1, 1, 1, 1]):
        super(fusion2d_block, self).__init__()
        self.pre_branch0 = nn.Sequential(
            conv3d_block(ch_in, ch_out // 8 * 3, (3, 3, 1), 1, (1, 1, 0)),
            conv3d_block(ch_out // 8 * 3, ch_out // 8 * 3, (1, 1, model_in), 1, (0, 0, 0)),
        )
        self.pre_branch1 = nn.Sequential(
            conv3d_block(ch_in, ch_out // 8 * 2, (3, 1, 1), 1, (1, 0, 0)),
            conv3d_block(ch_out // 8 * 2, ch_out // 8 * 2, (1, 3, 1), 1, (0, 1, 0)),
            conv3d_block(ch_out // 8 * 2, ch_out // 8 * 2, (1, 1, model_in), 1, (0, 0, 0)),
        )
        self.pre_branch2 = nn.Sequential(
            conv3d_block(ch_in, ch_out // 8 * 3, (3, 3, model_in), 1, (1, 1, 0)),
        )
        self.post_conv = conv2d_block(ch_out, ch_out, 3, 1, 1)

    def forward(self, x):
        x0 = self.pre_branch0(x)
        x1 = self.pre_branch1(x)
        x2 = self.pre_branch2(x)
        x0 = x0.squeeze(-1)
        x1 = x1.squeeze(-1)
        x2 = x2.squeeze(-1)
        x = torch.cat((x0, x1, x2), 1)
        x = self.post_conv(x)
        return x


class SE_fusion2d_block(nn.Module):
    def __init__(self, ch_in, ch_out, model_in, datasize):
        super(SE_fusion2d_block, self).__init__()
        self.pre_branch0 = nn.Sequential(
            conv3d_block(ch_in, ch_out // 8 * 1, (3, 3, 1), 1, (1, 1, 0)),
            conv3d_block(ch_out // 8 * 1, ch_out // 8 * 1, (1, 1, model_in), 1, (0, 0, 0)),
        )
        self.pre_branch1 = nn.Sequential(
            conv3d_block(ch_in, ch_out // 8 * 2, (3, 1, 1), 1, (1, 0, 0)),
            conv3d_block(ch_out // 8 * 2, ch_out // 8 * 2, (1, 3, 1), 1, (0, 1, 0)),
            conv3d_block(ch_out // 8 * 2, ch_out // 8 * 2, (1, 1, model_in), 1, (0, 0, 0)),
        )
        self.pre_branch2 = nn.Sequential(
            conv3d_block(ch_in, ch_out // 8 * 1, (3, 3, model_in), 1, (1, 1, 0)),
        )
        self.pre_branch3 = nn.Sequential(
            SE_block(ch_in * model_in, datasize),
            conv2d_block(ch_in * model_in, ch_out // 8 * 4, 3, 1, 1),
        )
        self.post_conv = conv2d_block(ch_out, ch_out, 3, 1, 1)

    def forward(self, x):
        x0 = self.pre_branch0(x)
        x1 = self.pre_branch1(x)
        x2 = self.pre_branch2(x)
        x0 = x0.squeeze(-1)
        x1 = x1.squeeze(-1)
        x2 = x2.squeeze(-1)
        x3 = x.permute(0, 1, 4, 2, 3)
        x3 = x3.reshape(x3.size(0), x3.size(1) * x3.size(2), 1, x3.size(3), x3.size(4))
        x3 = x3.squeeze(2)
        x3 = self.pre_branch3(x3)
        x = torch.cat((x0, x1, x2, x3), 1)
        x = self.post_conv(x)
        return x


class SE_fusion2d_block_with_pooling(nn.Module):
    def __init__(self, ch_in, ch_out, model_in, datasize):
        super(SE_fusion2d_block_with_pooling, self).__init__()
        self.pre_branch0 = nn.Sequential(
            conv3d_block(ch_in, ch_out // 8 * 1, 1, 1, 0),
            nn.MaxPool3d(kernel_size=(3, 3, model_in), stride=(1, 1, 2), padding=(1, 1, 0)),
        )
        self.pre_branch1 = nn.Sequential(
            conv3d_block(ch_in, ch_out // 8 * 3, (3, 1, 1), 1, (1, 0, 0)),
            conv3d_block(ch_out // 8 * 3, ch_out // 8 * 3, (1, 3, 1), 1, (0, 1, 0)),
            conv3d_block(ch_out // 8 * 3, ch_out // 8 * 3, (1, 1, model_in), 1, (0, 0, 0)),
        )
        self.pre_branch2 = nn.Sequential(
            conv3d_block(ch_in, ch_out // 8 * 1, 1, 1, 0),
            nn.AvgPool3d(kernel_size=(3, 3, model_in), stride=(1, 1, 2), padding=(1, 1, 0)),
        )
        self.pre_branch3 = nn.Sequential(
            SE_block(ch_in * model_in, datasize),
            conv2d_block(ch_in * model_in, ch_out // 8 * 3, 3, 1, 1),
        )
        self.post_conv = conv2d_block(ch_out, ch_out, 3, 1, 1)

    def forward(self, x):
        x0 = self.pre_branch0(x)
        x1 = self.pre_branch1(x)
        x2 = self.pre_branch2(x)
        x0 = x0.squeeze(-1)
        x1 = x1.squeeze(-1)
        x2 = x2.squeeze(-1)
        x3 = x.permute(0, 1, 4, 2, 3)
        x3 = x3.reshape(x3.size(0), x3.size(1) * x3.size(2), 1, x3.size(3), x3.size(4))
        x3 = x3.squeeze(2)
        x3 = self.pre_branch3(x3)
        x = torch.cat((x0, x1, x2, x3), 1)
        x = self.post_conv(x)
        return x

# ======================= 分类token的卷积 ======================= #
class fusion1d_block(nn.Module):
    def __init__(self, ch_in, ch_out, model_in):
        super(fusion1d_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=model_in, stride=1, padding=0, bias=True, groups=ch_in),
            nn.BatchNorm1d(ch_out),
            nn.PReLU(num_parameters=1, init=0.25),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.squeeze(-1)
        return x


class base_encoder(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(base_encoder, self).__init__()
        self.block1 = nn.Sequential(
            conv2d_block(ch_in, 64, 3, 1, 1),
            Multiconv_block1(64, 64),
        )

        self.block2 = nn.Sequential(
            conv2d_block(64, 128, 3, 1, 1),
            Multiconv_block2(128, 128),
        )

        self.stage3 = nn.Sequential(
            conv2d_block(128, 256, 3, 1, 1),
            conv2d_block(256, 256, 3, 1, 1),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.stage4 = nn.Sequential(
            conv2d_block(256, 256, 3, 1, 1),
            conv2d_block(256, 256, 3, 1, 1),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.stage5 = nn.Sequential(
            conv2d_block(256, 256, 3, 1, 1),
            conv2d_block(256, 256, 3, 1, 1),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.block3 = Multiconv_block3(256, ch_out)
        self.block4 = single_classifier(2048, 2)

    def forward(self, x):
        x11 = x[:, 0, :]
        x11 = torch.reshape(x11, shape=(32, 1, 192, 192))
        x22 = x[:, 1, :]
        x22 = torch.reshape(x22, shape=(32, 1, 192, 192))
        x33 = x[:, 2, :]
        x33 = torch.reshape(x33, shape=(32, 1, 192, 192))
        x44 = x[:, 3, :]
        x44 = torch.reshape(x44, shape=(32, 1, 192, 192))
        x1 = self.block1(x11)
        x2 = self.block2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        xa = self.block3(x5)
        xa = xa.view(x.size(0), -1)

        x1b = self.block1(x22)
        x2b = self.block2(x1b)
        x3b = self.stage3(x2b)
        x4b = self.stage4(x3b)
        x5b = self.stage5(x4b)
        xb = self.block3(x5b)
        xb = xb.view(x.size(0), -1)

        x1c = self.block1(x33)
        x2c = self.block2(x1c)
        x3c = self.stage3(x2c)
        x4c = self.stage4(x3c)
        x5c = self.stage5(x4c)
        xc = self.block3(x5c)
        xc = xc.view(x.size(0), -1)

        x1d = self.block1(x44)
        x2d = self.block2(x1d)
        x3d = self.stage3(x2d)
        x4d = self.stage4(x3d)
        x5d = self.stage5(x4d)
        xd = self.block3(x5d)
        xd = xd.view(x.size(0), -1)

        xz = torch.cat((xa, xb, xc, xd), 1)

        xz = self.block4(xz)
        return xz

    # x1 = self.block1(x)
    # x2 = self.block2(x1)
    # x3 = self.stage3(x2)
    # x4 = self.stage4(x3)
    # x5 = self.stage5(x4)
    # x = self.block3(x5)
    # x = x.view(x.size(0), -1)
    # return x, {'stage3': x3, 'stage4': x4, 'stage5': x5}

# ======================= 多尺度特征提取 ======================= #
class base_encoder_2(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(base_encoder_2, self).__init__()
        # 多尺度分支卷积MultiBranch conv(MBconv)-1 [c,192,192]->[64,96,96]
        self.block1 = nn.Sequential(
            conv2d_block(ch_in, ch_out=64, kernel_size=3, stride=1, padding=1),
            Multiconv_block1(ch_in=64, ch_out=64),
        )
        # 多尺度分支卷积MultiBranch conv(MBconv)-2 [64,96,96]->[128,48,48]
        self.block2 = nn.Sequential(
            conv2d_block(64, 128, 3, 1, 1),
            Multiconv_block2(128, 128),
        )
        # 卷积池化Convolution-Pooling-1 [128,48,48]->[256,24,24]
        self.stage3 = nn.Sequential(
            conv2d_block(128, 256, 3, 1, 1),
            conv2d_block(256, 256, 3, 1, 1),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        # 卷积池化Convolution-Pooling-2 ->[256,12,12]
        self.stage4 = nn.Sequential(
            conv2d_block(256, 256, 3, 1, 1),
            conv2d_block(256, 256, 3, 1, 1),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        # 卷积池化Convolution-Pooling-3 ->[256,6,6]
        self.stage5 = nn.Sequential(
            conv2d_block(256, 256, 3, 1, 1),
            conv2d_block(256, 256, 3, 1, 1),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        # 多尺度分支池化MultiBranch pooling ->[256,1,1]
        self.block3 = Multiconv_block3(256, ch_out)

    def forward(self, x):
        x1 = self.block1(x)  # [bs,c,192,192]->[bs,64,96,96]
        x2 = self.block2(x1)  # [bs,128,48,48]
        x3 = self.stage3(x2)  # [bs,256,24,24]
        x4 = self.stage4(x3)  # [bs,256,12,12]
        x5 = self.stage5(x4)  # [bs,256,6,6]
        x = self.block3(x5)  # [bs,ch_out,1,1]
        x = x.view(x.size(0), -1)  # [bs,ch_out]
        return x, {'stage3': x3, 'stage4': x4, 'stage5': x5}

# ======================= 双域注意力机制模块Dual-domain Attention mechanism module之空间注意力 ======================= #
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        x = input * x
        return x


class Model(nn.Module):
    def __init__(self, ch_in, ch_out, data_size, ch_out_2, args):
        """

        :param ch_in: 输入图像的维度
        :param ch_out: 最终输出至分类器的特征维度
        :param data_size: 输入的数据格式大小,前两位随便,后两位为输入图像的尺寸
        :param ch_out_2: 多模态信息融合部分(各阶段C-P块的输出部分)最终输出的特征维度
        :param args: 外置参数
        """
        super(Model, self).__init__()
        self.n_modal = len(args.modal.split('+'))  # 模态数量
        self.model_dict = {}
        self.MultiScaleEncoder = base_encoder_2(ch_in, ch_out) # 多尺度特征提取器
        self.classifier = single_classifier(ch_out*4, args.num_cls) # fc线性分类器
        self.ChannelAttn = SE_block(ch_in=args.batch_size,data_size=[ch_in*self.n_modal,args.batch_size,args.reshape[0], args.reshape[1]]) # 通道注意力
        self.stage3_fusion = nn.Sequential( # [1024,24,24]的fusion conv
            fusion2d_block(256, 256, self.n_modal),
            #                 adaptive_reduction_block(256, 256, [data_size[0], data_size[1], data_size[2] // 8, data_size[3] // 8], [data_size[2] // 32, data_size[3] // 32]),
            adaptive_reduction_block_three_branch(ch_in=256, ch_out=256,
                                                  data_size=[data_size[0], data_size[1], data_size[2] // 8, data_size[3] // 8],
                                                  reduction_size=[data_size[2] // 32, data_size[3] // 32]),  # 输出的特征图尺寸
            Multiconv_block3(256, ch_out_2),
        )
        self.stage4_fusion = nn.Sequential( # [1024,12,12]的fusion conv
            fusion2d_block(256, 256, self.n_modal),
            #                 adaptive_reduction_block(256, 256, [data_size[0], data_size[1], data_size[2] // 16, data_size[3] // 16], [data_size[2] // 32, data_size[3] // 32]),
            adaptive_reduction_block_three_branch(256, 256,
                                                  [data_size[0], data_size[1], data_size[2] // 16, data_size[3] // 16],
                                                  [data_size[2] // 32, data_size[3] // 32]),
            Multiconv_block3(256, ch_out_2),
        )
        self.stage5_fusion = nn.Sequential(# [1024,6,6]的fusion conv
            fusion2d_block(256, 256, self.n_modal),
            Multiconv_block3(256, ch_out_2),
        )
        self.z_fusion = fusion1d_block(ch_out_2, ch_out_2, self.n_modal)  # k=n_modal,s=1的卷积+BN-PRelu
        self.dropout = nn.Dropout(0.4)
        self.SpatialAttn = SpatialAttention(kernel_size=7)

    def forward(self, TEM_tensor, OM_tensor, IM_tensor):
        # print(np.shape(x))
        # x = torch.reshape(x, shape=(4, 32, 192, 192))
        # x = self.block3(x)
        # x = torch.reshape(x, shape=(32, 4, 192, 192))
        bs, c, h, w = TEM_tensor.shape
        x = torch.cat([TEM_tensor, OM_tensor,IM_tensor],dim=1)  # [bs,3c,192,192]
        x = torch.reshape(x, shape=(bs, c*self.n_modal, h, w))  # 将输入整形为[bs=32,num_modals,192,192]
        x = x.transpose(0, 1)  # [3c,bs,192,192]
        # 双域注意力机制
        x = self.SpatialAttn(x) # 空间注意力
        x = self.ChannelAttn(x) # 通道注意力
        x = x.transpose(1, 0)  # [bs,3c,192,192]
        # 将各模态重新分离
        x11 = x[:, 0:c, :, :]
        x11 = torch.reshape(x11, shape=(bs, c, h, w))
        x22 = x[:, c:2*c, :, :]
        x22 = torch.reshape(x22, shape=(bs, c, h, w))
        x33 = x[:, 2*c:3*c, :, :]
        x33 = torch.reshape(x33, shape=(bs, c, h, w))
        # x44 = x[:, 3, :, :]
        # x44 = torch.reshape(x44, shape=(32, 1, 192, 192))

        # x11 = self.dropout(x11)
        # x22 = self.dropout(x22)
        # x33 = self.dropout(x33)
        # x44 = self.dropout(x44)
        # 各模态进行多尺度特征提取,得到y_base_1=[bs,ch_out]和y_base_2={[bs,256,24,24], [bs,256,12,12], [bs,256,6,6]}
        y_base_a1, y_base_a2 = self.MultiScaleEncoder(x11)
        y_base_b1, y_base_b2 = self.MultiScaleEncoder(x22)
        y_base_c1, y_base_c2 = self.MultiScaleEncoder(x33)
        # y_base_d1, y_base_d2 = self.MultiScaleEncoder(x44)

        self.model_dict['model_' + str(0)] = y_base_a1, y_base_a2
        self.model_dict['model_' + str(1)] = y_base_b1, y_base_b2
        self.model_dict['model_' + str(2)] = y_base_c1, y_base_c2
        # self.model_dict['model_' + str(3)] = y_base_d1, y_base_d2

        z_local = {}  # 存放分类token [bs,ch_out]
        features = {}  # 存放不同阶段的C-P块输出
        for i in range(self.n_modal):
            tz, tfeatures = self.model_dict['model_' + str(i)]
            z_local['z_local_' + str(i)] = tz
            features['features_' + str(i)] = tfeatures
            if i == 0:
                z = tz.unsqueeze(-1)  # [bs,ch_out,1]
                stage3 = tfeatures['stage3'].unsqueeze(-1)  # [bs,256,24,24,1]
                stage4 = tfeatures['stage4'].unsqueeze(-1)
                stage5 = tfeatures['stage5'].unsqueeze(-1)
            else:
                z = torch.cat((z, tz.unsqueeze(-1)), -1)  # 将各模态的分类token输出和3个阶段的C-P块输出堆叠 [bs,ch_out,n_modal]
                stage3 = torch.cat((stage3, tfeatures['stage3'].unsqueeze(-1)), -1)  # [bs,256,24,24,n_modal]
                stage4 = torch.cat((stage4, tfeatures['stage4'].unsqueeze(-1)), -1)
                stage5 = torch.cat((stage5, tfeatures['stage5'].unsqueeze(-1)), -1)

        z3 = self.stage3_fusion(stage3)  # fusion conv [bs,256,24,24,n_modal]->[bs,ch_out,1,1]
        # z3 = self.dropout(z3)
        z3 = z3.view(z3.size(0), -1)  # [bs,ch_out]
        z4 = self.stage4_fusion(stage4)
        # z4 = self.dropout(z4)
        z4 = z4.view(z4.size(0), -1)
        z5 = self.stage5_fusion(stage5)
        # z5 = self.dropout(z5)
        z5 = z5.view(z5.size(0), -1)
        z = self.z_fusion(z)  # k=n_modal,s=1的卷积+BN-PRelu [bs,ch_out,n_modal]->[bs,ch_out]
        # z = torch.cat((z, z3, z4, z5), 1)
        # z = torch.cat((z, z3, z4, z5, y_base_a1, y_base_b1, y_base_c1, y_base_d1), 1)
        # 将各模态的C-P原始输出y_base_1,经fusion conv后的3阶段C-P块输出z3-z5,以及各模态总的分类token z拼接。
        # z = torch.cat((z, z3, z4, z5, y_base_a1, y_base_b1, y_base_c1), 1)  # [bs,ch_out*7]
        z = torch.cat((z,y_base_a1, y_base_b1, y_base_c1),1) # [bs, ch_out*4]
        z = self.dropout(z)
        z = self.classifier(z)

        return z

    def init_net(model, init_type='normal', init_gain=0.02):
        init_weights(model, init_type, gain=init_gain)

class Model2(nn.Module):
    def __init__(self, ch_in, ch_out, data_size, ch_in_2, ch_out_2):
        super(Model2, self).__init__()
        self.model_in = ch_in_2 # 4种模态图像
        self.model_dict = {}
        self.MultiScaleEncoder = base_encoder_2(ch_in, ch_out) # 多尺度特征提取器
        self.block2 = single_classifier(4096, 2) # 分类器
        self.block3 = SE_block(32,[4,32,192,192])
        self.stage3_fusion = nn.Sequential(
            fusion2d_block(256, 256, self.model_in),
            #                 adaptive_reduction_block(256, 256, [data_size[0], data_size[1], data_size[2] // 8, data_size[3] // 8], [data_size[2] // 32, data_size[3] // 32]),
            adaptive_reduction_block_three_branch(256, 256,
                                                  [data_size[0], data_size[1], data_size[2] // 8, data_size[3] // 8],
                                                  [data_size[2] // 32, data_size[3] // 32]),
            Multiconv_block3(256, ch_out_2),
        )
        self.stage4_fusion = nn.Sequential(
            fusion2d_block(256, 256, self.model_in),
            #                 adaptive_reduction_block(256, 256, [data_size[0], data_size[1], data_size[2] // 16, data_size[3] // 16], [data_size[2] // 32, data_size[3] // 32]),
            adaptive_reduction_block_three_branch(256, 256,
                                                  [data_size[0], data_size[1], data_size[2] // 16, data_size[3] // 16],
                                                  [data_size[2] // 32, data_size[3] // 32]),
            Multiconv_block3(256, ch_out_2),
        )
        self.stage5_fusion = nn.Sequential(
            fusion2d_block(256, 256, self.model_in),
            Multiconv_block3(256, ch_out_2),
        )
        self.z_fusion = fusion1d_block(ch_out_2, ch_out_2, self.model_in)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        # print(np.shape(x))
        # x = torch.reshape(x, shape=(4, 32, 192, 192))
        # x = self.block3(x)
        x = torch.reshape(x, shape=(32, 4, 192, 192))
        x11 = x[:, 0, :]
        x11 = torch.reshape(x11, shape=(32, 1, 192, 192))
        x22 = x[:, 1, :]
        x22 = torch.reshape(x22, shape=(32, 1, 192, 192))
        x33 = x[:, 2, :]
        x33 = torch.reshape(x33, shape=(32, 1, 192, 192))
        x44 = x[:, 3, :]
        x44 = torch.reshape(x44, shape=(32, 1, 192, 192))

        # x11 = self.dropout(x11)
        # x22 = self.dropout(x22)
        # x33 = self.dropout(x33)
        # x44 = self.dropout(x44)
        # 各模态进行多尺度特征提取,得到y_base_1=[256,1,1]和y_base_2={[1024,24,24], [1024,12,12], [1024,6,6]}
        y_base_a1, y_base_a2 = self.MultiScaleEncoder(x11)
        y_base_b1, y_base_b2 = self.MultiScaleEncoder(x22)
        y_base_c1, y_base_c2 = self.MultiScaleEncoder(x33)
        y_base_d1, y_base_d2 = self.MultiScaleEncoder(x44)


        self.model_dict['model_' + str(0)] = y_base_a1, y_base_a2
        self.model_dict['model_' + str(1)] = y_base_b1, y_base_b2
        self.model_dict['model_' + str(2)] = y_base_c1, y_base_c2
        self.model_dict['model_' + str(3)] = y_base_d1, y_base_d2

        z_local = {}
        features = {}
        for i in range(self.model_in):
            tz, tfeatures = self.model_dict['model_' + str(i)]
            z_local['z_local_' + str(i)] = tz
            features['features_' + str(i)] = tfeatures
            if i == 0:
                z = tz.unsqueeze(-1)
                stage3 = tfeatures['stage3'].unsqueeze(-1)
                stage4 = tfeatures['stage4'].unsqueeze(-1)
                stage5 = tfeatures['stage5'].unsqueeze(-1)
            else:
                z = torch.cat((z, tz.unsqueeze(-1)), -1)
                stage3 = torch.cat((stage3, tfeatures['stage3'].unsqueeze(-1)), -1)
                stage4 = torch.cat((stage4, tfeatures['stage4'].unsqueeze(-1)), -1)
                stage5 = torch.cat((stage5, tfeatures['stage5'].unsqueeze(-1)), -1)

        z3 = self.stage3_fusion(stage3)
        # z3 = self.dropout(z3)
        z3 = z3.view(z3.size(0), -1)
        z4 = self.stage4_fusion(stage4)
        # z4 = self.dropout(z4)
        z4 = z4.view(z4.size(0), -1)
        z5 = self.stage5_fusion(stage5)
        # z5 = self.dropout(z5)
        z5 = z5.view(z5.size(0), -1)
        z = self.z_fusion(z)
        # z = torch.cat((z, z3, z4, z5), 1)
        z = torch.cat((z,z3,z4,z5,y_base_a1, y_base_b1, y_base_c1, y_base_d1), 1)
        z = self.block2(z)

        return z



    def init_net(model, init_type='normal', init_gain=0.02):
        init_weights(model, init_type, gain=init_gain)


class Multi_fusion_encoder(nn.Module):
    def __init__(self, ch_in, ch_out, data_size):
        super(Multi_fusion_encoder, self).__init__()
        self.model_in = ch_in
        self.model_dict = {}
        for i in range(self.model_in):
            self.model_dict['model_' + str(i)] = base_encoder(1, ch_out)

        self.stage3_fusion = nn.Sequential(
            fusion2d_block(256, 256, self.model_in),
            #                 adaptive_reduction_block(256, 256, [data_size[0], data_size[1], data_size[2] // 8, data_size[3] // 8], [data_size[2] // 32, data_size[3] // 32]),
            adaptive_reduction_block_three_branch(256, 256,
                                                  [data_size[0], data_size[1], data_size[2] // 8, data_size[3] // 8],
                                                  [data_size[2] // 32, data_size[3] // 32]),
            Multiconv_block3(256, ch_out),
        )
        self.stage4_fusion = nn.Sequential(
            fusion2d_block(256, 256, self.model_in),
            #                 adaptive_reduction_block(256, 256, [data_size[0], data_size[1], data_size[2] // 16, data_size[3] // 16], [data_size[2] // 32, data_size[3] // 32]),
            adaptive_reduction_block_three_branch(256, 256,
                                                  [data_size[0], data_size[1], data_size[2] // 16, data_size[3] // 16],
                                                  [data_size[2] // 32, data_size[3] // 32]),
            Multiconv_block3(256, ch_out),
        )
        self.stage5_fusion = nn.Sequential(
            fusion2d_block(256, 256, self.model_in),
            Multiconv_block3(256, ch_out),
        )
        self.z_fusion = fusion1d_block(ch_out, ch_out, self.model_in)


    def forward(self, x):
        z_local = {}
        features = {}
        for i in range(self.model_in):
            tz, tfeatures = self.model_dict['model_' + str(i)](x[:, i: i + 1, :, :])
            z_local['z_local_' + str(i)] = tz
            features['features_' + str(i)] = tfeatures
            if i == 0:
                z = tz.unsqueeze(-1)
                stage3 = tfeatures['stage3'].unsqueeze(-1)
                stage4 = tfeatures['stage4'].unsqueeze(-1)
                stage5 = tfeatures['stage5'].unsqueeze(-1)
            else:
                z = torch.cat((z, tz.unsqueeze(-1)), -1)
                stage3 = torch.cat((stage3, tfeatures['stage3'].unsqueeze(-1)), -1)
                stage4 = torch.cat((stage4, tfeatures['stage4'].unsqueeze(-1)), -1)
                stage5 = torch.cat((stage5, tfeatures['stage5'].unsqueeze(-1)), -1)

        z3 = self.stage3_fusion(stage3)
        z3 = z3.view(z3.size(0), -1)
        z4 = self.stage4_fusion(stage4)
        z4 = z4.view(z4.size(0), -1)
        z5 = self.stage5_fusion(stage5)
        z5 = z5.view(z5.size(0), -1)
        z = self.z_fusion(z)
        z = torch.cat((z, z3, z4, z5), 1)
        return z, z_local, features

    def single_model_train(self):
        self.stage3_fusion.eval()
        self.stage4_fusion.eval()
        self.stage5_fusion.eval()
        self.z_fusion.eval()
        for i in range(self.model_in):
            self.model_dict.update({'model_' + str(i): self.model_dict['model_' + str(i)].train()})

    def fusion_model_train(self):
        self.stage3_fusion.train()
        self.stage4_fusion.train()
        self.stage5_fusion.train()
        self.z_fusion.train()
        for i in range(self.model_in):
            self.model_dict.update({'model_' + str(i): self.model_dict['model_' + str(i)].eval()})

    def all_model_train(self):
        self.stage3_fusion.train()
        self.stage4_fusion.train()
        self.stage5_fusion.train()
        self.z_fusion.train()
        for i in range(self.model_in):
            self.model_dict.update({'model_' + str(i): self.model_dict['model_' + str(i)].train()})

    def all_model_eval(self):
        self.stage3_fusion.eval()
        self.stage4_fusion.eval()
        self.stage5_fusion.eval()
        self.z_fusion.eval()
        for i in range(self.model_in):
            self.model_dict.update({'model_' + str(i): self.model_dict['model_' + str(i)].eval()})

    # def get_single_model_parameters(self):
    #     parameters = []
    #     for i in range(self.model_in):
    #         parameters.append({'params': self.model_dict['model_' + str(i)].parameters()})
    #     return parameters
    #
    # def get_fusion_model_parameters(self):
    #     parameters = []
    #     parameters.append({'params': self.stage3_fusion.parameters()})
    #     parameters.append({'params': self.stage4_fusion.parameters()})
    #     parameters.append({'params': self.stage5_fusion.parameters()})
    #     parameters.append({'params': self.z_fusion.parameters()})
    #     return parameters
    #
    # def get_all_model_parameters(self):
    #     parameters = []
    #     parameters.append({'params': self.stage3_fusion.parameters()})
    #     parameters.append({'params': self.stage4_fusion.parameters()})
    #     parameters.append({'params': self.stage5_fusion.parameters()})
    #     parameters.append({'params': self.z_fusion.parameters()})
    #     for i in range(self.model_in):
    #         parameters.append({'params': self.model_dict['model_' + str(i)].parameters()})
    #     return parameters
    #
    # def model_update(self, models):
    #     for i in range(self.model_in):
    #         self.model_dict.update({'model_' + str(i): models['model_' + str(i)]})
    #
    # def use_cuda(self):
    #     for i in range(self.model_in):
    #         self.model_dict.update({'model_' + str(i): self.model_dict['model_' + str(i)].cuda()})
    #     self.cuda()
    #
    # def save_model(self, path):
    #     parameters = {}
    #     parameters['stage3_fusion'] = self.stage3_fusion.state_dict()
    #     parameters['stage4_fusion'] = self.stage4_fusion.state_dict()
    #     parameters['stage5_fusion'] = self.stage5_fusion.state_dict()
    #     parameters['z_fusion'] = self.z_fusion.state_dict()
    #     for i in range(self.model_in):
    #         parameters['model_' + str(i)] = self.model_dict['model_' + str(i)].state_dict()
    #     torch.save(parameters, path)
    #
    # def load_model(self, path):
    #     print('load ', path)
    #     checkpoint = torch.load(path)
    #     self.stage3_fusion.load_state_dict(checkpoint['stage3_fusion'])
    #     self.stage4_fusion.load_state_dict(checkpoint['stage4_fusion'])
    #     self.stage5_fusion.load_state_dict(checkpoint['stage5_fusion'])
    #     self.z_fusion.load_state_dict(checkpoint['z_fusion'])
    #     for i in range(self.model_in):
    #         self.model_dict['model_' + str(i)].load_state_dict(checkpoint['model_' + str(i)])

    # 初始话模型参数
    def init_weights(self, init_type='normal', gain=0.02):
        print('init ', init_type)
        init_net(self.stage3_fusion, init_type=init_type, init_gain=gain)
        init_net(self.stage4_fusion, init_type=init_type, init_gain=gain)
        init_net(self.stage5_fusion, init_type=init_type, init_gain=gain)
        init_net(self.z_fusion, init_type=init_type, init_gain=gain)
        for i in range(self.model_in):
            init_net(self.model_dict['model_' + str(i)], init_type=init_type, init_gain=gain)


class single_classifier(nn.Module):
    def __init__(self, num_feature, num_class):
        super(single_classifier, self).__init__()
        self.logit = nn.Linear(num_feature, num_class)

    def forward(self, z):
        logit = self.logit(z)
        return logit


class combine_classifier(nn.Module):
    def __init__(self, num_feature_local=2048, num_feature_global=4096, num_class=2):
        super(combine_classifier, self).__init__()
        self.logit_global = nn.Linear(num_feature_global, num_class)
        self.logit_local = nn.Linear(num_feature_local, num_class)

    def forward(self, z_feature, z_class):
        local_logit = self.logit_local(z_class)
        global_logit = self.logit_global(torch.cat((z_feature, z_class), dim=1))
        return {'local_logit': local_logit, 'global_logit': global_logit}


def debug():
    data = torch.ones([4, 4, 192, 192, 4])
    data = data.cuda()
    #     model = Multi_fusion_encoder(4, 512, [12, 4, 192, 192])
    #     model.use_cuda()
    fusion = SE_fusion2d_block_with_pooling(4, 128, 4, [12, 4, 192, 192])
    #     fusion = SE_fusion2d_block(4, 128, 4, [12, 4, 192, 192])
    fusion.cuda()
    f = fusion(data)
    #     f, _, _ = model(data)
    print(f.size())


if __name__ == '__main__':
    debug()





