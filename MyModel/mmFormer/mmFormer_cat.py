#coding=utf-8
"""
将mmFormer [1] 最后的模态相关模块(模态间的Transformer,multimodal_transformer)改成直接cat的形式。

[1] mmFormer: Multimodal Medical Transformer for Incomplete Multimodal Learning of Brain Tumor Segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import general_conv2d_prenorm

basic_dims = 8
transformer_basic_dims = 512
mlp_dim = 4096
num_heads = 8
depth = 1
patch_size = 28  # todo


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.e1_c1 = nn.Conv2d(in_channels=3, out_channels=basic_dims, kernel_size=3, stride=1, padding=1,
                               padding_mode='reflect', bias=True)  # padding_mode='reflect' 镜像填充
        self.e1_c2 = general_conv2d_prenorm(basic_dims, basic_dims, pad_type='reflect')  # 预规范化+2D卷积
        self.e1_c3 = general_conv2d_prenorm(basic_dims, basic_dims, pad_type='reflect')

        self.e2_c1 = general_conv2d_prenorm(basic_dims, basic_dims * 2, stride=2,
                                            pad_type='reflect')  # 步长为2，对特征图进行下采样，下同。
        self.e2_c2 = general_conv2d_prenorm(basic_dims * 2, basic_dims * 2, pad_type='reflect')
        self.e2_c3 = general_conv2d_prenorm(basic_dims * 2, basic_dims * 2, pad_type='reflect')

        self.e3_c1 = general_conv2d_prenorm(basic_dims * 2, basic_dims * 4, stride=2, pad_type='reflect')
        self.e3_c2 = general_conv2d_prenorm(basic_dims * 4, basic_dims * 4, pad_type='reflect')
        self.e3_c3 = general_conv2d_prenorm(basic_dims * 4, basic_dims * 4, pad_type='reflect')

        self.e4_c1 = general_conv2d_prenorm(basic_dims * 4, basic_dims * 8, stride=2, pad_type='reflect')
        self.e4_c2 = general_conv2d_prenorm(basic_dims * 8, basic_dims * 8, pad_type='reflect')
        self.e4_c3 = general_conv2d_prenorm(basic_dims * 8, basic_dims * 8, pad_type='reflect')

        self.e5_c1 = general_conv2d_prenorm(basic_dims * 8, basic_dims * 16, stride=2, pad_type='reflect')
        self.e5_c2 = general_conv2d_prenorm(basic_dims * 16, basic_dims * 16, pad_type='reflect')
        self.e5_c3 = general_conv2d_prenorm(basic_dims * 16, basic_dims * 16, pad_type='reflect')

    def forward(self, x):  # Linear Proj. 线性投影，一张图像编码成五个特征。五阶段编码器，每个阶段包含两个卷积块
        x1 = self.e1_c1(x)  # 卷积块1(仅包含卷积层)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))  # 卷积块2

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        x5 = self.e5_c1(x4)
        x5 = x5 + self.e5_c3(self.e5_c2(x5))

        return x1, x2, x3, x4, x5


class MLP(nn.Module):
    def __init__(self, num_i, num_h, num_o):
        super(MLP, self).__init__()

        self.linear1 = torch.nn.Linear(num_i, num_h)  # in_feature, out_feature
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(num_h, num_h)  # 2个隐层
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(num_h, num_o)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.drop(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.drop(x)

        return x


class SelfAttention(nn.Module):
    def __init__(
            self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x)


class FeedForward(nn.Module):  # 两层(线性层)感知器
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, embedding_dim, depth, heads, mlp_dim, dropout_rate=0.1):
        super(Transformer, self).__init__()
        self.cross_attention_list = []
        self.cross_ffn_list = []
        self.depth = depth
        for j in range(self.depth):
            self.cross_attention_list.append(
                Residual(
                    PreNormDrop(
                        embedding_dim,
                        dropout_rate,
                        SelfAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate),
                    )
                )
            )
            self.cross_ffn_list.append(
                Residual(
                    PreNorm(embedding_dim, FeedForward(embedding_dim, mlp_dim, dropout_rate))
                )
            )

        self.cross_attention_list = nn.ModuleList(self.cross_attention_list)
        self.cross_ffn_list = nn.ModuleList(self.cross_ffn_list)

    def forward(self, x, pos):  # 位置编码。F`local_m Wm +Pm
        for j in range(self.depth):
            x = x + pos
            x = self.cross_attention_list[j](x)  # MSA Multi-head Self Attention (含LN:Layer Norm操作，计算Q、K、V)
            x = self.cross_ffn_list[j](x)  # FFN Feed-Forward Network 两层(线性层)感知器
        return x


class MaskModal(nn.Module):  # 模态掩码，造成模态缺失，迫使模型学习
    def __init__(self):
        super(MaskModal, self).__init__()

    def forward(self, x, mask):
        B, K, C, H, W = x.size()  # Batch_size, K=模态总数, Channels, Height, Weight
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        x = y.view(B, -1, H, W)
        return x


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.num_modals = len(args.modal.split('+'))  # 模态数量
        self.OM_encoder = Encoder()
        self.IM_encoder = Encoder()
        self.TEM_encoder = Encoder()

        ########### IntraFormer # Intra-modal Transformer 模态内Transformer
        # in_channels=8*16,out_channels=512
        self.OM_encode_conv = nn.Conv2d(basic_dims * 16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.IM_encode_conv = nn.Conv2d(basic_dims * 16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.TEM_encode_conv = nn.Conv2d(basic_dims * 16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)

        self.OM_decode_conv = nn.Conv2d(transformer_basic_dims, basic_dims * 16, kernel_size=1, stride=1, padding=0)
        self.IM_decode_conv = nn.Conv2d(transformer_basic_dims, basic_dims * 16, kernel_size=1, stride=1, padding=0)
        self.TEM_decode_conv = nn.Conv2d(transformer_basic_dims, basic_dims * 16, kernel_size=1, stride=1, padding=0)
        # 位置编码，创建torch.size([1, patch_size**2])的FloatTensor放入(模型的)Parameter列表中
        # 3D数据，所以是**3，2D数据，所以是**2
        self.OM_pos = nn.Parameter(torch.zeros(1, patch_size ** 2, transformer_basic_dims))
        self.IM_pos = nn.Parameter(torch.zeros(1, patch_size ** 2, transformer_basic_dims))
        self.TEM_pos = nn.Parameter(torch.zeros(1, patch_size ** 2, transformer_basic_dims))

        self.OM_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads,
                                             mlp_dim=mlp_dim)
        self.IM_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads,
                                            mlp_dim=mlp_dim)
        self.TEM_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads,
                                            mlp_dim=mlp_dim)

        ########### InterFormer
        self.masker = MaskModal()

        self.MLP = MLP(num_i=transformer_basic_dims * self.num_modals, num_h=128, num_o=args.num_cls)  # todo
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化至尺寸为(1,1)，通道数不变

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)  #

    def forward(self, OM_tensor, IM_tensor, TEM_tensor, mask):
        # extract feature from different layers
        # 一个模态的图分成五块 # todo 不用4层/模态，改2层/模态
        OM_x1, OM_x2, OM_x3, OM_x4, OM_x5 = self.OM_encoder(OM_tensor)
        IM_x1, IM_x2, IM_x3, IM_x4, IM_x5 = self.IM_encoder(IM_tensor)
        TEM_x1, TEM_x2, TEM_x3, TEM_x4, TEM_x5 = self.TEM_encoder(TEM_tensor)

        ########### IntraFormer # Intra-modal Transformer/模态内Transformer 的 tokenizer标记化器
        # the local feature maps F`local _m produced by the convolutional encoder is first flattened into a 1D sequence and transformed into token space by a linear projection.
        # self=(1,512,28,28),permute=(1,28,28,512),view(x.size(0),-1, trans)=(1,784=28*28,512)# 维度0整形为x.size(0)=1,维度3整形为trans=512,中间的自动整形分配（参数-1）
        OM_token_x5 = self.OM_encode_conv(OM_x5).permute(0, 2, 3, 1).contiguous().view(OM_tensor.size(0), -1,transformer_basic_dims)  # in_channels=8*16,out_channels=512, .permute重排维度, .contiguous深拷贝
        IM_token_x5 = self.IM_encode_conv(IM_x5).permute(0, 2, 3, 1).contiguous().view(IM_tensor.size(0), -1,transformer_basic_dims)  # 转成[Bsize,H,W,C]
        TEM_token_x5 = self.TEM_encode_conv(TEM_x5).permute(0, 2, 3, 1).contiguous().view(TEM_tensor.size(0), -1,transformer_basic_dims)

        # 位置编码 OM_token_x5(1,784,512), OM_pos(1,patch_size**2,512)
        OM_intra_token_x5 = self.OM_transformer(OM_token_x5, self.OM_pos)
        IM_intra_token_x5 = self.IM_transformer(IM_token_x5, self.IM_pos)
        TEM_intra_token_x5 = self.TEM_transformer(TEM_token_x5, self.TEM_pos)

        # 复原回(Bsize,H,W,C)
        OM_intra_x5 = OM_intra_token_x5.view(OM_tensor.size(0), patch_size, patch_size,
                                             transformer_basic_dims).permute(0,1,2,3).contiguous()
        IM_intra_x5 = IM_intra_token_x5.view(IM_tensor.size(0), patch_size, patch_size,
                                             transformer_basic_dims).permute(0,1,2,3).contiguous()
        TEM_intra_x5 = TEM_intra_token_x5.view(TEM_tensor.size(0), patch_size, patch_size,
                                             transformer_basic_dims).permute(0,1,2,3).contiguous()

        ########### IntraFormer # todo mask是解决模态缺失的，先不使用mask
        # (1,56,28,512) # x5_intra是F~global了，avgpool后拿去MLP分类
        # x5_intra = self.masker(torch.stack((OM_intra_x5, IM_intra_x5, TEM_intra_x5), dim=1), mask)
        x5_intra = torch.stack((OM_intra_x5, IM_intra_x5, TEM_intra_x5), dim=1)
        ########### InterFormer
        # chunk,将x5_intra按dim=1切割成num_modals个tensor块
        chunks = torch.chunk(x5_intra, self.num_modals, dim=1)
        OM_intra_x5, IM_intra_x5, TEM_intra_x5 = [chunk.squeeze(1) for chunk in chunks]
        # 将所有模态拼接起来
        multimodal_token_x5 = torch.cat(
            [OM_intra_x5.permute(0, 2, 3, 1).contiguous().view(OM_tensor.size(0), -1, transformer_basic_dims),
             IM_intra_x5.permute(0, 2, 3, 1).contiguous().view(IM_tensor.size(0), -1, transformer_basic_dims),
             TEM_intra_x5.permute(0, 2, 3, 1).contiguous().view(TEM_tensor.size(0), -1, transformer_basic_dims),
             # (Bsize,28*28=784,trans=512)
             ], dim=1)  # 模态拼接后=(1,784*num_modals,512)
        # 拼接位置编码=(1,784num_modals,512) # _pos一开始全零是正常
        multimodal_pos = torch.cat([self.OM_pos, self.IM_pos, self.TEM_pos], dim=1)

        multimodal_inter_token_x5 = multimodal_token_x5 + multimodal_pos  # 拼接后的模态叠加上位置编码 # todo
        # (1,1024,28,28)
        x = multimodal_inter_token_x5.view(multimodal_inter_token_x5.size(0), patch_size, patch_size,
                                           transformer_basic_dims * self.num_modals).permute(0, 3, 1,2).contiguous()
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # ======================= MLP ======================= #
        output = self.MLP(x)

        return output
