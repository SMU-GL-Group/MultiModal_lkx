import os
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
from tqdm import tqdm

def PixelTrans(img, PixVal):
    """
    调整特定像素值的像素透明度。
    """
    # 将图像转换为RGBA模式
    img.convert('RGBA')
    pixels = img.getdata()

    # 创建一个新的图像数据列表
    new_pixels = []
    for pixel in pixels:
        # 如果像素是黑色的，则设置Alpha通道为0（透明）
        if pixel[0] <= PixVal[0] and pixel[1] <= PixVal[1] and pixel[2] <= PixVal[2]:
            new_pixels.append((0, 0, 0, 0))
        else:
            new_pixels.append(pixel)

    # 将新的像素数据赋值给图像
    img.putdata(new_pixels)

    return img


#save_path = '/public/longkaixing/CrossModalScale/tools/特征降维结果/08-06_光镜（PASM）+荧光+电镜-301+270例_CrossMIL_TEMq_OMkv_IMkv_3Loss_240806(电镜q+光镜kv+荧光kv,wLoss)_TEMencoder_TSNE_features.npy'
save_path = '/public/longkaixing/CrossModalScale/tools/特征降维结果/4-22_光镜（PASM）+荧光+电镜-301+270例_CrossMIL(光镜+荧光+电镜,cat)_TEMencoder_TSNE_features.npy'
#save_path = '/public/longkaixing/CrossModalScale/tools/特征降维结果/08-06_光镜（PASM）+荧光+电镜-301+270例_CrossMIL_TEMq_OMkv_IMkv_3Loss_240806(电镜q+光镜kv+荧光kv,wLoss)_OMencoder_TSNE_features_per30.npy'
pca_features = np.load(save_path)  # [样本数,对应的xy坐标]
# 使用numpy格式的图片
data_path = '/public/longkaixing/CrossModalScale/tools/特征降维结果/4-22_光镜（PASM）+荧光+电镜-301+270例_CrossMIL(光镜+荧光+电镜,cat)_OMencoder_TSNE_DataIndex_OM-ori.npy'
data_index = torch.load(data_path)
data_path = data_index[
                       #'OM_path_lst'
                       'bag_lst'
                      ]
img_classes = []  # 用于存储不同类别的图像路径列表
t = transforms.Resize([224,224])
cls_color = {'MN':'red', 'IgAN':'yellow', 'LN':'blue'}
for i, (feature, img_path) in enumerate(tqdm(zip(pca_features, data_path))):
    # 电镜固定第一张
    img_path = img_path[0]

    cls = (img_path.split('/')[-1]).split('_')[1]
    img = Image.open(img_path)
    # 将黑色像素透明化
    # img = PixelTrans(img, PixVal=[40,40,40])
    # 绿色像素增强？让绿色更亮？
    # plt.imshow(img)
    # plt.show()
    img = t(img)
    img = ImageOps.expand(img, border=10, fill=cls_color[cls])
    x, y = feature
    # 将图像转换为NumPy数组
    image_array = np.array(img)
    # 将数组添加到对应类别的列表中
    img_classes.append([image_array,(x,y)])  # 将图像npy和坐标绑定

# 创建一个图形和轴
fig = plt.figure(facecolor='#ffffff')
ax = fig.add_subplot(111)
b = 120# 横纵轴偏移量

# 遍历每个点，并在散点图上显示对应的图像
random.seed(42)
random.shuffle(img_classes)  # 打乱类别顺序，避免某一类图像都在上面
for i, img_feature in enumerate(tqdm(img_classes)):
    img, feature = img_feature
    x, y = feature
    x = x*80 # 拉开坐标差距，增大图片间距离
    y = y*80
    # 显示图像
    im = ax.imshow(img, extent=(x-112, x+112, y-112, y+112))  # extent决定图像四个角的位置

# 设置轴的范围
ax.set_xlim(pca_features[:, 0].min()*80-b, pca_features[:, 0].max()*80+b)
ax.set_ylim(pca_features[:, 1].min()*80-b, pca_features[:, 1].max()*80+b)

# 隐藏坐标轴及刻度
plt.tick_params(axis='both', which='both', length=0, labelleft=False, labelbottom=False)
plt.axis('off')
# 显示图形
plt.tight_layout()
plt.show(dpi=900)
print()
