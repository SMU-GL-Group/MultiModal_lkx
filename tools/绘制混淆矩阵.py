import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# plt.rcParams['font.family'] = ['sans-serif']
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Times New Roman'

"MMGL"
# CM = [[0.3052, 0.5732, 0.1216],
#       [0.2506, 0.5966, 0.1528],
#       [0.2922, 0.5820, 0.1258]]

"mmFormer"
# CM = [[0.7960, 0.0680, 0.1360],
#       [0.0836, 0.8824, 0.0340],
#       [0.1390, 0.0434, 0.8176]]

"MDL-IIA"
# CM = [[0.7960, 0.0500, 0.1540],
#       [0.0114, 0.8782, 0.1104],
#       [0.0810, 0.0220, 0.8968]]

"MSAN"
# CM = [[0.9278, 0.0046, 0.0676],
#       [0.0268, 0.9388, 0.0344],
#       [0.0710, 0.0262, 0.9028]]

"AGGN"
# CM = [[0.3230, 0.5358, 0.1410],
#       [0.0064, 0.9936, 0.0000],
#       [0.1056, 0.3316, 0.5628]]

"Ours"
# CM = [[0.9712, 0.0046, 0.0242],
#       [0.0048, 0.9788, 0.0162],
#       [0.0444, 0.0388, 0.9168]]

"IM&TEM-USA"
# CM = [[0.7232, 0.2426, 0.0342],
#       [0.0746, 0.8580, 0.0670],
#       [0.0370, 0.3574, 0.6054]]

"OM&IM&TEM-USA"
CM = [[0.6206, 0.3212, 0.0582],
      [0.0582, 0.8576, 0.0842],
      [0.0504, 0.3516, 0.5980]]

"No Attention"
# CM = [[0.9166, 0.0202, 0.0636],
#       [0.0100, 0.9528, 0.0370],
#       [0.0654, 0.0324, 0.9020]]

"Self-Attention"
# CM = [[0.8866, 0.0114, 0.1020],
#       [0.0150, 0.9458, 0.0392],
#       [0.0808, 0.0270, 0.8920]]

"Modality-based Attention"
# CM = [[0.9124, 0.0164, 0.0710],
#       [0.0104, 0.9678, 0.0218],
#       [0.0710, 0.0334, 0.8956]]

"Cross-modal Attention"
# CM = [[0.9104, 0.0212, 0.0684],
#       [0.0206, 0.9356, 0.0440],
#       [0.0768, 0.0388, 0.8844]]

"混淆矩阵样例"
# CM = [[1.0000, 0.0000, 0.0000],
#       [0.2500, 0.5000, 0.2500],
#       [0.0500, 0.2000, 0.7500]]

"VGG16"
# CM = [[1.0000, 0.0000, 0.0000],
#       [0.2500, 0.5000, 0.2500],
#       [0.0500, 0.2000, 0.7500]]


fs = 23 # 字体大小
plt.figure(figsize=(8, 6))


sns_plot = sns.heatmap(CM, annot=True, linewidths=0.1, fmt='.4f', annot_kws={'size':30}, cmap='Blues', vmin=0, vmax=1,
                       xticklabels=['stage Ⅰ', 'stage Ⅱ','stage Ⅲ'], yticklabels=['stage Ⅰ', 'stage Ⅱ','stage Ⅲ'])
                       # xticklabels=['MN', 'IgAN','LN'], yticklabels=['MN', 'IgAN','LN'])
                       #xticklabels=['无', '有'], yticklabels=['无', '有'])
cbar = sns_plot.collections[0].colorbar
cbar.ax.tick_params(labelsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)

# plt.xlabel('模型预测', fontsize=23)
# plt.ylabel('真实标签', fontsize=23)
# plt.xlabel('mmFormer Predictions', fontsize=fs)
plt.xlabel('OM&IM&TEM Predictions', fontsize=fs)
plt.ylabel('Ground Truth', fontsize=fs)

plt.tight_layout()
plt.show()


# 根据混淆矩阵计算宏平均特异性specificity
def calculate_specificity(cm):
    cm = np.array(cm)
    num_classes = cm.shape[0]
    specificities = []

    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + tp
        fn = np.sum(cm[i, :]) - tp

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity)

    # 计算宏平均特异性
    macro_specificity = np.mean(specificities)

    return macro_specificity, specificities

# 计算宏平均特异性
macro_specificity, specificities = calculate_specificity(CM)

print(f"宏平均特异性: {macro_specificity:.4f}")
print(f"每个类别的特异性: {specificities}")
