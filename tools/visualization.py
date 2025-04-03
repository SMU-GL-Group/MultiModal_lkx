# 针对K折交叉验证的可视化
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
from data_augmentation.pytorch_grad_cam.grad_cam import GradCAM#, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
import cv2
from data_augmentation.pytorch_grad_cam.utils.image import show_cam_on_image
import _pickle as cPickle
from torchvision import transforms
from PIL import Image
import os
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

save_name = '日期_数据集_模型_(备注信息)'

# ======================= 类别可视化 ======================= #
def Classes(plot_image, args, title=None, num=None, under_patient=False):
    """
    :param plot_image:是否绘出柱状图
    :param args: 相关参数
    :param title: 绘图标题
    :param num: 是否返回每个类别的数量
    :param under_patient: 是否统计病人下的每张图片
    """
    # 遍历文件夹，一个文件夹对应一个类别
    # every_class = [cla for cla in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, cla))]
    # 排序，保证顺序一致
    # every_class.sort()
    every_class = args.cls_list

    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".tif"]  # 支持的文件后缀类型

    # 遍历每个文件夹下的文件
    for cla in every_class:
        cla_path = os.path.join(args.data_path, cla)
        if not under_patient:
            # 遍历所有文件路径
            images = [os.path.join(args.data_path, cla, i) for i in os.listdir(cla_path)]
            # 记录该类别的样本数量
            every_class_num.append(len(images))
        else:
            pat_img_num = [] # 存储每个病人的样本总数
            every_patient = [pat for pat in os.listdir(cla_path) if os.path.isdir(cla_path)]
            for pat in every_patient:
                pat_path = os.path.join(args.data_path, cla, pat)
                images = [os.path.join(args.data_path, cla, pat, i) for i in os.listdir(pat_path)
                          if os.path.splitext(i)[-1] in supported]
                pat_img_num.append(len(images))
            every_class_num.append(sum(pat_img_num))

    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(every_class)), every_class_num, align='center')
        # 将横坐标数字替换为相应的类别名称
        plt.xticks(range(len(every_class)), every_class,fontsize=15)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 0.7, s=str(v), ha='center', fontsize=15)
        # 设置x坐标
        plt.xlabel('类别',fontsize=15)
        # 设置y坐标
        plt.ylabel('数量',fontsize=15)
        # 设置柱状图的标题
        plt.title(title,fontsize=18)
        plt.tight_layout()
        plt.show()

    if num:
        return every_class_num


# ======================= 损失和准确率可视化 ======================= #
def Loss_Acc(epochs=None, savename=None, train_loss=None, test_loss=None,train_acc=None, test_acc=None, k=''):
    # =======acc=======
    if train_acc:
        plt.plot(range(1, epochs+1), train_acc, label='Train accuracy')
        plt.text(epochs+0.02, train_acc[-1]-0.01, round(train_acc[-1], 3), ha='right', va='top', color='#0000ff', fontsize=15)  # 默认蓝
    if test_acc:
        plt.plot(range(1, epochs+1), test_acc, label='Test accuracy')
        plt.text(epochs+0.02, test_acc[-1]-0.01, round(test_acc[-1], 3), ha='right', va='top', color='#ff2200', fontsize=15)  # 默认橙
    plt.legend(loc='best')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title(savename)
    plt.tight_layout()
    if savename is not None:
        plt.savefig(f'{os.getcwd()}/结果图/{savename}_k={k}_Acc.png')
    plt.show()

    # =======loss=======
    if train_loss:
        plt.plot(range(1, epochs+1), train_loss, label='Training loss')
    if test_loss:
        plt.plot(range(1, epochs+1), test_loss, label='Test loss')

    plt.legend(loc='best')  # 图例位置
    plt.ylabel('Cross entropy')
    plt.xlabel('Epoch')
    plt.title('{}'.format(savename))
    plt.tight_layout()
    if savename is not None:
        plt.savefig(f'{os.getcwd()}/结果图/{savename}_k={k}_Loss.png')
    plt.show()


# ======================= ROC曲线、AUC可视化 ======================= #
def plot_AUC(y_scores, y_test, n_classes, cls, k='', savename=None):
    # AUC of each classes
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    spe = dict() # 特异性specificity
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        # 计算特异性
        y_pred = (y_scores[:, i] >= 0.5).astype(int)  # 以0.5为阈值进行二值化
        cm = confusion_matrix(y_test[:, i], y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        spe[i] = specificity

        print(f'AUC {cls[i]}: {roc_auc[i]:.4f}, SPE {cls[i]}: {specificity:.4f}')

    # AUC of micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print('AUC micro-average: {:.4f}'.format(roc_auc["micro"]))

    # 计算宏平均特异性
    spe["macro"] = sum(spe.values()) / int(n_classes) # 字典转列表再求和取均值
    print('SPE macro-average: {:.4f}'.format(spe["macro"]))

    # AUC of macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    print('AUC macro-average: {:.4f}'.format(roc_auc["macro"]))

    # 绘制ROC
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot([0, 1])
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC class {} (AUC = {:.4})'.format(cls[i], roc_auc[i]), linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.title('{}  ROC'.format(savename))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc='best')
    plt.tight_layout()
    if savename is not None:
        plt.savefig(f'{os.getcwd()}/结果图/{savename}_k={k}_AUC.png')
    plt.show()

    return roc_auc, spe

# ======================= 特征图可视化 ======================= #
def feature_map(model, image, image_path, reshape, save_name, device):
    model_weights = []
    conv_layers = []
    model_children = list(model.children())  # counter to keep count of the conv layers
    counter = 0  # append all the conv layers and their respective weights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for child in model_children[i].children():
                if type(child) == nn.Conv2d:
                    counter += 1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
    # print(len(conv_layers))

    outputs = []
    names = []

    with torch.no_grad():
        # for image, label in train_loader:  # 迭代全部数据太吃内存了
        image0 = Image.open(image_path)
        trans = transforms.Resize(reshape)  # todo
        image_orig = trans(image0)
        # image_orig = image_orig[0, :, :, :]
        # image_orig = np.transpose(image_orig, (1, 2, 0))
        plt.figure()
        plt.imshow(image_orig, cmap='gray')
        plt.title('original image')
        plt.show()

        image = image.to(device)
        for layer in conv_layers:
            image = layer(image)
            outputs.append(image)
            names.append(str(layer))
    #print(len(outputs))  # print feature_maps
    # for feature_map in outputs:
    #     print(feature_map.shape)

    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map, 0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())
    #print(len(processed))

    # for fm in processed:
    #     print(fm.shape)

    fig = plt.figure()
    for i in range(len(processed)):
        # a = fig.add_subplot(4, 8, i+1)
        a = plt.subplot(1, 2, 1)
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        # for j in range(8):  # len(processed[i])):
        img_show = torch.from_numpy(processed[i])  # 把(64, 32, 32)变为(32, 32)以便imshow
        img_show = img_show[0, :, :]
        plt.imshow(img_show)
        a.set_title(names[i].split('(')[0], fontsize=30)
        b = plt.subplot(1, 2, 2)
        plt.imshow(image_orig)
        plt.title('original image')
        # b.set_title('original image')
        # plt.imshow(image_orig[:, :, 0])
        # a.axis("off")
    if save_name is not None:
        plt.savefig(str('./结果图/{}_feature_maps.jpg'.format(save_name)), bbox_inches='tight')
    plt.show()


# ======================= CAM梯度图可视化 ======================= #
def CAM(model, args, target_layers,k=''):
    model = model.to(args.device)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(args.reshape,antialias=True),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    #target_layers = [model.layer4[-1]]# 需要用CAM展示的特征层 （Resnet18 and 50）#todo
    #target_layers = [model.multimodal_transformer] # mmFormer,但是缺mask这个参数....
    #[model.features[-1]] # VGG and densenet161
    # model.blocks[-1].norm1 # ViT
    image = Image.open(args.CAM_img_path).convert('RGB')
    img = np.array(image, dtype=np.uint8)
    img_tensor = transform(img)
    input_tensor = torch.unsqueeze(img_tensor, dim=0).requires_grad_(True)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]

    # 在原图上叠加CAM
    # 读取文件中指定图片展示CAM
    trans = transforms.Resize(args.reshape,antialias=True)  # todo
    image_orig = trans(image)

    image_cam_show = image_orig
    image_cam_show = np.array(image_cam_show, dtype=np.uint8)
    visualization = show_cam_on_image(image_cam_show.astype(dtype=np.float32) / 255.,
                                      grayscale_cam, use_rgb=True)

    # 把CAM和原图的灰度图放在同一个框内展示
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.tight_layout()
    plt.imshow(image_orig)
    plt.title('original image')
    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title('plus CAM')
    if args.savename is not None:
        cur_path = os.getcwd()
        plt.savefig('{}/结果图/{}_CAM.png'.format(cur_path, args.savename+'_k=' + str(k)))
    plt.show()


# ======================= 多模态CAM梯度图可视化 ======================= #
def mmCAM(model, patient_path, reshape, savename, target_layers, modals,k=''):
    transform = transforms.Compose([transforms.Resize(reshape, antialias=True),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])

    for image_name in os.listdir(patient_path):
        if image_name.startswith('0') or image_name.startswith('test'):  # 电镜
            image0_path = os.path.join(patient_path, image_name)
        else:
            image1_path = os.path.join(patient_path, image_name)
    img0 = Image.open(image0_path).convert('RGB')  # 打开图片
    img1 = Image.open(image1_path).convert('RGB')
    img0_tensor = transform(img0)
    img1_tensor = transform(img1)
    img_cat = torch.cat((img0_tensor, img1_tensor), dim=0)
    # 增加一个batch_size=1的维度便于送入模型
    input_tensor = torch.unsqueeze(img_cat, dim=0).requires_grad_(True)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]

    # 在原图上叠加CAM
    if modals == 'TEM(含标尺)':
        image = Image.open(image0_path).convert('RGB') # 要展示的目标图像
    elif modals == 'IF':
        image = Image.open(image1_path).convert('RGB')
    trans = transforms.Resize(reshape,antialias=True)
    image_orig = trans(image)
    image_cam_show = np.array(image_orig, dtype=np.uint8)
    visualization = show_cam_on_image(image_cam_show.astype(dtype=np.float32) / 255., grayscale_cam, use_rgb=True)

    # 把CAM和原图放在同一个框内展示
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.tight_layout()
    plt.imshow(image_orig)
    plt.title('original image')
    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title('plus CAM')
    if savename is not None:
        cur_path = os.getcwd()
        plt.savefig('{}/结果图/{}_CAM.png'.format(cur_path, savename+'_k=' + str(k)))
    plt.show()
