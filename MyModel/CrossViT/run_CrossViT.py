#coding=utf-8
import os.path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import numpy as np
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from tools import visualization as vis
import seaborn as sns
import logging
from MMdataset import MMdata_TEMrandom
from tools.result_save import save_npy
from torch.utils.tensorboard import SummaryWriter
from utils.parser import setup
import warnings
import xlwt

warnings.filterwarnings("ignore")

# ======================= 定义训练函数 ======================= #
def train_one_epoch(loader, model, args, criterion, optimizer):
    total_sample_train = 0
    right_sample_train = 0
    train_loss = 0.0

    model.train()
    for idx, data in enumerate(loader):
        x, target = data
        x = x.to(device=args.device)
        target = target.to(device=args.device)

        output = model(x)

        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * target.size(0)
        # 将输出概率转换为预测类
        _, pred = torch.max(output, 1)
        # 将预测与真实标签进行比较
        correct_tensor_train = pred.eq(target.data.view_as(pred))
        total_sample_train += target.size(0)
        for right in correct_tensor_train:
            if right:
                right_sample_train += 1

    Train_acc = 100 * right_sample_train / total_sample_train

    Train_loss = train_loss / len(loader.sampler)

    print('\nTrain Acc: {:.4f}%, Train Loss: {:.4f}'.format(Train_acc, Train_loss))

    return Train_acc, Train_loss

# ======================= 定义测试函数 ======================= #
def Test(loader, model, args, criterion):
    total_sample_test = 0
    right_sample_test = 0
    test_loss = 0.0
    score_list, label_list = [], []
    preds, actuals = [], []
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(loader):
            x, target = data
            x = x.to(device=args.device)
            target = target.to(device=args.device)

            output = model(x)
            loss = criterion(output, target)

            test_loss += loss.item() * target.size(0)
            _, pred = torch.max(output, 1)

            preds.extend(pred.cpu().numpy())
            actuals.extend(target.cpu().numpy())

            # 用于计算AUC
            score_temp = output
            score_list.extend(score_temp.detach().cpu().numpy())
            label_list.extend(target.cpu().numpy())

            correct_tensor = pred.eq(target.data.view_as(pred))
            total_sample_test += target.size(0)
            for right in correct_tensor:
                if right:
                    right_sample_test += 1

        Test_acc = 100 * right_sample_test / total_sample_test

        score_array = np.array(score_list)
        # 将label转换成onehot形式
        label_tensor = torch.tensor(label_list)
        label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
        label_onehot = torch.zeros(label_tensor.shape[0], args.num_cls)
        label_onehot.scatter_(dim=1, index=label_tensor, value=1)
        label_onehot = np.array(label_onehot)

        # 计算并显示测试集的损失函数
        Test_loss = test_loss / len(loader.sampler)
        print("Test Acc: {:.4f}%, Test Loss: {:.4f}".format(Test_acc, Test_loss))

        return Test_acc, Test_loss, actuals, preds, score_array, label_onehot


# ======================= 参数设置 ======================= #
parser = argparse.ArgumentParser()
from CrossViT import crossvit_base_224
parser.add_argument('--Description',
                    default='run_CrossViT。使用CrossViT-Base模型进行单模态分类。')
parser.add_argument('--savename', default='11-30_光镜（PASM）+荧光+电镜-301+270例_CrossViT-B(电镜)',
                    help='命名格式：XX-XX（日期：月-日）_数据集名称_模型名称(模态名称,及备注信息)')

# *********************** 数据集相关 *********************** #
parser.add_argument('--data_path', default='/public/longkaixing/MMF/datasets/光镜（PASM）+荧光+电镜-301+270例', type=str)
parser.add_argument('--num_epochs', default=40, type=int)
parser.add_argument('--cls_list', default=['MN', 'IgAN','LN'], type=list, help='类别名称及其对应于label的顺序')
parser.add_argument('--sample_index', type=str,
                        default='/public/longkaixing/MMF/MyMethod/数据集交叉划分index/光镜（PASM）+荧光+电镜-301+270例',
                        help="五折交叉验证划分数据集的索引路径")


parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--modal', default='TEM', type=str,
                    help='所用模态：IM+TEM / OM_+TEM / OM_+IM, OM_ 是分割后的光镜，OM-ori 是没分割的光镜')
parser.add_argument('--lr', default=5e-5, type=float)
parser.add_argument('--num_cls', default=3, type=int, help='分为几类')
parser.add_argument('--device', default='cuda:1', type=str)
parser.add_argument('--kfold', default=5, type=int)
parser.add_argument('--weight_decay', default=5e-5, type=float)
parser.add_argument('--reshape', default=[299, 299])

criterion = nn.CrossEntropyLoss()

args = parser.parse_args()
if args.savename is not None:
    args.savepath = '/public/longkaixing/CrossModalScale/model_save/{}'.format(args.savename)
    setup(args, 'training')
    os.makedirs(args.savepath, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.savepath, 'summary'))
args.transforms = transforms.Compose([
    transforms.Resize(args.reshape, antialias=True),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def main():
    logging.info(str(args))  # 打印参数信息
    # ======================= 载入数据集 ======================= #
    # vis.Classes(plot_image=True, args=args, under_patient=False, title='病人数量')
    data_set = MMdata_TEMrandom(transform=args.transforms, root=args.data_path, modal=args.modal)
    print('数据集：', args.data_path)

    # ======================= K折交叉验证 ======================= #
    k_train_acc_lst, k_train_loss_lst = [], []  # K次平均结果
    k_test_acc_lst, k_test_loss_lst = [], []
    train_loader_lst, test_loader_lst = [], []

    for k in range(0, args.kfold):
        # 载入预先划分的数据集index
        train_index = np.load('{}_train_index_k={}.npy'.format(args.sample_index, k + 1))
        test_index = np.load('{}_test_index_k={}.npy'.format(args.sample_index, k + 1))
        train_loader = DataLoader(data_set, batch_size=args.batch_size, sampler=train_index,num_workers=0,
                                  drop_last=True,pin_memory=False)
        test_loader = DataLoader(data_set, batch_size=args.batch_size, sampler=test_index,num_workers=0,
                                 drop_last=True,pin_memory=False)
        train_loader_lst.append(train_loader)
        test_loader_lst.append(test_loader)

    wb = xlwt.Workbook() # 用于记录全部折次数据
    # ======================= 模型训练/测试 ======================= #
    for i in range(0, args.kfold): # 从第i+1折开始

        # ======================= 模型设置 ======================= #
        model = crossvit_base_224(pretrained=True)
        model.head = nn.ModuleList([nn.Linear(384, args.num_cls), nn.Linear(768,args.num_cls)])
        model = model.to(args.device)
        model = nn.DataParallel(model, device_ids=[1])
        # if i == 0:
        #     print(model)
        for param in model.parameters():
            param.requires_grad = True  # 冻结所有参数时为False

        ##########Setting learning schedule and optimizer
        train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}]
        optimizer = torch.optim.Adam(train_params, betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
        lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs,
                                                                     eta_min=0.000005)  # 余弦退火调整学习率
        print('# ======================= K折交叉验证：{}/{} ======================= #'.format(i + 1, args.kfold))
        train_loader = train_loader_lst[i]
        test_loader = test_loader_lst[i]
        # 用于记录每轮结果
        train_acc_lst, train_loss_lst = [], []
        test_acc_lst, test_loss_lst = [], []

        for epoch in tqdm(range(1, args.num_epochs + 1)):
            Train_Acc, Train_Loss = train_one_epoch(train_loader, model, args, criterion, optimizer)
            Test_Acc, Test_Loss, Actuals, Preds, Score_array, Label_onehot = Test(test_loader, model, args, criterion)
            train_acc_lst.append(Train_Acc)
            train_loss_lst.append(Train_Loss)
            test_acc_lst.append(Test_Acc)
            test_loss_lst.append(Test_Loss)
            lr_schedule.step()  # 更新学习率
        # 记录每折训练测试曲线
        k_train_acc_lst.append(train_acc_lst)
        k_train_loss_lst.append(train_loss_lst)
        k_test_acc_lst.append(test_acc_lst)
        k_test_loss_lst.append(test_loss_lst)

        # ======================= 保存模型(每折) ======================= #
        if args.savename is not None:
            file_name = os.path.join(args.savepath, 'model_last_k={}.pth'.format(i + 1))
            torch.save({
                'epoch': epoch,
                #'state_dict': model.state_dict(),
                'state_dict': model.module.state_dict(),  # todo 用到nn.DataParallel时保存模型的方式
                'optim_dict': optimizer.state_dict(),
            },
                file_name)

            model_save = os.path.join(args.savepath, 'model_last_k={}.pt'.format(i + 1))
            #torch.save(model, model_save)  # 同时要保留模型.py文件
            torch.save(model.module, model_save)  # todo 用到nn.DataParallel时保存模型的方式

        # ======================= 可视化(每一折最后一轮) ======================= #
        # 损失和准确率可视化
        image_result = os.path.join(os.getcwd(), '结果图')
        if os.path.exists(image_result) is False:
            os.makedirs(image_result)
        vis.Loss_Acc(args=args, train_loss=train_loss_lst, test_loss=test_loss_lst,
                     train_acc=train_acc_lst, test_acc=test_acc_lst, k=i + 1)

        # AUC、ROC可视化
        npy_result = os.path.join(os.getcwd(), '结果acc_loss')
        if os.path.exists(npy_result) is False:
            os.makedirs(npy_result)
        np.save('{}/结果acc_loss/{}_{}_k={}'.format(os.getcwd(), args.savename, 'Score_array', i + 1),
                Score_array)  # 只需每折最后一轮的ROC
        np.save('{}/结果acc_loss/{}_{}_k={}'.format(os.getcwd(), args.savename, 'Label_onehot', i + 1), Label_onehot)
        roc_auc = vis.plot_AUC(y_scores=Score_array, y_test=Label_onehot, args=args, k=i + 1,
                     cls=args.cls_list)  # 取每折最后一轮Test的结果

        # 混淆矩阵可视化
        CM = confusion_matrix(y_true=Actuals, y_pred=Preds)
        CM = (CM.astype('float') / CM.sum(axis=1)[:, np.newaxis])  # 归一化
        CM = np.around(CM, decimals=3, out=None)
        plt.figure(figsize=(14, 6))
        sns_plot = sns.heatmap(CM, annot=True, linewidths=0.1, fmt=',', annot_kws={'size': 25}, cmap='Blues',
                               xticklabels=args.cls_list, yticklabels=args.cls_list)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title('Confusion Matrix: {}'.format(args.savename), fontsize=15)
        plt.xlabel('Pred', fontsize=25)
        plt.ylabel('True', fontsize=25)
        plt.tight_layout()
        if args.savename is not None:
            cur_path = os.getcwd()
            plt.savefig('{}/结果图/{}_CM.png'.format(cur_path, args.savename + '_k=' + str(i + 1)))
        plt.show()

        # 其余指标
        print(classification_report(y_true=Actuals, y_pred=Preds, target_names=args.cls_list))
        f = open(os.path.join(args.savepath, 'hetero_training.txt'), 'a')
        f.write('k=' + str(i + 1) + '\n' + classification_report(y_true=Actuals, y_pred=Preds, target_names=args.cls_list) + '\n')

        # 数据汇总
        ws = wb.add_sheet('{}_k={}'.format(args.savename.split('_')[0], i+1))
        # 准确率
        ws.write(0,0, '测试集准确率') # 行，列
        ws.write(1,0, round(float(test_acc_lst[-1]), 3))
        # AUC
        ws.write(0,1, 'AUC')
        for row in range(3):
            ws.write(row+1,1, round(float(roc_auc[row]), 3))
        # 混淆矩阵
        ws.write(0,2, '混淆矩阵')
        for row in range(3):
            for col in range(3):
                ws.write(row+1,col+2, CM[row,col])
        # Precision
        ws.write(0,5, 'Precision')
        for row in range(3):
            ws.write(row+1, 5, round(float(precision_score(y_true=Actuals, y_pred=Preds, average=None)[row]),2))
        # Recall
        ws.write(0,6, 'Recall')
        for row in range(3):
            ws.write(row+1, 6, round(float(recall_score(y_true=Actuals, y_pred=Preds, average=None)[row]),2))
        # F1-score
        ws.write(0, 7, 'F1-score')
        for row in range(3):
            ws.write(row+1, 7, round(float(f1_score(y_true=Actuals, y_pred=Preds, average=None)[row]), 2))

    # ======================= 结果记录 ======================= #
    save_npy(nparr=k_train_acc_lst, main_name='k_train_acc_lst', save_name=args.savename)
    save_npy(nparr=k_train_loss_lst, main_name='k_train_loss_lst', save_name=args.savename)
    save_npy(nparr=k_test_acc_lst, main_name='k_test_acc_lst', save_name=args.savename)
    save_npy(nparr=k_test_loss_lst, main_name='k_test_loss_lst', save_name=args.savename)
    wb.save(os.path.join(args.savepath, 'results.xls'))
    print('K折交叉验证平均结果：')
    k_train_acc_lst = torch.tensor(k_train_acc_lst)
    k_train_loss_lst = torch.tensor(k_train_loss_lst)
    k_test_acc_lst = torch.tensor(k_test_acc_lst)
    k_test_loss_lst = torch.tensor(k_test_loss_lst)
    k_Train_Acc = sum(k_train_acc_lst[:, -1]) / args.kfold
    k_Train_Loss = sum(k_train_loss_lst[:, -1]) / args.kfold
    k_Test_Acc = sum(k_test_acc_lst[:, -1]) / args.kfold
    k_Test_Loss = sum(k_test_loss_lst[:, -1]) / args.kfold
    print('Train_Acc: {:.4f}%, Train_Loss: {:.4f} \n Test_Acc: {:.4f}%, Test_Loss: {:.4f}'.format(
        k_Train_Acc, k_Train_Loss, k_Test_Acc, k_Test_Loss))

    print("完成")


if __name__ == '__main__':
    main()
