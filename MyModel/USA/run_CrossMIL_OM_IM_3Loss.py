#coding=utf-8
"""
模态消融实验：不使用TEM模态时，Ours模型退化为OM、IM的简单特征融合，加上加权损失。
"""
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
from MMdataset import MMdata_EMIL
from tools.result_save import save_json, write2xlsx
from torch.utils.tensorboard import SummaryWriter
from utils.parser import setup
from torch.nn.utils.rnn import pad_sequence
import warnings
from torch.nn import functional as F
import xlwt
import setproctitle
import datetime
warnings.filterwarnings("ignore")

# ======================= 定义训练函数 ======================= #
def train_one_epoch(loader, model, args, criterion, optimizer):
    total_sample_train = 0
    train_loss = 0.0
    batch_acc = 0  # 一个batch内的准确数量

    model.train()
    for idx, data in enumerate(loader):
        bags_tensor, OM_tensor, IM_tensor, target = data
        OM = OM_tensor.to(device=args.device, non_blocking=True)
        IM = IM_tensor.to(device=args.device, non_blocking=True)

        target = target.to(device=args.device, non_blocking=True)
        target_onehot = F.one_hot(target, args.num_cls).float()
        target_onehot = target_onehot.to(device=args.device, non_blocking=True)

        # 最终预测结果以及各模态预测结果
        output, OM_output, IM_output = model(None, OM_tensor=OM, IM_tensor=IM)

        # 每个batch单独计算loss和预测结果pred
        pred = []
        for b in range(args.batch_size):
            if b == 0:
                fusion_loss = criterion(output[b], target_onehot[b].unsqueeze(0))

                OM_loss = criterion(OM_output[b], target_onehot[b].unsqueeze(0))
                IM_loss = criterion(IM_output[b], target_onehot[b].unsqueeze(0))
            else:
                fusion_loss += criterion(output[b], target_onehot[b].unsqueeze(0))

                OM_loss += criterion(OM_output[b], target_onehot[b].unsqueeze(0))
                IM_loss += criterion(IM_output[b], target_onehot[b].unsqueeze(0))

            pred.append(torch.max(output[b], dim=1)[1])

        optimizer.zero_grad()
        total_loss = (0.3/0.8)*OM_loss + (0.5/0.8)*IM_loss + fusion_loss
        total_loss.backward()
        optimizer.step()

        train_loss += total_loss.item()

        # 计算一个batch内的准确率
        for b in range(args.batch_size):
            batch_acc += (torch.eq(pred[b], target[b].sum().item()))
        total_sample_train += target.size(0)

    Train_acc = (100 * batch_acc / total_sample_train).item()

    Train_loss = train_loss / total_sample_train

    print('\nTrain Acc: {:.4f}%, Train Loss: {:.4f}'.format(Train_acc, Train_loss))

    return Train_acc, Train_loss

# ======================= 定义测试函数 ======================= #
def Test(loader, model, args, criterion):
    total_sample_test = 0
    test_loss = 0.0
    batch_acc = 0  # 一个batch内的准确数量
    score_list, label_list = [], []
    preds, actuals = [], []
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(loader):
            bags_tensor, OM_tensor, IM_tensor, target = data
            OM = OM_tensor.to(device=args.device, non_blocking=True)
            IM = IM_tensor.to(device=args.device, non_blocking=True)

            target = target.to(device=args.device, non_blocking=True)
            target_onehot = F.one_hot(target, args.num_cls).float()
            target_onehot = target_onehot.to(device=args.device, non_blocking=True)

            # 光镜+荧光+电镜
            output, OM_output, IM_output = model(None, OM_tensor=OM, IM_tensor=IM)

            # 每个batch单独计算loss和预测结果pred
            pred = []
            for b in range(args.batch_size):
                if b == 0:
                    fusion_loss = criterion(output[b], target_onehot[b].unsqueeze(0))

                    OM_loss = criterion(OM_output[b], target_onehot[b].unsqueeze(0))
                    IM_loss = criterion(IM_output[b], target_onehot[b].unsqueeze(0))
                else:
                    fusion_loss += criterion(output[b], target_onehot[b].unsqueeze(0))

                    OM_loss += criterion(OM_output[b], target_onehot[b].unsqueeze(0))
                    IM_loss += criterion(IM_output[b], target_onehot[b].unsqueeze(0))

                pred.append(torch.max(output[b], dim=1)[1])
                preds.extend(torch.max(output[b], dim=1)[1].cpu().numpy())
                score_list.extend(output[b].cpu().numpy())

            total_loss = (0.3/0.8) * OM_loss + (0.5/0.8) * IM_loss + fusion_loss

            test_loss += total_loss.item()

            actuals.extend(target.cpu().numpy())

            # 用于计算AUC
            label_list.extend(target.cpu().numpy())

            # 计算一个batch内的准确率
            for b in range(args.batch_size):
                batch_acc += (torch.eq(pred[b], target[b].sum().item()))
            total_sample_test += target.size(0)

        Test_acc = (100 * batch_acc / total_sample_test).item()

        score_array = np.array(score_list)
        # 将label转换成onehot形式
        label_tensor = torch.tensor(label_list)
        label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
        label_onehot = torch.zeros(label_tensor.shape[0], args.num_cls)
        label_onehot.scatter_(dim=1, index=label_tensor, value=1)
        label_onehot = np.array(label_onehot)

        # 计算并显示测试集的损失函数
        Test_loss = test_loss / total_sample_test
        print("Test Acc: {:.4f}%, Test Loss: {:.4f}".format(Test_acc, Test_loss))

        return Test_acc, Test_loss, actuals, preds, score_array, label_onehot


# ======================= 参数设置 ======================= #
parser = argparse.ArgumentParser()
from MyModel.USA import CrossMIL_OM_IM_3Loss_250403


parser.add_argument('--Description',
                    default='run_CrossMIL_OM_IM_3Loss。光镜+荧光,wLoss。光镜、荧光模态特征拼接。')
parser.add_argument('--savename', default='250403_光镜（PASM）+荧光+电镜-301+270例_CrossMIL_OM_IM_3Loss_250403')
setproctitle.setproctitle("lkx_预计用至04/04/18：00.pm")  # 修改进程名称（放在最前面）

# *********************** 数据集相关 *********************** #
parser.add_argument('--data_path', default='/public/longkaixing/MMF/datasets/光镜（PASM）+荧光+电镜-301+270例', type=str)
parser.add_argument('--num_epochs', default=40, type=int)
parser.add_argument('--cls_list', default=['MN', 'IgAN','LN'], type=list, help='类别名称及其对应于label的顺序')
parser.add_argument('--sample_index', type=str,
                        default='/public/longkaixing/MMF/MyMethod/数据集交叉划分index/光镜（PASM）+荧光+电镜-301+270例',
                        help="五折交叉验证划分数据集的索引路径")

# parser.add_argument('--data_path', default='/public/longkaixing/CrossModalScale/datasets/MN分期-三模态全-前三期', type=str)
# parser.add_argument('--num_epochs', default=40, type=int)
# parser.add_argument('--cls_list', default=['一期', '二期','三期'], type=list, help='类别名称及其对应于label的顺序')
# parser.add_argument('--sample_index', type=str,
#                         default='/public/longkaixing/CrossModalScale/数据集交叉划分index/MN分期-三模态全-前三期',
#                         help="k折交叉验证划分数据集的索引路径")
# ******************************************************** #

parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--modal', default='OM_+IM', type=str, help='所用模态：IM+TEM / OM+TEM / OM+IM (OM_ 为分割后的光镜, OM-ori为分割前)')
parser.add_argument('--lr', default=5e-5, type=float)
parser.add_argument('--num_cls', default=3, type=int, help='分为几类')
parser.add_argument('--device', default='cuda:1', type=str)
parser.add_argument('--kfold', default=5, type=int)
parser.add_argument('--weight_decay', default=5e-5, type=float)
parser.add_argument('--CAM_img_path',
                    default='/public/longkaixing/MMF/datasets/光镜+荧光+电镜-301例-3类/IgA/KB2009806/...',
                    type=str)
parser.add_argument('--reshape', default=[224, 224])


criterion = nn.CrossEntropyLoss()

args = parser.parse_args()

args.savepath = '/public/longkaixing/CrossModalScale/model_save/{}'.format(args.savename)
setup(args, 'training')
os.makedirs(args.savepath, exist_ok=True)

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
    data_set = MMdata_EMIL(transform=args.transforms, root=args.data_path, modal=args.modal)
    print('数据集：', args.data_path)

    # ======================= K折交叉验证 ======================= #
    k_train_acc_lst, k_train_loss_lst = [], []  # K次平均结果
    k_test_acc_lst, k_test_loss_lst = [], []
    train_loader_lst, test_loader_lst = [], []

    for i in range(0, args.kfold):
        # 载入预先划分的数据集index
        train_index = np.load('{}_train_index_k={}.npy'.format(args.sample_index, i + 1))
        #random.shuffle(train_index)
        test_index = np.load('{}_test_index_k={}.npy'.format(args.sample_index, i + 1))
        #random.shuffle(test_index)
        # todo 电镜用MIL时加上 ,collate_fn=TEM_dataset_collate
        train_loader = DataLoader(data_set, batch_size=args.batch_size, sampler=train_index[:],num_workers=0,
                                  drop_last=True,pin_memory=False) # 锁页内存，加速数据传输读取
        test_loader = DataLoader(data_set, batch_size=args.batch_size, sampler=test_index[:],num_workers=0,
                                 drop_last=True,pin_memory=False)
        train_loader_lst.append(train_loader)
        test_loader_lst.append(test_loader)

    wb = xlwt.Workbook() # 用于记录全部折次数据

    log_dir = os.path.join(args.savepath, 'tensorboard_logs')
    os.makedirs(log_dir, exist_ok=True)  # 确保目录存在
    # ======================= 模型训练/测试 ======================= #
    for k in range(1, args.kfold+1): # 从第k折开始
        # 获取当前时间并格式化
        current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        custom_file_name = f"_{current_time}-fold{k}.tfevents"
        writer = SummaryWriter(log_dir=log_dir, filename_suffix=custom_file_name)

        # ======================= 模型设置 ======================= #
        model = CrossMIL_OM_IM_3Loss_250403.Model(args)
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
        print('# ======================= K折交叉验证：{}/{} ======================= #'.format(k, args.kfold))
        train_loader = train_loader_lst[k-1]
        test_loader = test_loader_lst[k-1]
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
            # 记录损失和准确率
            writer.add_scalar('Train/Loss', Train_Loss, epoch)
            writer.add_scalar('Train/Accuracy', Train_Acc, epoch)
            writer.add_scalar('Test/Loss', Test_Loss, epoch)
            writer.add_scalar('Test/Accuracy', Test_Acc, epoch)

        # 记录每折训练测试曲线
        k_train_acc_lst.append(train_acc_lst)
        k_train_loss_lst.append(train_loss_lst)
        k_test_acc_lst.append(test_acc_lst)
        k_test_loss_lst.append(test_loss_lst)

        # ======================= 保存模型(每折) ======================= #
        if args.savename is not None:
            file_name = os.path.join(args.savepath, 'model_last_k={}.pth'.format(k))
            torch.save({
                'epoch': epoch,
                #'state_dict': model.state_dict(),
                'state_dict': model.module.state_dict(),  # todo 用到nn.DataParallel时保存模型的方式
                'optim_dict': optimizer.state_dict(),
            },
                file_name)

            model_save = os.path.join(args.savepath, 'model_last_k={}.pt'.format(k))
            #torch.save(model, model_save)  # 同时要保留模型.py文件
            torch.save(model.module, model_save)  # todo 用到nn.DataParallel时保存模型的方式

        # ======================= 可视化(每一折最后一轮) ======================= #
        # 损失和准确率可视化
        image_result = os.path.join(os.getcwd(), '结果图')
        if os.path.exists(image_result) is False:
            os.makedirs(image_result)
        vis.Loss_Acc(args.num_epochs, args.savename, train_loss=train_loss_lst, test_loss=test_loss_lst,
                     train_acc=train_acc_lst, test_acc=test_acc_lst, k=k)  # todo

        # AUC、ROC可视化
        npy_result = os.path.join(os.getcwd(), '结果acc_loss')
        if os.path.exists(npy_result) is False:
            os.makedirs(npy_result)
        np.save(f'{os.getcwd()}/结果acc_loss/{args.savename}_Score_array_k={k}', Score_array)  # 只需每折最后一轮的ROC
        # 将actuals_array转换为one-hot编码
        actuals_onehot = F.one_hot(torch.tensor(Actuals), num_classes=args.num_cls).numpy()
        np.save(f'{os.getcwd()}/结果acc_loss/{args.savename}_Label_onehot_k={k}', actuals_onehot)
        roc_auc, spe = vis.plot_AUC(y_scores=Score_array, y_test=actuals_onehot, n_classes=args.num_cls, k=k,
                                    cls=args.cls_list, savename=args.savename)  # 取每折最后一轮Test的结果

        # 混淆矩阵可视化
        Actuals = np.argmax(actuals_onehot, axis=1)  # 类别索引形式
        CM = confusion_matrix(y_true=Actuals, y_pred=Preds)
        CM = (CM.astype('float') / CM.sum(axis=1)[:, np.newaxis])  # 归一化 不采用时记得修改write2xlsx中写入CM的类型
        CM = np.around(CM, decimals=4, out=None)
        plt.figure(figsize=(8, 6))
        sns_plot = sns.heatmap(CM, annot=True, linewidths=0.1, fmt=',', annot_kws={'size': 25}, cmap='Blues',
                               xticklabels=args.cls_list, yticklabels=args.cls_list)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title(f'{args.savename}', fontsize=15)
        plt.xlabel('Pred', fontsize=25)
        plt.ylabel('True', fontsize=25)
        plt.tight_layout()
        if args.savename is not None:
            plt.savefig(f'{os.getcwd()}/结果图/{args.savename}_k={k}_CM.png')
        plt.show()

        # CAM可视化
        # vis.CAM(model, args, target_layers=[model.module.TEM_encoder[7][-1]], k=i+1) # todo nn.DataParallel

        # 其余指标
        print(classification_report(y_true=Actuals, y_pred=Preds, target_names=args.cls_list, zero_division=0,
                                    digits=4))
        f = open(os.path.join(args.savepath, 'hetero_training.txt'), 'a')
        f.write('\n' + 'k=' + str(k) + '\n' + classification_report(y_true=Actuals, y_pred=Preds,
                                                                    target_names=args.cls_list, zero_division=0,
                                                                    digits=4) + '\n')

        # 数据汇总
        write2xlsx(wb, args.num_cls, args.savename, k, test_acc_lst, roc_auc, CM, Actuals, Preds, spe)

        writer.close()

    # ======================= 结果记录 ======================= #
    save_json(k_train_acc_lst, main_name='k_train_acc_lst', save_name=args.savename)
    save_json(k_train_loss_lst, main_name='k_train_loss_lst', save_name=args.savename)
    save_json(k_test_acc_lst, main_name='k_test_acc_lst', save_name=args.savename)
    save_json(k_test_loss_lst, main_name='k_test_loss_lst', save_name=args.savename)
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
    writer.close()
    print('Train_Acc: {:.4f}%, Train_Loss: {:.4f} \n Test_Acc: {:.4f}%, Test_Loss: {:.4f}'.format(
        k_Train_Acc, k_Train_Loss, k_Test_Acc, k_Test_Loss))

    print("完成")


if __name__ == '__main__':
    main()
