#coding=utf-8
"""
训练CMUSNet。
"""
import os.path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import seaborn as sns
import logging
from MMdataset import MMdata_EMIL
from torch.utils.tensorboard import SummaryWriter

from torch.nn.utils.rnn import pad_sequence
import warnings
from torch.nn import functional as F
import xlwt
import setproctitle
import datetime
import pandas as pd

from util.pred_loading import load_config
from util import visualization as vis
from util.parser import log_setup
from util.fold_vis_save import save_config, save_json, write2xlsx

join = os.path.join

warnings.filterwarnings("ignore")

# ======================= 重写collate_fn,为了使Dataloader在batch_size不一致时也能工作 ======================= #
def TEM_dataset_collate(batch):
    Bags = []
    img_OM, img_IM = [], []
    labels = []
    OMs = torch.tensor(0)
    IFs = torch.tensor(0)
    for bag, OM, IM, y in batch:
        Bags.append(torch.stack(bag, dim=0))
        if not type(OM) == list:
            img_OM.append(OM)
        if not type(IM) == list:
            img_IM.append(IM)
        labels.append(y)

    if len(img_OM) != 0:
        OMs = torch.stack(img_OM) # [B,C,H,W]
    if len(img_IM) != 0:
        IFs = torch.stack(img_IM)
    labels = torch.stack(labels)
    return Bags, OMs, IFs, labels


# ======================= 定义训练函数 ======================= #
def train_one_epoch(loader, model, args, criterion, optimizer):
    total_sample_train = 0
    train_loss = 0.0
    batch_acc = 0  # 一个batch内的准确数量
    Modal_loss = {'OM_loss': [], 'IM_loss': [], 'TEM_loss': []}

    model.train()
    for idx, data in enumerate(loader):
        bags_tensor, OM_tensor, IM_tensor, target = data
        OM = OM_tensor.cuda()
        IM = IM_tensor.cuda()

        target = target.cuda()
        target_onehot = F.one_hot(target, args.num_cls).float()
        target_onehot = target_onehot.cuda()

        # 将图像数量不等的张量用0填充至等长，以便组成(Batch_size, 图像数量, 通道数, 长, 宽)的张量
        bags = pad_sequence(bags_tensor, batch_first=True, padding_value=0).cuda()
        # 最终预测结果以及各模态预测结果
        output = model(bags, OM_tensor=OM, IM_tensor=IM)

        # 每个batch单独计算loss和预测结果pred
        pred, epoch_avg_d = [], []
        for b in range(args.batch_size):
            if b == 0:
                fusion_loss = criterion(output['fusion'][b], target_onehot[b].unsqueeze(0))

                OM_loss = criterion(output['OM'][b], target_onehot[b].unsqueeze(0))
                IM_loss = criterion(output['IM'][b], target_onehot[b].unsqueeze(0))
                TEM_loss = criterion(output['TEM'][b], target_onehot[b].unsqueeze(0))
            else:
                fusion_loss += criterion(output['fusion'][b], target_onehot[b].unsqueeze(0))

                OM_loss += criterion(output['OM'][b], target_onehot[b].unsqueeze(0))
                IM_loss += criterion(output['IM'][b], target_onehot[b].unsqueeze(0))
                TEM_loss += criterion(output['TEM'][b], target_onehot[b].unsqueeze(0))

            pred.append(torch.max(output['fusion'][b], dim=1)[1])
            epoch_avg_d.append(output['distance'].cpu().numpy())

        optimizer.zero_grad()
        total_loss = 0.3*OM_loss + 0.5*IM_loss + 0.2*TEM_loss + fusion_loss

        total_loss.backward()
        optimizer.step()

        train_loss += total_loss.item()
        Modal_loss['OM_loss'].append(OM_loss.item())  # 更新字典中的OM_loss
        Modal_loss['IM_loss'].append(IM_loss.item())  # 更新字典中的IM_loss
        Modal_loss['TEM_loss'].append(TEM_loss.item())  # 更新字典中的TEM_loss

        # 计算一个batch内的准确率
        for b in range(args.batch_size):
            batch_acc += (torch.eq(pred[b], target[b].sum().item()))
        total_sample_train += target.size(0)

    Train_acc = (100 * batch_acc / total_sample_train).item()

    Train_loss = train_loss / total_sample_train
    Modal_loss['OM_loss'] = sum(Modal_loss['OM_loss']) / total_sample_train
    Modal_loss['IM_loss'] = sum(Modal_loss['IM_loss']) / total_sample_train
    Modal_loss['TEM_loss'] = sum(Modal_loss['TEM_loss']) / total_sample_train

    print(f'\nTrain Acc: {Train_acc:.4f}%, Train Loss: {Train_loss:.4f}')
    print(f'{Modal_loss}')
    print(f'每轮平均距离：{np.mean(epoch_avg_d):.4f}')

    return Train_acc, Train_loss, Modal_loss, np.mean(epoch_avg_d)

# ======================= 定义测试函数 ======================= #
def Test(loader, model, args, criterion):
    total_sample_test = 0
    test_loss = 0.0
    batch_acc = 0  # 一个batch内的准确数量
    Modal_loss = {'OM_loss': [], 'IM_loss': [], 'TEM_loss': []}
    score_list, label_list = [], []
    preds, actuals = [], []
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(loader):
            bags_tensor, OM_tensor, IM_tensor, target = data
            OM = OM_tensor.cuda()
            IM = IM_tensor.cuda()

            target = target.cuda()
            target_onehot = F.one_hot(target, args.num_cls).float()
            target_onehot = target_onehot.cuda()

            # 将图像数量不等的张量用0填充至等长，以便组成(Batch_size, 图像数量, 通道数, 长, 宽)的张量
            bags = pad_sequence(bags_tensor, batch_first=True, padding_value=0).cuda()
            # 光镜+荧光+电镜
            output = model(bags, OM_tensor=OM, IM_tensor=IM)

            # 每个batch单独计算loss和预测结果pred
            pred = []
            for b in range(args.batch_size):
                if b == 0:
                    fusion_loss = criterion(output['fusion'][b], target_onehot[b].unsqueeze(0))

                    OM_loss = criterion(output['OM'][b], target_onehot[b].unsqueeze(0))
                    IM_loss = criterion(output['IM'][b], target_onehot[b].unsqueeze(0))
                    TEM_loss = criterion(output['TEM'][b], target_onehot[b].unsqueeze(0))
                else:
                    fusion_loss += criterion(output['fusion'][b], target_onehot[b].unsqueeze(0))

                    OM_loss += criterion(output['OM'][b], target_onehot[b].unsqueeze(0))
                    IM_loss += criterion(output['IM'][b], target_onehot[b].unsqueeze(0))
                    TEM_loss += criterion(output['TEM'][b], target_onehot[b].unsqueeze(0))

                pred.append(torch.max(output['fusion'][b], dim=1)[1])
                preds.extend(torch.max(output['fusion'][b], dim=1)[1].cpu().numpy())
                score_list.extend(output['fusion'][b].cpu().numpy())

            total_loss = 0.3 * OM_loss + 0.5 * IM_loss + 0.2 * TEM_loss + fusion_loss

            test_loss += total_loss.item()
            Modal_loss['OM_loss'].append(OM_loss.item())  # 更新字典中的OM_loss
            Modal_loss['IM_loss'].append(IM_loss.item())  # 更新字典中的IM_loss
            Modal_loss['TEM_loss'].append(TEM_loss.item())  # 更新字典中的TEM_loss

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
        Modal_loss['OM_loss'] = sum(Modal_loss['OM_loss']) / total_sample_test
        Modal_loss['IM_loss'] = sum(Modal_loss['IM_loss']) / total_sample_test
        Modal_loss['TEM_loss'] = sum(Modal_loss['TEM_loss']) / total_sample_test

        print("Test Acc: {:.4f}%, Test Loss: {:.4f}".format(Test_acc, Test_loss))
        print(f'{Modal_loss}')

        return Test_acc, Test_loss, Modal_loss, actuals, preds, score_array, label_onehot


# ======================= 参数设置 ======================= #
parser = argparse.ArgumentParser()
from MyModel.CMUS_Net import CMUSNet

setproctitle.setproctitle("lkx_预计用至12/01/23：00.am")  # 修改进程名称（放在最前面）
parser.add_argument('--Description',
    default='train_CMUSNet。电镜q+光镜kv+荧光kv,wLoss。电镜加权光镜、荧光。电镜特征距离筛查与加权(权重归一化)。模态拼接。')
parser.add_argument('--cfg', type=str, help='配置文件路径',
    default='/public/longkaixing/CrossModalScale/config/251130_CMUSNet.yaml')

args, _ = parser.parse_known_args()
config = yaml.load(open(f"{args.cfg}", "r"), Loader=yaml.FullLoader)

if args.cfg:  # 将config的参数配置复制添加到args中
    config = load_config(args.cfg)
    for key, value in config.items():
        setattr(args, key, value)

# -------- 数据扩增 --------
args.transforms = transforms.Compose([
    transforms.Resize([args.input_size, args.input_size], antialias=True),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------- 记录设置 --------
# 基础保存路径
args.savepath = join(args.save_dir, args.save_name)
os.makedirs(args.savepath, exist_ok=True)

# 记录参数信息并打印
log_setup(args.savepath, 'training')
logging.info(str(config))  # 打印参数信息

# 记录配置文件
config_save_path = save_config(args.cfg, args.savepath, args.save_name)

# 记录训练日志
log_dir = join(args.savepath, 'tensorboard_logs')
os.makedirs(log_dir, exist_ok=True)  # 确保目录存在

# -------- 训练环境设置 --------
gpu_ids = tuple(args.gpu_ids)
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)


# ======================= 载入数据集 ======================= #
data_set = MMdata_EMIL(transform=args.transforms, root=args.data_path, modal=args.modal)
print('数据集：', args.data_path)

# ======================= K折交叉验证 ======================= #
wb = xlwt.Workbook() # 用于记录全部折次数据
k_train_acc_lst, k_train_loss_lst, k_train_modal_loss_lst = [], [], []  # K次平均结果
k_test_acc_lst, k_test_loss_lst, k_test_modal_loss_lst = [], [], []
k_epoch_avg_d_lst = []
train_loader_lst, test_loader_lst = [], []

for k_idx in range(1, args.kfold+1):
    # 载入预先划分的数据集index
    train_index = np.load('{}_train_index_k={}.npy'.format(args.sample_index, k_idx))
    test_index = np.load('{}_test_index_k={}.npy'.format(args.sample_index, k_idx))
    # todo 电镜用MIL时加上 ,collate_fn=TEM_dataset_collate
    train_loader = DataLoader(data_set, batch_size=args.batch_size, sampler=train_index[:20],num_workers=args.num_workers,
                              drop_last=True,collate_fn=TEM_dataset_collate,pin_memory=False) # 锁页内存，加速数据传输读取
    test_loader = DataLoader(data_set, batch_size=args.batch_size, sampler=test_index[:10],num_workers=args.num_workers,
                             drop_last=True,collate_fn=TEM_dataset_collate,pin_memory=False)
    train_loader_lst.append(train_loader)
    test_loader_lst.append(test_loader)

# ======================= 模型训练/测试 ======================= #
for k_idx in range(1, args.kfold+1): # 从第k折开始
    # 获取当前时间并格式化
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    custom_file_name = f"_{current_time}-fold{k_idx}.tfevents"
    writer = SummaryWriter(log_dir=log_dir, filename_suffix=custom_file_name)
    # ======================= 模型设置 ======================= #
    model = CMUSNet.Model(args)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=[gpu_ids])  # 分布式训练时使用的gpu索引
    # if i == 0:
    #     print(model)
    for param in model.parameters():
        param.requires_grad = True  # 冻结所有参数时为False

    criterion = nn.CrossEntropyLoss()

    ##########Setting learning schedule and optimizer
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}]
    optimizer = torch.optim.Adam(train_params, betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs,
                                                                 eta_min=0.000005)  # 余弦退火调整学习率
    print('# ======================= K折交叉验证：{}/{} ======================= #'.format(k_idx, args.kfold))
    train_loader = train_loader_lst[k_idx-1]
    test_loader = test_loader_lst[k_idx-1]
    # 用于记录每轮结果
    train_acc_lst, train_loss_lst, train_modal_loss_lst = [], [], []
    epoch_avg_d_lst = []
    test_acc_lst, test_loss_lst, test_modal_loss_lst  = [], [], []

    fold_best_acc = 0
    args.n_epoch = args.num_epochs
    counter = 0  # 用于记录早停
    for epoch in tqdm(range(1, args.num_epochs + 1)):
        counter += 1
        Train_Acc, Train_Loss, Train_Modal_Loss, Epoch_Avg_D = train_one_epoch(train_loader, model, args, criterion, optimizer)
        Test_Acc, Test_Loss, Test_Modal_Loss, Actuals, Preds, Score_array, Label_onehot = Test(test_loader, model, args, criterion)
        train_acc_lst.append(Train_Acc)
        train_loss_lst.append(Train_Loss)
        train_modal_loss_lst.append(Train_Modal_Loss)
        epoch_avg_d_lst.append(float(Epoch_Avg_D))
        test_acc_lst.append(Test_Acc)
        test_loss_lst.append(Test_Loss)
        test_modal_loss_lst.append(Test_Modal_Loss)

        lr_schedule.step()  # 更新学习率

        if Test_Acc > fold_best_acc:
            counter = 0
            fold_best_acc = Test_Acc
        # if counter > args.stop_epochs:  # 早停机制
        #     print('Triggered the early stop mechanism.')
        #     args.n_epoch = epoch # 变更当前轮次总数
        #     break

    # 记录每折训练测试曲线
    k_train_acc_lst.append(train_acc_lst)
    k_train_loss_lst.append(train_loss_lst)
    k_train_modal_loss_lst.append(train_modal_loss_lst)
    k_epoch_avg_d_lst.append(epoch_avg_d_lst)
    k_test_acc_lst.append(test_acc_lst)
    k_test_loss_lst.append(test_loss_lst)
    k_test_modal_loss_lst.append(test_modal_loss_lst)

    # ======================= 保存模型(每折) ======================= #
    if args.save_name is not None:
        file_name = os.path.join(args.savepath, 'model_last_k={}.pth'.format(k_idx))
        torch.save({
            'epoch': epoch,
            #'state_dict': model.state_dict(),
            'state_dict': model.module.state_dict(),  # todo 用到nn.DataParallel时保存模型的方式
            'optim_dict': optimizer.state_dict(),
        },
            file_name)

        model_save = os.path.join(args.savepath, 'model_last_k={}.pt'.format(k_idx))
        #torch.save(model, model_save)  # 同时要保留模型.py文件
        torch.save(model.module, model_save)  # todo 用到nn.DataParallel时保存模型的方式

    # ======================= 可视化(每一折最后一轮) ======================= #
    # 损失和准确率可视化
    image_result = os.path.join(os.getcwd(), '结果图')
    if os.path.exists(image_result) is False:
        os.makedirs(image_result)
    vis.Loss_Acc(args.n_epoch, args.save_name, train_loss=train_loss_lst, test_loss=test_loss_lst,
                 train_acc=train_acc_lst, test_acc=test_acc_lst, k=k_idx)  # todo

    # AUC、ROC可视化
    npy_result = os.path.join(os.getcwd(), '结果acc_loss')
    if os.path.exists(npy_result) is False:
        os.makedirs(npy_result)
    np.save(f'{os.getcwd()}/结果acc_loss/{args.save_name}_Score_array_k={k_idx}', Score_array)  # 只需每折最后一轮的ROC
    # 将actuals_array转换为one-hot编码
    actuals_onehot = F.one_hot(torch.tensor(Actuals), num_classes=args.num_cls).numpy()
    np.save(f'{os.getcwd()}/结果acc_loss/{args.save_name}_Label_onehot_k={k_idx}', actuals_onehot)
    roc_auc, spe = vis.plot_AUC(y_scores=Score_array, y_test=actuals_onehot, n_classes=args.num_cls, k=k_idx,
                                cls=args.cls_list, savename=args.save_name)  # 取每折最后一轮Test的结果

    # 混淆矩阵可视化
    Actuals = np.argmax(actuals_onehot, axis=1)  # 类别索引形式
    CM = confusion_matrix(y_true=Actuals, y_pred=Preds)
    CM_norm = CM.astype('float') / CM.sum(axis=1)[:, np.newaxis]

    annot = np.empty_like(CM, dtype=object)
    for i in range(CM.shape[0]):
        for j in range(CM.shape[1]):
            count = CM[i, j]
            percent = CM_norm[i, j] * 100
            annot[i, j] = f"{int(count)}\n({percent:.2f}%)"

    plt.figure(figsize=(8, 6))
    sns.heatmap(CM_norm, annot=annot, fmt='', cmap='Blues',
                vmin=0, vmax=1, # cbar显示值域
                xticklabels=args.cls_list, yticklabels=args.cls_list,
                annot_kws={'size': 25})
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(f'{args.save_name}_k={args.k_idx}', fontsize=15)
    plt.xlabel('Pred', fontsize=25)
    plt.ylabel('True', fontsize=25)
    plt.tight_layout()
    plt.savefig(os.path.join(image_result, f'{args.save_name}_k={args.k_idx}_CM.png'))
    plt.show()
    plt.close()

    # 分类报告
    report_dict = classification_report(y_true=Actuals,
                                        y_pred=Preds,
                                        labels=range(args.num_classes),  # 添加这一行
                                        target_names=args.cls_list,
                                        output_dict=True,
                                        zero_division=0)
    report_df = pd.DataFrame(report_dict).T.round(4)
    logging.info(f'\nk={args.k_idx}\n{report_df.to_string()}')

    report = classification_report(y_true=Actuals, y_pred=Preds, target_names=args.cls_list, zero_division=0,
                                   digits=4)
    log_message = f"K-Fold: {k_idx}\n" + report
    # 打印到控制台
    print(log_message)
    # 将内容写入文件
    with open(os.path.join(args.savepath, 'hetero_training.txt'), 'a') as f:
        f.write(log_message + '\n')

    # 数据汇总
    write2xlsx(wb, args.num_cls, args.save_name, k_idx, test_acc_lst, roc_auc, CM, Actuals, Preds, spe)

    writer.close()

# ======================= 结果记录 ======================= #
save_json(k_train_acc_lst, main_name='k_train_acc_lst', save_name=args.save_name, path=args.savepath)
save_json(k_train_loss_lst, main_name='k_train_loss_lst', save_name=args.save_name, path=args.savepath)
save_json(k_train_modal_loss_lst, main_name='k_train_modal_loss_lst', save_name=args.save_name, path=args.savepath)
save_json(k_epoch_avg_d_lst, main_name='k_epoch_avg_d_lst', save_name=args.save_name, path=args.savepath)
save_json(k_test_acc_lst, main_name='k_test_acc_lst', save_name=args.save_name, path=args.savepath)
save_json(k_test_loss_lst, main_name='k_test_loss_lst', save_name=args.save_name, path=args.savepath)
save_json(k_test_modal_loss_lst, main_name='k_test_modal_loss_lst', save_name=args.save_name, path=args.savepath)
wb.save(os.path.join(args.savepath, 'results.xls'))
print('K折交叉验证平均结果：')
k_Train_Acc = np.mean([acc_lst[-1] for acc_lst in k_train_acc_lst])
k_Train_Loss = np.mean([loss_lst[-1] for loss_lst in k_train_loss_lst])
k_Test_Acc = np.mean([acc_lst[-1] for acc_lst in k_test_acc_lst])
k_Test_Loss = np.mean([loss_lst[-1] for loss_lst in k_test_loss_lst])
print('Train_Acc: {:.4f}%, Train_Loss: {:.4f} \n Test_Acc: {:.4f}%, Test_Loss: {:.4f}'.format(
    k_Train_Acc, k_Train_Loss, k_Test_Acc, k_Test_Loss))
