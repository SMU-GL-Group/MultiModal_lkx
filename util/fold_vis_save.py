#coding=utf-8
"""
K折交叉验证训练结果可视化及保存。
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import xlwt
import pandas as pd
from util import visualization as vis
import logging
import os
import shutil
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import json
from datetime import datetime

join = os.path.join


# 保存准确率与损失
def save_json(data, main_name, save_name, path):
    if save_name is not None:
        save_path = f'{path}/结果acc_loss/{save_name}_{main_name}.json'
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f'数据保存成功：{save_path}')


# 确保指标数组长度等于n_classes
def pad_to_n_classes(arr, n):
    if len(arr) < n:
        return np.pad(arr, (0, n - len(arr)), 'constant', constant_values=0.0)
    return arr

def write2xlsx(wb, n_classes, savename, k, test_acc_lst, roc_auc, CM, Actuals, Preds, spe):
    """
    将结果写入.xlsx文件。
    :param wb: 创建的工作表.xlsx文件
    :param n_classes: 分类类别数量
    :param savename: 保存名
    :param k: 第k折
    :param test_acc_lst:测试准确率列表
    :param roc_auc: auc值
    :param CM: 混淆矩阵
    :param Actuals: 真值数组
    :param Preds: 模型预测数组
    :param spe: 特异性值
    :return:
    """
    d = 4 # 保留的小数位
    # 数据汇总
    ws = wb.add_sheet('{}_k={}'.format(savename.split('_')[0], k))
    # 准确率
    ws.write(0, 0, '测试集准确率')  # 行，列
    ws.write(1, 0, round(float(test_acc_lst[-1]), d))
    # AUC
    ws.write(0, 1, 'AUC')
    for row in range(n_classes):
        ws.write(row + 1, 1, round(float(roc_auc[row]), d))
    # 混淆矩阵
    ws.write(0, 2, '混淆矩阵')
    for row in range(n_classes):
        for col in range(n_classes):
            ws.write(row + 1, col + 2, int(CM[row, col])) # float(CM[row,col])
    # Precision
    precision = precision_score(y_true=Actuals, y_pred=Preds, average=None, zero_division=0)
    precision = pad_to_n_classes(precision, n_classes)
    ws.write(0, 5, 'Precision')
    for row in range(n_classes):
        ws.write(row + 1, 5, round(float(precision[row]), d))
    # Recall
    recall = recall_score(y_true=Actuals, y_pred=Preds, average=None, zero_division=0)
    recall = pad_to_n_classes(recall, n_classes)
    ws.write(0, 6, 'Recall')
    for row in range(n_classes):
        ws.write(row + 1, 6, round(float(recall[row]), d))
    # Specificity
    spe = pad_to_n_classes(spe, n_classes)
    ws.write(0, 7, 'Specificity')
    for row in range(n_classes):
        ws.write(row + 1, 7, round(float(spe[row]), d))
    # F1-score
    f1 = f1_score(y_true=Actuals, y_pred=Preds, average=None)
    f1 = pad_to_n_classes(f1, n_classes)
    ws.write(0, 8, 'F1-score')
    for row in range(n_classes):
        ws.write(row + 1, 8, round(float(f1[row]), d))


def write2xlsx_multilabel(wb, savename, k,
                          test_hamming_loss,
                          prec_macro, rec_macro, f1_macro,
                          prec_per_cls, rec_per_cls, f1_per_cls, auc_per_cls):
    ws = wb.add_sheet(f'{savename.split("_")[0]}_k={k}')
    ws.write(0, 0, 'Hamming Loss')
    ws.write(1, 0, round(test_hamming_loss, 4))

    ws.write(0, 1, 'Macro Precision')
    ws.write(1, 1, round(prec_macro, 4))
    ws.write(0, 2, 'Macro Recall')
    ws.write(1, 2, round(rec_macro, 4))
    ws.write(0, 3, 'Macro F1')
    ws.write(1, 3, round(f1_macro, 4))

    # Per-class metrics starting from col=5
    for i, (p, r, f, a) in enumerate(zip(prec_per_cls, rec_per_cls, f1_per_cls, auc_per_cls)):
        ws.write(0, 5 + i * 4, f'Class{i}_P')
        ws.write(1, 5 + i * 4, round(p, 4))
        ws.write(0, 6 + i * 4, f'Class{i}_R')
        ws.write(1, 6 + i * 4, round(r, 4))
        ws.write(0, 7 + i * 4, f'Class{i}_F1')
        ws.write(1, 7 + i * 4, round(f, 4))
        ws.write(0, 8 + i * 4, f'Class{i}_AUC')
        ws.write(1, 8 + i * 4, round(a, 4) if not np.isnan(a) else 'N/A')


def model_save(save_dir, epoch, model, optimizer, scheduler, k, DataParallel=None):
    """
    保存模型。
    :param save_dir: 保存到的目录。
    :param epoch: 当前轮次。
    :param model: 模型。
    :param optimizer: 优化器。
    :param scheduler: 学习率调度器。
    :param k: 第k折。
    :param DataParallel: 是否使用了分布式训练。
    :return:
    """
    file_name = os.path.join(save_dir, f'last_model_k={k}.pth')
    save_path = os.path.join(save_dir, f'last_model_k={k}.pt')
    if DataParallel: # todo 用到nn.DataParallel时保存模型的方式
        torch.save({
        'epoch': epoch,
        'state_dict': model.module.state_dict(),
        'optim_dict': optimizer.state_dict(),
        'scheduler_dict': scheduler.state_dict(),
    },
        file_name)
        torch.save(model.module, save_path)
    else:
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'scheduler_dict': scheduler.state_dict(),
        },
            file_name)
        torch.save(model, save_path)  # 同时要保留模型.py文件


# 保存配置文件
def save_config(config_path, save_dir, savename='config'):
    """
    将配置文件保存为 YAML 格式，文件名包含当前时间。
    :param config_path: 配置字典的路径。
    :param save_dir: 保存目录。
    :param savename: 文件名前缀，默认为 "config"。
    """
    # 获取当前时间并格式化为字符串
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 构造文件名
    filename = f"{savename}_{current_time}.yaml"
    # 构造完整的保存路径
    save_path = os.path.join(save_dir, filename)

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    shutil.copy(config_path, save_path)

    return save_path


def fold_visualization_and_save(k_idx,
                                args,
                                wb,
                                train_loss_lst,
                                train_acc_lst,
                                test_loss_lst,
                                test_acc_lst,
                                Test_result):
    """
    一次性完成：
      1. 绘制并保存 loss/acc 曲线
      2. 计算 AUC、绘制 ROC 曲线并保存
      3. 绘制混淆矩阵（带百分比）
      4. 打印 classification-report
      5. 将关键指标写入 xlsx（返回 wb，主函数最后统一 save）

    参数
    ----
    k_idx : int               当前折编号
    args  : argparse.Namespace  全局超参/路径
    config: dict              yaml 配置（含 cls_list）
    train/test_loss_lst, train/test_acc_lst: list  训练过程曲线
    Test_result: dict         test_one_epoch 返回值，含 actuals/preds/scores
    """
    savepath = args.savepath
    savename = args.savename
    cls_list = args.cls_list
    num_classes = args.num_classes
    max_epoch = getattr(args, 'max_epoch', None)

    image_result = os.path.join(savepath, '结果图')
    os.makedirs(image_result, exist_ok=True)
    # 1. loss / acc 曲线
    if max_epoch:
        vis.Loss_Acc(savepath, max_epoch, savename,
                    train_loss=train_loss_lst, test_loss=test_loss_lst,
                    train_acc=train_acc_lst, test_acc=test_acc_lst, k=k_idx)

    # 2. ROC / AUC
    npy_result = os.path.join(savepath, '结果acc_loss')
    os.makedirs(npy_result, exist_ok=True)

    score_array = Test_result['scores']
    actuals = Test_result['actuals']
    actuals_onehot = np.eye(num_classes)[actuals]

    np.save(os.path.join(npy_result, f'{savename}_Score_array_k={k_idx}.npy'), score_array)
    np.save(os.path.join(npy_result, f'{savename}_Label_onehot_k={k_idx}.npy'), actuals_onehot)

    roc_auc, spe = vis.plot_AUC(y_scores=score_array,
                                y_test=actuals_onehot,
                                n_classes=num_classes,
                                path=savepath,
                                k=k_idx,
                                cls=cls_list,
                                savename=savename)

    # 3. 混淆矩阵
    preds = Test_result['preds']
    CM = confusion_matrix(y_true=actuals, y_pred=preds, labels=range(num_classes))
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
                xticklabels=cls_list, yticklabels=cls_list,
                annot_kws={'size': 25})
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(f'{savename}_k={k_idx}', fontsize=15)
    plt.xlabel('Pred', fontsize=25)
    plt.ylabel('True', fontsize=25)
    plt.tight_layout()
    plt.savefig(os.path.join(image_result, f'{savename}_k={k_idx}_CM.png'))
    plt.show()
    plt.close()

    # 4. 分类报告
    # 确保类别数量匹配
    if len(cls_list) != num_classes:
        raise ValueError(f"类别数量不匹配: cls_list有{len(cls_list)}个类别，但num_classes={num_classes}")
    report_dict = classification_report(y_true=actuals,
                                        y_pred=preds,
                                        labels=range(num_classes),  # 添加这一行
                                        target_names=cls_list,
                                        output_dict=True,
                                        zero_division=0)
    report_df = pd.DataFrame(report_dict).T.round(4)
    logging.info(f'\nk={k_idx}\n{report_df.to_string()}')

    # 5. xlsx 汇总（返回 wb，主函数最后统一 save）
    write2xlsx(wb, num_classes, savename,
               k_idx, test_acc_lst, roc_auc,
               CM.astype(int),
               actuals, preds, spe)
    return wb

