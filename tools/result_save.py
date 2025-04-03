import numpy as np
import os
import torch
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import json


# 保存准确率与损失
def save_json(data, main_name, save_name):
    if save_name is not None:
        cur_path = os.getcwd()
        save_path = f'{cur_path}/结果acc_loss/{save_name}_{main_name}.json'
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f'数据保存成功：{save_path}')


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
            ws.write(row + 1, col + 2, CM[row, col]) # float(CM[row,col])
    # Precision
    ws.write(0, 5, 'Precision')
    for row in range(n_classes):
        ws.write(row + 1, 5, round(float(precision_score(y_true=Actuals, y_pred=Preds, average=None, zero_division=0)[row]), d))
    # Recall
    ws.write(0, 6, 'Recall')
    for row in range(n_classes):
        ws.write(row + 1, 6, round(float(recall_score(y_true=Actuals, y_pred=Preds, average=None, zero_division=0)[row]), d))
    # Specificity
    ws.write(0, 7, 'Specificity')
    for row in range(n_classes):
        ws.write(row + 1, 7, round(float(spe[row]), d))
    # F1-score
    ws.write(0, 8, 'F1-score')
    for row in range(n_classes):
        ws.write(row + 1, 8, round(float(f1_score(y_true=Actuals, y_pred=Preds, average=None)[row]), d))


def model_save(args, epoch, model, optimizer, k, DataParallel=None):
    """
    保存模型。
    :param args:模型参数
    :param epoch:
    :param model:
    :param optimizer:
    :param k: k折交叉验证索引
    :return:
    """
    file_name = os.path.join(args.savepath, f'last_model_k={k}.pth')
    model_save = os.path.join(args.savepath, f'last_model_k={k}.pt')
    if DataParallel: # todo 用到nn.DataParallel时保存模型的方式
        torch.save({
        'epoch': epoch,
        'state_dict': model.module.state_dict(),
        'optim_dict': optimizer.state_dict(),
    },
        file_name)
        torch.save(model.module, model_save)
    else:
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
        },
            file_name)
        torch.save(model, model_save)  # 同时要保留模型.py文件

