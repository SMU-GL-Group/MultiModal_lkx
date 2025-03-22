import numpy as np
import os
import xlwt
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score


# 保存准确率与损失
def save_npy(nparr, main_name, save_name):
    """
    :param nparr: 需要保存的数组
    :param main_name: 保存的名字（说明）
    :param save_name: 保存的名字
    """
    if save_name is not None:
        cur_path = os.getcwd()
        np.save('{}/结果acc_loss/{}_{}'.format(cur_path, save_name, main_name), nparr)
        print('准确率或损失保存成功：{}/结果acc_loss/{}_{}.npy'.format(cur_path,save_name, main_name))


def write2xlsx(wb, args, k, test_acc_lst, roc_auc, CM, Actuals, Preds, spe):
    """
    将结果写入.xlsx文件。
    :param wb: 创建的工作表.xlsx文件
    :param args: 模型配置参数
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
    ws = wb.add_sheet('{}_k={}'.format(args.savename.split('_')[0], k + 1))
    # 准确率
    ws.write(0, 0, '测试集准确率')  # 行，列
    ws.write(1, 0, round(float(test_acc_lst[-1]), 3))
    # AUC
    ws.write(0, 1, 'AUC')
    for row in range(3):
        ws.write(row + 1, 1, round(float(roc_auc[row]), 4))
    # 混淆矩阵
    ws.write(0, 2, '混淆矩阵')
    for row in range(3):
        for col in range(3):
            ws.write(row + 1, col + 2, CM[row, col])
    # Precision
    ws.write(0, 5, 'Precision')
    for row in range(3):
        ws.write(row + 1, 5, round(float(precision_score(y_true=Actuals, y_pred=Preds, average=None)[row]), d))
    # Recall
    ws.write(0, 6, 'Recall')
    for row in range(3):
        ws.write(row + 1, 6, round(float(recall_score(y_true=Actuals, y_pred=Preds, average=None)[row]), d))
    # Specificity
    ws.write(0, 7, 'Specificity')
    for row in range(3):
        ws.write(row + 1, 7, round(float(spe[row]), d))
    # F1-score
    ws.write(0, 8, 'F1-score')
    for row in range(3):
        ws.write(row + 1, 8, round(float(f1_score(y_true=Actuals, y_pred=Preds, average=None)[row]), d))


