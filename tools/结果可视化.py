import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ======================= 计算多分类k折交叉验证均值 ======================= #
def complete_AUC_k(y_scores_lst, y_test_lst, args, kfold):
    k_tpr = [] # 存放每折ROC
    k_fpr = []
    mean_fpr = np.linspace(0,1,100)
    for k in range(kfold):
        y_scores = y_scores_lst[k]
        y_test = y_test_lst[k]
        # AUC of each classes
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(args.num_cls): # 先对类别做平均，再对折做平均
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_scores[:, i])

        # AUC of macro-average # 多分类宏平均：先算各个类别的ROC，再进行算数平均，适合各类别样变量不平衡情况
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(args.num_cls)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(args.num_cls):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= args.num_cls
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        print('AUC macro-average: {:.4f}'.format(roc_auc["macro"]))
        k_fpr.append(all_fpr)
        k_tpr.append(np.interp(mean_fpr, all_fpr, mean_tpr)) # 用插值法平滑ROC曲线
        k_tpr[-1][0] = 0.0
        # # AUC of micro-average # 多分类微平均：先算各类的fp、tp、fn、tn等，再算总ROC，适合类别样变均衡情况
        # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_scores.ravel())
        # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # print('AUC micro-average: {:.4f}'.format(roc_auc["micro"]))

    # 计算k折的平均macro-average
    k_mean_tpr = np.mean(k_tpr, axis=0)
    k_mean_tpr[-1] = 1.0
    k_mean_auc = auc(mean_fpr, k_mean_tpr) # 计算k折AUC均值

    return mean_fpr, k_mean_tpr, k_mean_auc


parser = argparse.ArgumentParser()
parser.add_argument('--num_cls', default=3, type=int, help='分为几类')
parser.add_argument('--savename', default='A')
args = parser.parse_args()
# # ============================= AUC、ROC ============================= #
Score_array_lst, Label_onehot_lst = [], []
name1 = '11-26_光镜（PASM）+荧光+电镜-301+270例_MDL_IIA(光镜+荧光+电镜固一)'
for k in range(1,6):
    Score_array_lst.append(np.load('/public/longkaixing/CrossModalScale/MyModel/MDL-IIA/结果acc_loss/{}_Score_array_k={}.npy'.format(name1, k))[:])
    Label_onehot_lst.append(np.load('/public/longkaixing/CrossModalScale/MyModel/MDL-IIA/结果acc_loss/{}_Label_onehot_k={}.npy'.format(name1, k))[:])
mean_fpr1, k_mean_tpr1, k_mean_auc1 = complete_AUC_k(y_scores_lst=Score_array_lst, y_test_lst=Label_onehot_lst, args=args, kfold=5)


Score_array_lst, Label_onehot_lst = [], []
name2 = '12-01_光镜（PASM）+荧光+电镜-301+270例_mmFormer_cat(光镜+荧光+电镜固一)'
for k in range(1,6):
    Score_array_lst.append(np.load('/public/longkaixing/CrossModalScale/MyModel/mmFormer/结果acc_loss/{}_Score_array_k={}.npy'.format(name2, k))[:])
    Label_onehot_lst.append(np.load('/public/longkaixing/CrossModalScale/MyModel/mmFormer/结果acc_loss/{}_Label_onehot_k={}.npy'.format(name2, k))[:])
mean_fpr2, k_mean_tpr2, k_mean_auc2 = complete_AUC_k(y_scores_lst=Score_array_lst, y_test_lst=Label_onehot_lst, args=args, kfold=5)


Score_array_lst, Label_onehot_lst = [], []
name3 = '09-26_光镜（PASM）+荧光+电镜-301+270例_MultiScaleAttn_3M3Loss_240926(TEM固定一张)'
for k in range(1,6):
    Score_array_lst.append(np.load('/public/longkaixing/CrossModalScale/MyModel/MSAN/结果acc_loss/{}_Score_array_k={}.npy'.format(name3, k))[:])
    Label_onehot_lst.append(np.load('/public/longkaixing/CrossModalScale/MyModel/MSAN/结果acc_loss/{}_Label_onehot_k={}.npy'.format(name3, k))[:])
mean_fpr3, k_mean_tpr3, k_mean_auc3 = complete_AUC_k(y_scores_lst=Score_array_lst, y_test_lst=Label_onehot_lst, args=args, kfold=5)


Score_array_lst, Label_onehot_lst = [], []
name4 = '09-03_光镜（PASM）+荧光+电镜-301+270例_AGGN_240903(光镜+荧光+电镜)'
for k in range(1,6):
    Score_array_lst.append(np.load('/public/longkaixing/CrossModalScale/MyModel/AGGN/结果acc_loss/{}_Score_array_k={}.npy'.format(name4, k))[:])
    Label_onehot_lst.append(np.load('/public/longkaixing/CrossModalScale/MyModel/AGGN/结果acc_loss/{}_Label_onehot_k={}.npy'.format(name4, k))[:])
mean_fpr4, k_mean_tpr4, k_mean_auc4 = complete_AUC_k(y_scores_lst=Score_array_lst, y_test_lst=Label_onehot_lst, args=args, kfold=5)

Score_array_lst, Label_onehot_lst = [], []
name5 = '08-06_光镜（PASM）+荧光+电镜-301+270例_CrossMIL_TEMq_OMkv_IMkv_3Loss_240806(电镜q+光镜kv+荧光kv,wLoss)'
for k in range(1,6):
    Score_array_lst.append(np.load('/public/longkaixing/CrossModalScale/结果acc_loss/{}_Score_array_k={}.npy'.format(name5, k))[:])
    Label_onehot_lst.append(np.load('/public/longkaixing/CrossModalScale/结果acc_loss/{}_Label_onehot_k={}.npy'.format(name5, k))[:])
mean_fpr5, k_mean_tpr5, k_mean_auc5 = complete_AUC_k(y_scores_lst=Score_array_lst, y_test_lst=Label_onehot_lst, args=args, kfold=5)

# Score_array_lst, Label_onehot_lst = [], []
# name5 = '10-23_光镜+荧光+电镜-301例-3类_resnet_TEM_EMIL_wcat_frozen(光镜+荧光+电镜random)'
# for k in range(1,6):
#     Score_array_lst.append(np.load('/public/longkaixing/MMF/MyMethod/结果acc_loss/{}_Score_array_k={}.npy'.format(name5, k))[:])
#     Label_onehot_lst.append(np.load('/public/longkaixing/MMF/MyMethod/结果acc_loss/{}_Label_onehot_k={}.npy'.format(name5, k))[:])
# mean_fpr5, k_mean_tpr5, k_mean_auc5 = complete_AUC_k(y_scores_lst=Score_array_lst, y_test_lst=Label_onehot_lst, args=args, kfold=5)
#
#
# Score_array_lst, Label_onehot_lst = [], []
# name6 = '10-25_光镜+荧光+电镜-301例-3类_resnet_TEM_IMIL_frozen(光镜+荧光+电镜IMIL)'
# for k in range(1,6):
#     Score_array_lst.append(np.load('/public/longkaixing/MMF/MyMethod/结果acc_loss/{}_Score_array_k={}.npy'.format(name6, k))[:])
#     Label_onehot_lst.append(np.load('/public/longkaixing/MMF/MyMethod/结果acc_loss/{}_Label_onehot_k={}.npy'.format(name6, k))[:])
# mean_fpr6, k_mean_tpr6, k_mean_auc6 = complete_AUC_k(y_scores_lst=Score_array_lst, y_test_lst=Label_onehot_lst, args=args, kfold=5)

# 绘制ROC
fig, ax = plt.subplots(figsize=(8, 6), dpi=100) # 画布fig，绘图区ax
plt.rc('font',family='Times New Roman') # 修改全局字体
#cls = ['SLE', 'MN', 'IgA']
ax.plot(mean_fpr1, k_mean_tpr1, label='MDL-IIA (AUC={:.4})'.format(k_mean_auc1), linestyle='solid', color='green',linewidth=2)
ax.plot(mean_fpr2, k_mean_tpr2, label='mmFormer (AUC={:.4})'.format(k_mean_auc2), linestyle='solid', color='goldenrod',linewidth=2)
ax.plot(mean_fpr3, k_mean_tpr3, label='MSAN (AUC={:.4})'.format(k_mean_auc3), linestyle='solid', color='blue',linewidth=2)
ax.plot(mean_fpr4, k_mean_tpr4, label='AGGN (AUC={:.4})'.format(k_mean_auc4), linestyle='solid', color='purple',linewidth=2)
ax.plot(mean_fpr5, k_mean_tpr5, label='Ours (AUC={:.4})'.format(k_mean_auc5), linestyle='solid', color='red',linewidth=2)

# ax.plot(mean_fpr1, k_mean_tpr1, label='OM&IM (AUC={:.4})'.format(k_mean_auc1), linestyle='solid', color='red',linewidth=2)
# ax.plot(mean_fpr2, k_mean_tpr2, label='IM&TEM (AUC={:.4})'.format(k_mean_auc2), linestyle='solid', color='goldenrod',linewidth=2)
# ax.plot(mean_fpr3, k_mean_tpr3, label='OM&TEM (AUC={:.4})'.format(k_mean_auc3), linestyle='solid', color='blue',linewidth=2)
# ax.plot(mean_fpr4, k_mean_tpr4, label='OM&IM&TEM (AUC={:.4})'.format(k_mean_auc4), linestyle='solid', color='green',linewidth=2)

# ax.plot(mean_fpr5, k_mean_tpr5, label='OM&IM&TEM(含标尺)(random) AUC={:.4})'.format(k_mean_auc5), linestyle='solid', color='cyan',linewidth=2)
# ax.plot(mean_fpr6, k_mean_tpr6, label='OM&IM&TEM(含标尺)(IMIL) AUC={:.4})'.format(k_mean_auc5), linestyle='solid', color='pink',linewidth=2)
#plt.title('{}  ROC'.format(args.savename))
# 去掉右上坐标框线
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# 设置x,y轴长度
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.xticks(fontname='Times New Roman',fontsize=18)
plt.yticks(fontname='Times New Roman',fontsize=18)
ax.set_xlabel('False positive rate', fontdict={'family': 'Times New Roman', 'size': 25})
ax.set_ylabel('True positive rate', fontdict={'family': 'Times New Roman', 'size': 25})
ax.text(-0.125,1,'',fontdict={'family': 'Times New Roman', 'weight':'bold', 'size': 25})
ax.legend(loc='best', fontsize=14)
ax.plot([0, 1], linestyle='--', color='black') # 绘制对角线


if args.savename is not None:
    cur_path = os.getcwd()
    plt.savefig('{}/{}_AUC.png'.format(cur_path, args.savename + '_k=' + str(k)), dpi=1500)
plt.show()

print('{}/{}'.format(cur_path,args.sacename))

