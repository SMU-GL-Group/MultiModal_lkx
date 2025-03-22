#coding=utf-8
import os.path
from torchvision import models, transforms
from sklearn.model_selection import KFold
import numpy as np
import argparse
from MMdataset import MMdata_EMIL


# ======================= 参数设置 ======================= #
parser = argparse.ArgumentParser()

parser.add_argument('--Description',
                    default='划分五折交叉验证数据集的index，存为npy。')
parser.add_argument('--data_path', default='/public/longkaixing/CrossModalScale/datasets/MN分期-三模态全-前三期', type=str)
parser.add_argument('--kfold', default=5, type=int)
parser.add_argument('--reshape', default=[224, 224])
parser.add_argument('--modal', default='TEM', type=str, help='所用模态')

args = parser.parse_args()
args.transforms = transforms.Compose([
    transforms.Resize(args.reshape, antialias=True),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ======================= 载入数据集 ======================= #
data_set = MMdata_EMIL(transform=args.transforms, root=args.data_path, modal=args.modal)
print('数据集：', args.data_path)

# ======================= K折交叉验证 ======================= #
train_loader_lst, test_loader_lst = [], []

kf = KFold(n_splits=args.kfold, shuffle=True, random_state=10)
idx = 0
for train_index, test_index in kf.split(data_set):
    idx += 1
    np.random.shuffle(train_index)
    np.random.shuffle(test_index)
    np.save('{}/数据集交叉划分index/MN分期-三模态全-前三期_{}_k={}'.format(os.getcwd(), 'train_index', idx), train_index)
    np.save('{}/数据集交叉划分index/MN分期-三模态全-前三期_{}_k={}'.format(os.getcwd(), 'test_index', idx), test_index)

print('done')
