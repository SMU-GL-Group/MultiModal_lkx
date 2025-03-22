#coding=utf-8
import os
import random

import pandas as pd
import torch
from torch.utils.data import Dataset
join = os.path.join
from PIL import Image
from torchvision import transforms
from sklearn.utils import shuffle


# ======================= 三模态共用(电镜基于嵌入/在线提取特征) ======================= #
class MMdata_EMIL(Dataset):
    def __init__(self, root='', transform=transforms.ToTensor(), modal=''):
        self.transforms = transform
        self.modal = modal

        IM_path_lst = []
        OM_path_lst = []
        TEM_path_lst = []
        bag_lst = []
        label_lst = []
        for cls in os.listdir(root):
            for patient in os.listdir(join(root, cls)):
                Inst_lst = []
                # 一期=0，二期=1，三期=2，四期=3
                label_lst.append((os.listdir(root)).index(cls))  # MN=0,IgAN=1,LN=2 # others=0,MN=1 # 每个病人记录对应标签 'SLE=0,MN=1,IgA=2' # todo 和features的.csv的不一样呀
                for img in os.listdir(join(root, cls, patient)):
                    if 'IM' in modal and img.startswith('IM'):
                        img_path = join(root, cls, patient, img)
                        IM_path_lst.append(img_path)

                    elif 'OM_' in modal and img.startswith('OM_'):
                        img_path = join(root, cls, patient, img)
                        OM_path_lst.append(img_path)

                    elif 'OM-ori' in modal and img.startswith('OM-ori'):
                        img_path = join(root, cls, patient, img)
                        OM_path_lst.append(img_path)

                    # elif 'TEM' in modal and img.startswith('TEM'):
                    else:
                        img_path = join(root, cls, patient, img)
                        TEM_path_lst.append(img_path)
                        Inst_lst.append(img_path)
                bag_lst.append(Inst_lst)
        self.IM_path_lst = IM_path_lst
        self.OM_path_lst = OM_path_lst
        self.TEM_path_lst = TEM_path_lst
        self.bag_lst = bag_lst
        self.label_lst = label_lst

    def __getitem__(self, idx):
        y = torch.tensor(self.label_lst[idx])
        if 'TEM' in self.modal: # 含电镜
            bag = self.bag_lst[idx]
            bag_tensor = []
            for k in range(len(bag)):
                # print("{}".format(bag[k]))
                try:
                    bag_tensor.append(self.transforms(Image.open(bag[k]).convert('RGB')))
                except:
                    print('wrong')
                # bag_tensor.append(self.transforms(Image.open(bag[k]).convert('RGB')))

            if len(self.modal.split('+')) == 1: # 单模态电镜

                return bag, bag_tensor, [], [], y

            elif len(self.modal.split('+')) == 2: # 双模态含电镜
                if 'IM' in self.modal:
                    img = Image.open(self.IM_path_lst[idx])
                    IM_tensor = self.transforms(img)

                    return bag_tensor, [], IM_tensor, y
                elif 'OM' in self.modal:
                    img = Image.open(self.OM_path_lst[idx])
                    OM_tensor = self.transforms(img)

                    return bag_tensor, OM_tensor, [], y

            elif len(self.modal.split('+')) == 3: # 三模态含电镜
                IM_img = Image.open(self.IM_path_lst[idx])
                IM_tensor = self.transforms(IM_img)
                OM_img = Image.open(self.OM_path_lst[idx])
                OM_tensor = self.transforms(OM_img)

                return bag_tensor, OM_tensor, IM_tensor, y

        else: # 无电镜
            if len(self.modal.split('+')) == 1: # 单模态非电镜
                if 'IM' in self.modal:
                    img = Image.open(self.IM_path_lst[idx])
                    img_tensor = self.transforms(img)
                elif 'OM' in self.modal:
                    img = Image.open(self.OM_path_lst[idx])
                    img_tensor = self.transforms(img)

                return img_tensor, y
            elif len(self.modal.split('+')) == 2: # 双模态无电镜
                OM_img = Image.open(self.OM_path_lst[idx])
                OM_tensor = self.transforms(OM_img)
                IM_img = Image.open(self.IM_path_lst[idx])
                IM_tensor = self.transforms(IM_img)

                return [], OM_tensor, IM_tensor, y

    def __len__(self): # 决定idx的范围
        return len(self.label_lst)


# ======================= 查看挑选器挑了哪些电镜图 ======================= #
class MMdata_iclassifier(Dataset):
    def __init__(self, root='', transform=transforms.ToTensor(), modal=''):
        self.transforms = transform
        self.modal = modal

        IM_path_lst = []
        OM_path_lst = []
        TEM_path_lst = []
        bag_lst = []
        label_lst = []
        for cls in os.listdir(root):
            for patient in os.listdir(join(root, cls)):
                Inst_lst = []
                label_lst.append((os.listdir(root)).index(cls))  # others=0,MN=1 # 每个病人记录对应标签 'SLE=0,MN=1,IgA=2' # todo 和features的.csv的不一样呀
                for img in os.listdir(join(root, cls, patient)):
                    if 'IF' in modal and img.startswith('IF'):
                        img_path = join(root, cls, patient, img)
                        IM_path_lst.append(img_path)

                    elif 'OM' in modal and img.startswith('OM'):
                        img_path = join(root, cls, patient, img)
                        OM_path_lst.append(img_path)

                    elif 'TEM' in modal and img.startswith('TEM'):
                        img_path = join(root, cls, patient, img)
                        TEM_path_lst.append(img_path)
                        Inst_lst.append(img_path)
                bag_lst.append(Inst_lst)
        self.IM_path_lst = IM_path_lst
        self.OM_path_lst = OM_path_lst
        self.TEM_path_lst = TEM_path_lst
        self.bag_lst = bag_lst
        self.label_lst = label_lst

    def __getitem__(self, idx):
        y = torch.tensor(self.label_lst[idx])
        bag_image = []
        if 'TEM' in self.modal: # 含电镜
            bag = self.bag_lst[idx]
            bag_tensor = []
            for k in range(len(bag)):
                bag_tensor.append(self.transforms(Image.open(bag[k])))
                bag_image.append(bag[k])
            if len(self.modal.split('+')) == 1: # 单模态电镜

                return bag_tensor, [], [], y, bag_image

            elif len(self.modal.split('+')) == 2: # 双模态含电镜
                if 'IF' in self.modal:
                    img = Image.open(self.IM_path_lst[idx])
                    IM_tensor = self.transforms(img)

                    return bag_tensor, [], IM_tensor, y
                elif 'OM' in self.modal:
                    img = Image.open(self.OM_path_lst[idx])
                    OM_tensor = self.transforms(img)

                    return bag_tensor, OM_tensor, [], y

            elif len(self.modal.split('+')) == 3: # 三模态含电镜
                IM_img = Image.open(self.IM_path_lst[idx])
                IM_tensor = self.transforms(IM_img)
                OM_img = Image.open(self.OM_path_lst[idx])
                OM_tensor = self.transforms(OM_img)

                return bag_tensor, OM_tensor, IM_tensor, y

        else: # 无电镜
            if len(self.modal.split('+')) == 1: # 单模态非电镜
                if 'IF' in self.modal:
                    img = Image.open(self.IM_path_lst[idx])
                    img_tensor = self.transforms(img)
                elif 'OM' in self.modal:
                    img = Image.open(self.OM_path_lst[idx])
                    img_tensor = self.transforms(img)

                return img_tensor, y
            elif len(self.modal.split('+')) == 2: # 双模态无电镜
                OM_img = Image.open(self.OM_path_lst[idx])
                OM_tensor = self.transforms(OM_img)
                IM_img = Image.open(self.IM_path_lst[idx])
                IM_tensor = self.transforms(IM_img)

                return [], OM_tensor, IM_tensor, y

    def __len__(self): # 决定idx的范围
        return len(self.label_lst)


# ======================= 三模态共用(等比例采样) ======================= #
class MMdata_weight(Dataset):
    def __init__(self, root='', transform=transforms.ToTensor(), modal='', n_sampler=int()):
        self.transforms = transform
        self.modal = modal
        self.root = root

        IM_path_lst = []
        OM_path_lst = []
        TEM_path_lst = []
        bag_lst = []
        label_lst = []
        for cls in os.listdir(root):
            # 打乱病人样本并抽取固定数量的样本
            sample_lst = os.listdir(join(root, cls))
            random.seed(10)
            random.shuffle(sample_lst)
            sample_lst = sample_lst[:n_sampler]
            for patient in sample_lst: # 一-三期=0-2 # MN=0,IgAN=1,LN=2 # others=0, MN=1 # others=0, MN=1, IgAN=2 # others=0, MN=1, IgAN=2, LN=3
                label_lst.append((os.listdir(root)).index(cls))  # 每个病人记录对应标签 'SLE=0,MN=1,IgA=2'
                for img in os.listdir(join(root, cls, patient)):
                    if 'IM' in modal and img.startswith('IM'):
                        img_path = join(root, cls, patient, img)
                        IM_path_lst.append(img_path)

                    elif 'OM' in modal and img.startswith('OM'):
                        img_path = join(root, cls, patient, img)
                        OM_path_lst.append(img_path)

                    elif 'TEM' in modal and img.startswith('TEM'):
                        img_path = join(root, cls, patient, img)
                        TEM_path_lst.append(img_path)
                bag_lst.append(join(root, cls, patient))
        self.IM_path_lst = IM_path_lst
        self.OM_path_lst = OM_path_lst
        self.TEM_path_lst = TEM_path_lst
        self.bag_lst = bag_lst
        self.label_lst = label_lst

    def __getitem__(self, idx):
        y = torch.tensor(self.label_lst[idx])
        if 'TEM' in self.modal: # 含电镜
            img_path = os.listdir(self.bag_lst[idx])
            TEM_path = [x for x in img_path if x.startswith('TEM')]
            TEM = random.choice(TEM_path) # 随机挑选
            cls, patient = TEM.split('_')[1], TEM.split('_')[2]
            img = Image.open(os.path.join(self.root, cls, patient, TEM))
            TEM_tensor = self.transforms(img)

            if len(self.modal.split('+')) == 1: # 单模态电镜

                return TEM_tensor, y

            elif len(self.modal.split('+')) == 2: # 双模态含电镜
                if 'IM' in self.modal:
                    img = Image.open(self.IM_path_lst[idx])
                    IM_tensor = self.transforms(img)

                    return TEM_tensor, [], IM_tensor, y
                elif 'OM' in self.modal:
                    img = Image.open(self.OM_path_lst[idx])
                    OM_tensor = self.transforms(img)

                    return TEM_tensor, OM_tensor, [], y

            elif len(self.modal.split('+')) == 3: # 三模态含电镜
                IM_img = Image.open(self.IM_path_lst[idx])
                IM_tensor = self.transforms(IM_img)
                OM_img = Image.open(self.OM_path_lst[idx])
                OM_tensor = self.transforms(OM_img)

                return TEM_tensor, OM_tensor, IM_tensor, y

        else: # 无电镜
            if len(self.modal.split('+')) == 1: # 单模态非电镜
                if 'IM' in self.modal:
                    img = Image.open(self.IM_path_lst[idx])
                    img_tensor = self.transforms(img)
                elif 'OM' in self.modal:
                    img = Image.open(self.OM_path_lst[idx])
                    img_tensor = self.transforms(img)

                return img_tensor, y
            elif len(self.modal.split('+')) == 2: # 双模态无电镜
                OM_img = Image.open(self.OM_path_lst[idx])
                OM_tensor = self.transforms(OM_img)
                IM_img = Image.open(self.IM_path_lst[idx])
                IM_tensor = self.transforms(IM_img)

                return [], OM_tensor, IM_tensor, y

    def __len__(self): # 决定idx的范围
        return len(self.label_lst)

# ======================= TEM专用随机采样 ======================= #
class TEM_random(Dataset):
    def __init__(self, root='', transform=transforms.ToTensor()):
        self.transforms = transform

        TEM_path_lst = []
        bag_lst = []
        label_lst = []
        for cls in os.listdir(root):
            for patient in os.listdir(join(root, cls)):
                Inst_lst = []
                # 一期=0，二期=1，三期=2，四期=3
                label_lst.append((os.listdir(root)).index(cls))  # MN=0,IgAN=1,LN=2 # others=0,MN=1 # 每个病人记录对应标签 'SLE=0,MN=1,IgA=2' # todo 和features的.csv的不一样呀
                for img in os.listdir(join(root, cls, patient)):
                    # if img.startswith('TEM'):
                    img_path = join(root, cls, patient, img)
                    TEM_path_lst.append(img_path)
                    Inst_lst.append(img_path)
                bag_lst.append(Inst_lst)

        self.TEM_path_lst = TEM_path_lst
        self.bag_lst = bag_lst
        self.label_lst = label_lst

    def __getitem__(self, idx):
        y = torch.tensor(self.label_lst[idx])

        bag = self.bag_lst[idx]
        random.seed(42)
        try:
            bag_tensor = self.transforms(Image.open(random.choice(bag)).convert('RGB')) # 随机从包中挑选一张图像
        except:
            print('wrong')

        return bag_tensor, y

    def __len__(self): # 决定idx的范围
        return len(self.label_lst)

# ======================= 多模态(TEM-random) ======================= #
class MMdata_TEMrandom(Dataset):
    def __init__(self, root='', transform=transforms.ToTensor(), modal='', n_sampler=int()):
        self.transforms = transform
        self.modal = modal
        self.root = root

        IM_path_lst = []
        OM_path_lst = []
        TEM_path_lst = []
        bag_lst = []
        label_lst = []
        for cls in os.listdir(root):
            for patient in os.listdir(join(root, cls)): # 一-三期=0-2 # MN=0,IgAN=1,LN=2 # others=0, MN=1 # others=0, MN=1, IgAN=2 # others=0, MN=1, IgAN=2, LN=3
                label_lst.append((os.listdir(root)).index(cls))  # 每个病人记录对应标签 'SLE=0,MN=1,IgA=2'
                for img in os.listdir(join(root, cls, patient)):
                    if 'IM' in modal and img.startswith('IM'):
                        img_path = join(root, cls, patient, img)
                        IM_path_lst.append(img_path)

                    elif 'OM_' in modal and img.startswith('OM_'):
                        img_path = join(root, cls, patient, img)
                        OM_path_lst.append(img_path)

                    elif 'OM-ori' in modal and img.startswith('OM-ori'):
                        img_path = join(root, cls, patient, img)
                        OM_path_lst.append(img_path)

                    elif 'TEM' in modal and img.startswith('TEM'):
                        img_path = join(root, cls, patient, img)
                        TEM_path_lst.append(img_path)
                bag_lst.append(join(root, cls, patient))
        self.IM_path_lst = IM_path_lst
        self.OM_path_lst = OM_path_lst
        self.TEM_path_lst = TEM_path_lst
        self.bag_lst = bag_lst
        self.label_lst = label_lst

    def __getitem__(self, idx):
        y = torch.tensor(self.label_lst[idx])
        if 'TEM' in self.modal: # 含电镜
            img_path = os.listdir(self.bag_lst[idx])
            TEM_path = [x for x in img_path if x.startswith('TEM')]
            TEM = TEM_path[0] # 固定用第一张
            # TEM = random.choice(TEM_path) # 随机挑选一张电镜
            cls, patient = TEM.split('_')[1], TEM.split('_')[2]
            img = Image.open(os.path.join(self.root, cls, patient, TEM))
            TEM_tensor = self.transforms(img)

            if len(self.modal.split('+')) == 1: # 单模态电镜

                return TEM_tensor, y

            elif len(self.modal.split('+')) == 2: # 双模态含电镜
                if 'IM' in self.modal:
                    img = Image.open(self.IM_path_lst[idx])
                    IM_tensor = self.transforms(img)

                    return TEM_tensor, [], IM_tensor, y
                elif 'OM' in self.modal:
                    img = Image.open(self.OM_path_lst[idx])
                    OM_tensor = self.transforms(img)

                    return TEM_tensor, OM_tensor, [], y

            elif len(self.modal.split('+')) == 3: # 三模态含电镜
                IM_img = Image.open(self.IM_path_lst[idx])
                IM_tensor = self.transforms(IM_img)
                OM_img = Image.open(self.OM_path_lst[idx])
                OM_tensor = self.transforms(OM_img)

                return TEM_tensor, OM_tensor, IM_tensor, y

        else: # 无电镜
            if len(self.modal.split('+')) == 1: # 单模态非电镜
                if 'IM' in self.modal:
                    img = Image.open(self.IM_path_lst[idx])
                    img_tensor = self.transforms(img)
                elif 'OM' in self.modal:
                    img = Image.open(self.OM_path_lst[idx])
                    img_tensor = self.transforms(img)

                return img_tensor, y
            elif len(self.modal.split('+')) == 2: # 双模态无电镜
                OM_img = Image.open(self.OM_path_lst[idx])
                OM_tensor = self.transforms(OM_img)
                IM_img = Image.open(self.IM_path_lst[idx])
                IM_tensor = self.transforms(IM_img)

                return [], OM_tensor, IM_tensor, y

    def __len__(self): # 决定idx的范围
        return len(self.label_lst)

# ======================= 多尺度输入 ======================= #
class MultiScaleInput(Dataset):
    def __init__(self, root='', transform=transforms.ToTensor(), modal=''):
        self.resize_512 = transforms.Resize([512, 512])
        self.resize_256 = transforms.Resize([256, 256])
        self.resize_128 = transforms.Resize([128, 128])
        self.resize_64 = transforms.Resize([64, 64])

        self.transforms = transform

        self.modal = modal

        OM_path_lst = []
        label_lst = []
        for cls in os.listdir(root):
            for patient in os.listdir(join(root, cls)):
                Inst_lst = []
                # 一期=0，二期=1，三期=2，四期=3
                label_lst.append((os.listdir(root)).index(cls))  # MN=0,IgAN=1,LN=2 # others=0,MN=1 # 每个病人记录对应标签 'SLE=0,MN=1,IgA=2' # todo 和features的.csv的不一样呀
                for img in os.listdir(join(root, cls, patient)):
                    if 'OM' in modal and img.startswith('OM-ori'):
                        img_path = join(root, cls, patient, img)
                        OM_path_lst.append(img_path)

        self.OM_path_lst = OM_path_lst
        self.label_lst = label_lst

    def __getitem__(self, idx):
        y = torch.tensor(self.label_lst[idx])

        if 'OM' in self.modal:
            img = Image.open(self.OM_path_lst[idx])
            img_512 = self.resize_512(img)
            img_512_t = self.transforms(img_512)

            img_256 = self.resize_256(img)
            img_256_t = self.transforms(img_256)

            img_128 = self.resize_128(img)
            img_128_t = self.transforms(img_128)

            img_64 = self.resize_64(img)
            img_64_t = self.transforms(img_64)

        return img_512_t, img_256_t, img_128_t, img_64_t, y

    def __len__(self): # 决定idx的范围
        return len(self.label_lst)
