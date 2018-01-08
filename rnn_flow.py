import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from PIL import Image

#create kitti dataset
def default_loader(path):
    return Image.open(path).convert('RGB')

class kitti2015_train_dataset(Dataset) :
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        super(kitti2015_train_dataset, self).__init__()
        fh = open(txt, 'r')
        #train data format :  first second golden
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            left,right,golden = line.split()
            imgs.append((left,right,golden))
        self.imgs = imgs
        self.transfrom = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        left,  right, golden = self.imgs[index]
        left_img = self.loader(left)
        right_img = self.loader(right)
        golden_img = self.loader(golden)
        return left_img,  right_img, golden_img

    def __len__(self):
        return len(self.imgs)


class kitti2015_test_dataset(Dataset) :
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        super(kitti2015_test_dataset, self).__init__()
        fh = open(txt, 'r')
        #train data format :  first second golden
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            left,right = line.split()
            imgs.append((left,right))
        self.imgs = imgs
        self.transfrom = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        left,  right= self.imgs[index]
        left_img = self.loader(left)
        right_img = self.loader(right)
        return left_img,  right_img

    def __len__(self):
        return len(self.imgs)

train_data = kitti2015_train_dataset(txt='', transform=torchvision.transforms.ToTensor())
data_loader = DataLoader(train_data,batch_size=8, shuffle=True)






