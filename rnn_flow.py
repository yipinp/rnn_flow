import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
import torch.nn.functional as F

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

class rnn_flow(torch.nn.Module):
    def __init__(self,input_size,hidden_size,num_layers)):
        super(rnn_flow,self).__init__()
        self.rnn0 = torch.nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size
            num_layers = num_layers
            batch_first = True,
        )

        self.rnn1 = torch.nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size
            num_layers = num_layers
            batch_first = True,
        )

        self.conv1= nn.Sequential(
            nn.Conv2d(
                in_channels= 3,
                out_channels= 16,
                kernel_size= 3,
                stride= 1,
                padding = 1,
            )
            nn.ReLU()
            #nn.MaxPool2d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,3,1,1)
            nn.ReLU()

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1)
            nn.ReLU()
        )

    def forward(self,left,right):
        x = torch.cat((left,right),dim=0)
        x = self.conv1(x)
        x = self.base_layer(x)
        return x








