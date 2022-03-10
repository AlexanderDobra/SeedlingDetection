import copy
import os
import numpy as np
import torch
import random
import time
from PIL import Image
import xml.etree.ElementTree as ET
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.conv import ConvTranspose2d
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision.models.resnet import resnet101, resnet50, resnet152, resnet18, resnet34


class MDE(nn.Module):
    def __init__(self, res, layers, pretrained):
        super(MDE, self).__init__()
        if res == 18:
            resnet = resnet50(pretrained=pretrained, progress=False)
        if res == 34:
            resnet = resnet50(pretrained=pretrained, progress=False)
        if res == 50:
            resnet = resnet50(pretrained=pretrained, progress=False)
        if res == 101:
            resnet = resnet101(pretrained=pretrained, progress=False)
        if res == 152:
            resnet = resnet152(pretrained=pretrained, progress=False)

        self.layers = layers

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = nn.Sequential(resnet.layer1) # 256
        self.layer2 = nn.Sequential(resnet.layer2) # 512
        self.layer3 = nn.Sequential(resnet.layer3) # 1024
        self.layer4 = nn.Sequential(resnet.layer4) # 2048

        #Toplayer
        self.toplayer=nn.Conv2d(2048,256,1,1,0)

        #3x3 convolution fusion feature
        self.smooth1=nn.Conv2d(256,256,3,1,1)
        self.smooth2=nn.Conv2d(256,256,3,1,1)
        self.smooth3=nn.Conv2d(256,256,3,1,1)

        #Lateral layers
        self.latlayer1=nn.Conv2d(1024,256,1,1,0)
        self.latlayer2=nn.Conv2d(512,256,1,1,0)
        self.latlayer3=nn.Conv2d(256,256,1,1,0)

        self.upconv1=nn.ConvTranspose2d(256, 256, 4, 4) #???

        self.predict1=nn.Conv2d(256,64,3,1,1)
        self.relu1=nn.ReLU()

        self.predict2=nn.Conv2d(64,1,3,1,1)
        self.relu2=nn.ReLU()

    def _upsample_add(self,x,y):
        _,_,H,W=y.shape
        return F.interpolate(x,size=(H,W),mode='bilinear', align_corners=True)+y
    
    def forward(self, x):

        # Bottom-up
        c1 = self.layer0(x) 
        c2 = self.layer1(c1) # 256 channels, 1/4 size
        c3 = self.layer2(c2) # 512 channels, 1/8 size
        c4 = self.layer3(c3) # 1024 channels, 1/16 size

        if self.layers == 4:
            #Bottom-up
            c5 = self.layer4(c4) # 2048 channels, 1/32 size
            # Top-down
            p5 = self.toplayer(c5)
            p4 = self._upsample_add(p5, self.latlayer1(c4)) # 256 channels, 1/16 size
            p4 = self.smooth1(p4)
        elif self.layers == 3:
            # Top-down (new toplayer)
            p4 = self.latlayer1(c4)

        p3 = self._upsample_add(p4, self.latlayer2(c3)) # 256 channels, 1/8 size
        p3 = self.smooth2(p3) # 256 channels, 1/8 size
        p2 = self._upsample_add(p3, self.latlayer3(c2)) # 256, 1/4 size
        p2 = self.smooth3(p2) # 256 channels, 1/4 size

        x = self.upconv1(p2)

        x = self.predict1(x)
        x = self.relu1(x)

        x = self.predict2(x)
        x = self.relu2(x)

        return x

#model = MDE(type, layers, pretrain).to(device)
