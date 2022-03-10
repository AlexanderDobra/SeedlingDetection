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
from models.losses import *
from torch.utils.data import DataLoader
from models.depthset import SeedlingDataset
#from depthsetold import SeedlingDataset
#from depthsetnewold import SeedlingDataset
from sklearn.model_selection import train_test_split
from torchvision.models.resnet import resnet101, resnet50, resnet152, resnet18, resnet34

#CONFIG:
epochs = 20000
batchsize = 25
learningrate = 0.00001
datasetsplit = 0
weight_decay = 4e-5

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dataloaders():

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    train_dataset = SeedlingDataset(ROOT_DIR)

    train_set, test_set = train_test_split(train_dataset, test_size=0.2, random_state=datasetsplit)
    train, valid = train_test_split(train_set, test_size=0.1, random_state=datasetsplit)

    train_data_loader = DataLoader(
        train,
        batch_size=batchsize,
        shuffle=True,
    )

    test_data_loader = DataLoader(
        test_set,
        batch_size=batchsize,
        shuffle=False,
    )

    val_data_loader = DataLoader(
        valid,
        batch_size=batchsize,
        shuffle=False,
    )

    #print('length of trainset: ', len(train_data_loader))
    #print('length of valset: ', len(val_data_loader))
    #print('length of testset: ', len(test_data_loader))

    return train_data_loader, test_data_loader, val_data_loader

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

def train(train_data_loader, model, optimizer, device):
    running_loss = 0
    model.train()
    for i, data in enumerate(train_data_loader):

        loss, l1_losses, out, target, _, orig_inp, edges = calc_loss(data, model,
                                                                 l1_criterion, grad_criterion, normal_criterion,
                                                                 device=device,
                                                                 interpolate=True,
                                                                 edge_factor=0.6,
                                                                 batch_idx=i)

        optimizer.zero_grad()
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        #plotify(mode='add', count, loss.cpu().detach().numpy())

    train_loss = running_loss/len(train_data_loader.dataset)

    return train_loss

def test(test_data_loader, model, device):
    running_loss = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_data_loader):

            loss, l1_losses, out, target, _, orig_inp, edges = calc_loss(data, model,
                                                                    l1_criterion, grad_criterion, normal_criterion,
                                                                    device=device,
                                                                    interpolate=True,
                                                                    edge_factor=0.6,
                                                                    batch_idx=i)

        #plotify(mode='add', count, loss.cpu().detach().numpy())

    running_loss += loss.item()
    train_loss = running_loss/len(test_data_loader.dataset)

    model.train()

    return train_loss

def plotify(mode, x=None, y=None):
    global count
    global x1
    global y1
    if mode == 'empty':
        count = 1
        x1 = []
        y1 = []
    if mode == 'add':
        x1.append(x)
        y1.append(y)
        count += 1
    if mode == 'plot':
        plt.plot(x1, y1)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.savefig('Res50Layers4_MSE_ADAM.png')

def mod(type, layers, pretrain):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MDE(type, layers, pretrain).to(device)

    return model, device

#Set criterion, opitmizer & scalar
def optimer(model):

    #optimizer = optim.Adam(model.parameters(), lr=learningrate)
    optimizer = optim.Adam(model.parameters(), lr=learningrate,betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay) #betas=(0.9, 0.999), eps=1e-08
    
    return optimizer

def run(seed, type, layers, pretrain):

    start = time.time()

    set_seed(seed)

    trainset, testset, validset = dataloaders()
    model, device = mod(type, layers, pretrain)
    optimizer = optimer(model)

    best_loss = 100
    #plotify(mode='empty')

    for epoch in range(epochs):

        newtime = time.time()

        if ((newtime - start) / 60) < (60*12):

            train_loss = train(trainset, model, optimizer, device)
            val_loss = test(validset, model, device)
            print(f"Epoch #{epoch+1} val_loss: {val_loss}")
            if val_loss < best_loss:
                model = model.to("cpu")
                best_model = copy.deepcopy(model)
                model = model.to(device)
                best_loss = val_loss

    #plotify(mode='plot')

    best_model.to(device)
    test_loss = test(testset, best_model, device)

    end = time.time()
    totaltime = (end - start) / 60

    #Empty plotting arrays
    #plotify(mode='empty')

    return totaltime, test_loss, best_model.to('cpu')