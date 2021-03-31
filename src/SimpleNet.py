import sys  
sys.path.insert(0, '../src')
import config
import util

import numpy as np

import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
os.environ["PYTHONHASHSEED"] = str(config.SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print(f"Device: {device} Name: {torch.cuda.get_device_name()}")

class SimpleNet(nn.Module):
    def __init__(self, len_out=14):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5)  # stride=1, padding=0
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(3, 8, 5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(8, 16, 5)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 10 * 10, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)  # 14 classes
        
        self.out = nn.Linear(64, len_out)
        
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))  # size: (batch_size*)3channels*110*110
        x = self.pool2(F.relu(self.conv2(x)))  # size: (batch_size*)8channels*53*53
        x = self.pool3(F.relu(self.conv3(x)))  # size: (batch_size*)8channels*53*53
        x = x.view(-1, 16 * 10 * 10)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        
        out = torch.sigmoid(self.out(x))
        #x = F.softmax(self.fc3(x))                # change to softmax if multiclass
        return out

def make_SimpleNet_model(num_classes=14):
    return SimpleNet(num_classes).to(device)