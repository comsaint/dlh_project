#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys  
sys.path.insert(0, '../src')
import config
import util

import data_processing as dp
import numpy as np

import random
import os
import torch
from torch.utils.data import Dataset

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import accuracy_score, roc_auc_score

from PIL import Image

random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
os.environ["PYTHONHASHSEED"] = str(config.SEED)
torch.multiprocessing.freeze_support()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print(f"Device: {device} Name: {torch.cuda.get_device_name()}")

num_epochs=config.NUM_EPOCHS

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop((112,112)),
        transforms.RandomHorizontalFlip(),  # data augmentation
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((128,128)),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class NihDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        image  = Image.open(self.dataframe.loc[idx, 'Image_path']).convert('L')  # via .getband(), some images are know to have 4 channels. Here we convert them to 1-channel grayscale.
        target = self.dataframe.loc[idx, 'Labels']
            
        if self.transform:
            image = self.transform(image)
        
        return image, target

def load_data(dataframe, transform=None, batch_size=32, shuffle=True, num_workers=4):
    '''
    Data Loader with batch loading and transform.
    '''
    image_data = NihDataset(dataframe, transform=transform)
    pin = device=='cpu'
    num_workers=num_workers*int(device!='cpu')
    loader = torch.utils.data.DataLoader(image_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin)
    return loader

class NihDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        image  = Image.open(self.dataframe.loc[idx, 'Image_path']).convert('L')  # via .getband(), some images are know to have 4 channels. Here we convert them to 1-channel grayscale.
        target = self.dataframe.loc[idx, 'Labels']
            
        if self.transform:
            image = self.transform(image)
        
        return image, target

def load_data(dataframe, transform=None, batch_size=32, shuffle=True, num_workers=4):
    '''
    Data Loader with batch loading and transform.
    '''
    image_data = NihDataset(dataframe, transform=transform)
    pin = device=='cpu'
    num_workers=num_workers*int(device!='cpu')
    loader = torch.utils.data.DataLoader(image_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin)
    return loader

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

def multi_category_loss_fn(outputs, targets):
    
    tl = []
    for o,t in zip(outputs.T, targets.T):
        tl.append(nn.BCELoss()(o, t))
        
    return sum(tl) / len(tl)

def train_model(model, train_data_loader, val_data_loader, criterion, optimizer,
               root_dir=config.ROOT_PATH, data_dir=config.INPUT_DATA_DIR):
    
    print(f"Training started") 
    print(f"    Mode          : {device}")
    print(f"    Model type    : {type(model)}")
    
    start_time = time.time()

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        model.train()
        print(f"Epoch {epoch+1}")
        
        running_loss = 0.0
        for i, data in enumerate(train_data_loader, 0):
            torch.cuda.empty_cache()

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.type(torch.FloatTensor).to(device)

            # forward + backward + optimize
            outputs = model(inputs)
            loss = multi_category_loss_fn(outputs, labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print(".", end ="")
            if i % 10 == 9:    # print every 10 mini-batches
                print(f' Epoch: {epoch + 1:>2} , Bacth: {i + 1:>3} , loss: {running_loss / (i+1)} Average batch time: {(time.time()-start_time)/(i+1)} secs')
                #running_loss = 0.0

        print()
        
        # validate every epoch
        val_loss = 0
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            model.eval()
            Y_prob, Y_pred, Y_true = [], [], []
            for idx, (images, labels) in enumerate(val_data_loader):
                probs = model(images.to(device))
                labels = labels.type(torch.FloatTensor).to(device)

                val_loss += multi_category_loss_fn(probs, labels)
                predicted = probs > 0.5

                probs = probs.cpu().detach().numpy().astype(float)
                predicted = predicted.cpu().detach().numpy().astype(float)
                labels = labels.cpu().detach().numpy().astype(float)

                Y_prob.append(probs)
                Y_pred.append(predicted)
                Y_true.append(labels)
                
        Y_prob = np.concatenate(Y_prob, axis=0)
        Y_pred = np.concatenate(Y_pred, axis=0)
        Y_true = np.concatenate(Y_true, axis=0)
        
        train_losses.append(running_loss/len(train_data_loader))
        val_losses.append(val_loss/len(val_data_loader))
        acc = accuracy_score(Y_true, Y_pred)
        roc = roc_auc_score(Y_true, Y_prob)
        
        print(f"Epoch              : {epoch+1}/{num_epochs}")
        print(f"Training Loss      : {train_losses[-1]}")
        print(f"Validation Loss    : {val_losses[-1]}")
        print(f"Validation Accuracy: {acc}")
        print(f"Validation ROC     : {roc}")
            
    print(f'Training Finished. Total time: {time.time()-start_time} secs.')

    MODEL_PATH = os.path.join(root_dir, data_dir, f'SimpleNet_{num_epochs}epoch.pth')
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model Saved at: {MODEL_PATH}")
    
    return model
    

def run(disease_filter_list=None, position_filter_list=None,num_epochs=config.NUM_EPOCHS):
    
    print(f"Num Epochs        : {num_epochs}")
    print(f"Num Classes       : {len(disease_filter_list)}")
    print(f"Target Class List : {disease_filter_list}\n")
    
    df_data , labels      = dp.load_data()
    
    if disease_filter_list:
        labels = disease_filter_list
        #df_data = df_data[df_data['Finding Labels'].isin(disease_filter_list)]
          
    if position_filter_list:
        df_data = df_data[df_data['View Position'].isin(position_filter_list)]
    
    df_data , dict_labels = dp.multi_hot_label(df_data, labels)
    df_train, df_test     = dp.make_train_test_split(df_data)
    
    df_dev, df_val   = dp.train_test_split(df_test) 
    df_train, df_dev, df_val = df_train.reset_index(drop=True), df_dev.reset_index(drop=True), df_val.reset_index(drop=True)

    model = SimpleNet(len(dict_labels)).to(device)

    criterion = nn.CrossEntropyLoss()  # change to CrossEntropyLoss if  multiclass
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    num_epochs = config.NUM_EPOCHS
    
    train_data_loader = load_data(df_train, transform=data_transforms['train'], shuffle=False, batch_size=config.TRAIN_BATCH_SIZE  ,  num_workers=config.TRAIN_WORKERS)
    val_data_loader   = load_data(df_dev  , transform=data_transforms['test'] , shuffle=False, batch_size=config.VAL_BATCH_SIZE  , num_workers=config.VAL_WORKERS)

    model = train_model(model, train_data_loader, val_data_loader, criterion, optimizer)

if __name__ == "__main__":
    run()
