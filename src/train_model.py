import sys  
sys.path.insert(0, '../src')
import config
import util
import dataset

import numpy as np

import random
import os
import torch

import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import accuracy_score, roc_auc_score

random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
os.environ["PYTHONHASHSEED"] = str(config.SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_model(model, train_data_loader, val_data_loader, criterion, optimizer, num_epochs=config.NUM_EPOCHS):

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
    
    return model

def save_model(model, num_epochs=config.NUM_EPOCHS, num_classes=14, root_dir=config.ROOT_PATH, model_dir=config.MODEL_DIR):
    
    MODEL_PATH = os.path.join(root_dir, model_dir, f'{str(model.__class__.__name__)}_{num_epochs}epoch_{num_classes}classes.pth')
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model Saved at: {MODEL_PATH}")

def multi_category_loss_fn(outputs, targets):
    
    tl = []
    for o,t in zip(outputs.T, targets.T):
        tl.append(nn.BCELoss()(o, t))
        
    return sum(tl) / len(tl)
