import sys
import time
import random
import os
import config
from dataset import make_data_transform, load_data
from data_processing import load_data_file, train_test_split, make_train_test_split
from model import initialize_model
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, classification_report
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, '../src')

random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
os.environ["PYTHONHASHSEED"] = str(config.SEED)

def train_model(model, train_data_loader, val_data_loader, criterion, optimizer, writer, num_epochs=config.NUM_EPOCHS, use_model_loss=False,verbose=True,):

    print(f"Training started") 
    print(f"    Mode          : {config.DEVICE}")
    print(f"    Model type    : {type(model)}")
    
    start_time = time.time()

    train_losses, val_losses, val_rocs = [], [], []
    best_val_roc = 0.0
    best_model_path= ''
    
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        model.train()
        print(f"Epoch {epoch+1}")
        
        running_loss = 0.0
        for i, data in enumerate(train_data_loader, 0):
            
            iterations = len(train_data_loader)
            
            if config.DEVICE == 'cuda':
                torch.cuda.empty_cache()
        
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            
            inputs = inputs.to(config.DEVICE)
            labels = labels.type(torch.LongTensor)

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = model(inputs).to(config.DEVICE)
            
            if use_model_loss:
                loss = model.loss(inputs, outputs, labels, mean_error=True, reconstruct=True)
            else:
                loss = criterion(outputs, labels)
            
            if config.DEVICE == 'cuda':
                torch.cuda.empty_cache()
                
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += float(loss.item())
            print(".", end="")
            if verbose:
                if i+1 % 1 == 0:  # print every 10 mini-batches
                    print(f' Epoch: {epoch + 1:>2}, Bacth: {i + 1:>3} / {iterations} , loss: {running_loss / (i + 1)} Average batch time: {(time.time() - start_time) / (i + 1)} secs')
        
        train_losses.append(running_loss / len(train_data_loader))  # keep trace of train loss in each epoch
        writer.add_scalar("Loss/train", running_loss / len(train_data_loader), epoch)  # write loss to TensorBoard
        
        print(f'Time elapsed: {(time.time()-start_time)/60.0:.1f} minutes.')
        if epoch % 5 == 0:  # save every 5 epochs
            save_model(model, epoch)  

        # validate every epoch
        
        print("Validating...")
        val_loss, val_auc, _, _, _ = eval_model(model, val_data_loader, criterion, use_model_loss)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("ROCAUC/val", val_auc, epoch)
        val_losses.append(val_loss)
        val_rocs.append(val_auc)
        if val_auc > best_val_roc:
            best_val_roc = val_auc
            best_model_path = save_model(model, epoch, best=True)

    print(f'Finished Training. Total time: {(time.time() - start_time) / 60} minutes.')
    print(f"Best ROC achieved on validation set: {best_val_roc:3f}")
    writer.flush()
    writer.close()
    return model, train_losses, val_losses, best_val_roc, val_rocs, best_model_path


def eval_model(model, loader, criterion, use_model_loss=False):
    # validate every epoch
    loss = 0.0
    
    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        model.eval()
        # empty tensors to hold results
        Y_prob, Y_true, Y_pred = [], [], []
        for inputs, labels in loader:
            if config.DEVICE == 'cuda':
                torch.cuda.empty_cache()
                
            probs = model(inputs.to(config.DEVICE))  # prediction probability
            labels = labels.type(torch.LongTensor).to(config.DEVICE)  # true labels
            
            if use_model_loss:
                loss += model.loss(inputs, probs, labels, mean_error=False, reconstruct=False).data[0]
                probs = model.get_preduction(probs) 
            else:
                loss += float(criterion(probs, labels))
                
            _, predicted = torch.max(probs, dim=1)
            # stack results
            Y_prob.append(probs[:, -1])  # probability of positive class
            Y_true.append(labels)
            Y_pred.append(predicted)
            
    loss = loss/len(loader)
    
    # convert to numpy
    Y_prob = torch.cat(Y_prob).detach().cpu().numpy()
    Y_pred = torch.cat(Y_pred).detach().cpu().numpy()
    Y_true = torch.cat(Y_true).detach().cpu().numpy()

    print(loss)
    
    # TODO: use other metrics here
    print(f"ROC: {roc_auc_score(Y_true, Y_prob):.3f}")
    auc = roc_auc_score(Y_true, Y_prob)
    print(classification_report(Y_true, Y_pred))

    return loss, auc, Y_prob, Y_pred, Y_true


def save_model(model, num_epochs, root_dir=config.ROOT_PATH, model_dir=config.MODEL_DIR, best=False):
    if best:
        path = os.path.join(root_dir, model_dir, f'{model.__class__.__name__}_{config.DISEASE}_best.pth')
    else:
        path = os.path.join(root_dir, model_dir, f'{model.__class__.__name__}_{config.DISEASE}_{num_epochs}epoch.pth')
    torch.save(model.state_dict(), path)
    print(f"Model Saved at: {path}")
    return path


