import sys
import time
import random
import os
import config
from math import inf
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, classification_report
from utils import calculate_metric
from utils import writer

sys.path.insert(0, '../src')

random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
os.environ["PYTHONHASHSEED"] = str(config.SEED)

def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=config.NUM_EPOCHS, use_model_loss=False,verbose=True):

    print(f"Training started") 
    print(f"    Mode          : {config.DEVICE}")
    print(f"    Model type    : {type(model)}")
    
    start_time = time.time()

    train_losses, val_losses, val_rocs = [], [], []
    best_val_loss, roc_at_best_val_loss = inf, 0.0
    best_model_path = ''
    
    for epoch in range(1, num_epochs + 1):  # loop over the dataset multiple times
        
        model.train()
        print(f"Epoch {epoch+1}")
        
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            
            iterations = len(train_data_loader)
            
            if config.DEVICE == 'cuda':
                torch.cuda.empty_cache()
        
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to(config.DEVICE), labels.type(torch.LongTensor).to(config.DEVICE)
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

            loss = torch.mean(loss)
            
            running_loss += float(loss.item())
            
            # print statistics
            print(".", end="")
            if verbose:
                if i+1 % 10 == 0:  # print every 10 mini-batches
                    print(f' Epoch: {epoch:>2} '
                          f' Bacth: {i + 1:>3} / {iterations} '
                          f' loss: {running_loss / (i + 1):>4} '
                          f' Average batch time: {(time.time() - start_time) / (i + 1)} secs')
        print(f"\nTraining loss: {running_loss / len(train_loader):>4f}")
        train_losses.append(running_loss / len(train_loader))  # keep trace of train loss in each epoch
        writer.add_scalar("Loss/train", running_loss / len(train_loader), epoch)  # write loss to TensorBoard
        print(f'Time elapsed: {(time.time()-start_time)/60.0:.1f} minutes.')
                
        if epoch % 5 == 0:  # save every 5 epochs
            save_model(model, epoch)  

        # validate every epoch
        print("Validating...")
        val_loss, val_auc, _, _, _ = eval_model(model, valid_loader, criterion, use_model_loss)
        
        print(f"Validation loss: {val_loss:>4f}")
        print(f"Validation ROC: {val_auc:>3f}")
        
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("ROCAUC/val", val_auc, epoch)
        
        val_losses.append(val_loss)
        val_rocs.append(val_auc)
        
        # save model if best ROC
        if val_auc > best_val_roc:
            best_val_roc = val_auc
            roc_at_best_val_loss = val_rocs
            best_model_path = save_model(model, epoch, best=True)
        writer.flush()

    print(f'Finished Training. Total time: {(time.time() - start_time) / 60} minutes.')
    print(f"Best ROC achieved on validation set: {best_val_roc:3f}")
    
    return model, train_losses, val_losses, best_val_loss, val_rocs, roc_at_best_val_loss, best_model_path

def eval_model(model, loader, criterion, use_model_loss=False, threshold=0.50, verbose=True):
    """
    evaluate test data on model and criterion, based of macro-average ROCAUC.
    """
    
    # validate every epoch
    loss = 0.0
    
    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        model.eval()
    
        # empty tensors to hold results
        y_prob, y_true, y_pred = [], [], []
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
                
            predicted = probs > threshold
            
            # stack results
            y_prob.append(probs)
            y_true.append(labels)
            y_pred.append(predicted)
            
    loss = loss/len(loader)
    
    # convert to numpy
    y_prob = torch.cat(y_prob).detach().cpu().numpy()
    y_pred = torch.cat(y_pred).detach().cpu().numpy()
    y_true = torch.cat(y_true).detach().cpu().numpy()

    # TODO: use other metrics here
    # compute macro-average ROCAUC
    macro_roc_auc = calculate_metric(y_prob, y_true)

    # print result
    if verbose:
        print(classification_report(y_true, y_pred))

    return loss, macro_roc_auc, y_prob, y_pred, y_true



def save_model(model, num_epochs, root_dir=config.ROOT_PATH, model_dir=config.MODEL_DIR, best=False):
    if best:
        path = os.path.join(root_dir, model_dir, f'{model.__class__.__name__}_{config.DISEASE}_best.pth') #Name from class such as for CapsNet
    else:
        path = os.path.join(root_dir, model_dir, f'{model.__class__.__name__}_{config.DISEASE}_{num_epochs}epoch.pth')
    
    torch.save(model.state_dict(), path)
    print(f"Model Saved at: {path}")
    return path


