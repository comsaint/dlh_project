import sys
import time
import random
import os
import config
from math import inf
import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.cuda.amp import GradScaler
import torch.backends.cudnn as cudnn
from utils import calculate_metric
from utils import writer
from torchvision.utils import save_image
import warnings
import gc

sys.path.insert(0, '../src')

random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
os.environ["PYTHONHASHSEED"] = str(config.SEED)

scaler = GradScaler()
gradient_accumulations = 8  # 1=no accumulation

warnings.filterwarnings("ignore")
warnings.simplefilter('always')

from dataset import make_data_transform, load_data
import pandas as pd

def train_model(m, train_loader, valid_loader, criterion, optimizer, scheduler=None, num_epochs=config.NUM_EPOCHS,
                save_freq=5, use_model_loss=False, verbose=True):
    print(f"Training started") 
    print(f"    Mode          : {config.DEVICE}")
    print(f"    Model type    : {type(m)}")

    start_time = time.time()

    train_losses, val_losses, val_rocs = [], [], []
    best_val_loss, roc_at_best_val_loss = inf, 0.0
    best_model_path = ''
    best_epoch = 0
    
    # similar to optimizer.zero_grad(). https://discuss.pytorch.org/t/model-zero-grad-or-optimizer-zero-grad/28426/6
    m.zero_grad()
    cudnn.benchmark = True
    for epoch in range(1, num_epochs+1):  # loop over the dataset multiple times
        
        m.train()
        print(f"Epoch {epoch}")
        
        running_loss = 0.0
        epoch_start_time = time.time()
        for i, (inputs, labels) in enumerate(train_loader, 0):
            iterations = len(train_loader)
            
            if config.DEVICE == 'cuda':
                torch.cuda.empty_cache()
                
            # get the inputs: a list of [inputs, labels]
            # `labels` is an 15-dim list of tensors
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)

            # forward + backward + optimize
            outputs = m(inputs)
            hidden  = None
            
            if type(outputs) == tuple and len(outputs) == 2:
                hidden  = outputs[1]
                outputs = outputs[0]    
                
            outputs.to(config.DEVICE)
            
            if use_model_loss:
                reconstructions = None
                loss = m.loss(inputs, outputs, labels, mean_error=True, reconstruct=False)
                
                if type(loss) == tuple and len(loss) == 2:
                    reconstructions  = loss[1]
                    loss = loss[0]
                
                # Reproduce the decoded image in Run Directory
                if i == 0 and reconstructions != None: 
                    folder_path = os.path.join(config.ROOT_PATH, config.OUT_DIR, config.MODEL_NAME)
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    save_image(inputs         ,f'{folder_path}/epoch_{epoch}_01_original.png')
                    if hidden != None:
                        save_image(hidden     ,f'{folder_path}/epoch_{epoch}_02_hidden.png')
                    save_image(reconstructions,f'{folder_path}/epoch_{epoch}_03_reconstr.png')
                 
                del reconstructions
                
                #probs = m.get_preduction(outputs).to(config.DEVICE) 
                #loss = criterion(probs, labels).to(config.DEVICE)
                #loss.requres_grad = True
                #loss /= 2
            else:
                loss = criterion(outputs, labels)
                
            if config.DEVICE == 'cuda':
                #optimize gpu mem
                del hidden,outputs,inputs,labels
                gc.collect()
            
                torch.cuda.empty_cache()
                scaler.scale(loss/gradient_accumulations).backward()
            
                # gradient accumulation trick
                # https://towardsdatascience.com/i-am-so-done-with-cuda-out-of-memory-c62f42947dca
                if (i + 1) % gradient_accumulations == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    m.zero_grad()
                    running_loss += float(loss.item())

            else:
                # zero the parameter gradients
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss = torch.mean(loss)
                running_loss += float(loss.item())
                
            # print statistics
            print(".", end="")
            if verbose:
                if (i+1) % 50 == 0:  # print every 50 mini-batches
                    print(f' Epoch: {epoch:>2} '
                          f' Bacth: {i + 1:>3} / {iterations} '
                          f' loss: {running_loss / (i + 1):>6.4f} '
                          f' Average batch time: {(time.time() - epoch_start_time) / (i + 1):>4.3f} secs')
            
        print(f"\nTraining loss: {running_loss / len(train_loader):>4f}")
        train_losses.append(running_loss / len(train_loader))  # keep trace of train loss in each epoch
        writer.add_scalar("Loss/train", running_loss / len(train_loader), epoch)  # write loss to TensorBoard
        print(f'Time elapsed: {(time.time()-start_time)/60.0:.1f} minutes.')

        if epoch % save_freq == 0:  # save model and checkpoint for inference or training
            save_model(m, epoch)
            save_checkpoint(m, epoch, optimizer, running_loss)
        save_model(m, epoch) ################
        
        # validation
        print("Validating...")
        m.eval()
        
        val_loss, val_auc, y_prob, _, y_true = eval_model(m, valid_loader, criterion, use_model_loss=use_model_loss)
        val_losses.append(val_loss)
        val_rocs.append(val_auc)
        
        # save model if best ROC
        if val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = val_loss
            roc_at_best_val_loss = val_auc
            best_model_path = save_model(m, epoch, best=True)
        
        print(f"Validation loss: {val_loss:>4f}")
        print(f"Validation ROC: {val_auc:>3f}")
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("ROCAUC/val", val_auc, epoch)
        writer.add_pr_curve('PR/val', y_true, y_prob, epoch)

        # step the learning rate
        if scheduler is not None:
            scheduler.step(val_loss)

        writer.flush()

    print(f'Finished Training. Total time: {(time.time() - start_time) / 60} minutes.')
    print(f"Best ROC on validation set: {best_val_loss:3f}, achieved on epoch #{best_epoch}")

    return m, train_losses, val_losses, best_val_loss, val_rocs, roc_at_best_val_loss, best_model_path


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
        cudnn.benchmark = True
        for i, (inputs, labels) in enumerate(loader):
            probs = model(inputs.to(config.DEVICE))  # prediction probability
            
            if type(probs) == tuple and len(probs) == 2:
                probs = probs[0].to(config.DEVICE)    
                
            labels = labels.to(config.DEVICE)  # true labels
            
            if use_model_loss:
                loss += model.loss(inputs, probs, labels, mean_error=True, reconstruct=False).item()
                probs = model.get_preduction(probs).to(config.DEVICE) 
                #loss += float(torch.mean(criterion(probs, labels)))
            else:
                loss += float(torch.mean(criterion(probs, labels)))
                
            predicted = probs > threshold
            
            # stack results
            y_prob.append(probs)
            y_true.append(labels)
            y_pred.append(predicted)
            
            print(".", end="")
            if (i+1) % 10 == 0 :  # print every 10 mini-batches
                print(f' {int(i*100/len(loader)):>2}% complete ')
                
            loss += float(criterion(probs, labels))
    
    print(" 100% complete")
    loss = loss/len(loader)
    
    # convert to numpy
    y_prob = torch.cat(y_prob).detach().cpu().numpy()
    y_pred = torch.cat(y_pred).detach().cpu().numpy()
    y_true = torch.cat(y_true).detach().cpu().numpy()
    
    

    # compute macro-average ROCAUC, and ROCAUC of each class
    macro_roc_auc, roc_aucs = calculate_metric(y_prob, y_true)

    # print results
    if verbose:
        #print(classification_report(y_true, y_pred, target_names=config.TEXT_LABELS))
        print(f"Disease:{'':<22}ROCAUC")
        for i, lb in enumerate(config.TEXT_LABELS):
            print(f"{lb:<30}: {roc_aucs[i]:.4f}")
        print(f"\nROCAUC (Macro): {macro_roc_auc}")

    return loss, macro_roc_auc, y_prob, y_pred, y_true


def save_model(model, num_epochs, root_dir=config.ROOT_PATH, model_dir=config.MODEL_DIR, best=False, verbose=True):
    model_subdir = 'model_' + config.MODEL_NAME
    folder_path = os.path.join(root_dir, model_dir, model_subdir)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if best:
        path = os.path.join(folder_path, f'{config.MODEL_NAME}_best.pth')
    else:
        path = os.path.join(folder_path, f'{config.MODEL_NAME}_{num_epochs}epoch.pth')
    torch.save(model.state_dict(), path)
    if verbose:
        print(f"Model Saved at: {path}")
    return path


def save_checkpoint(model, epoch, optimizer, loss):
    root_dir = config.ROOT_PATH
    ckpt_dir = config.CHECKPOINT_DIR
    folder_path = os.path.join(root_dir, ckpt_dir)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    ckpt_subdir = 'checkpoint_' + config.MODEL_NAME
    path = os.path.join(folder_path, ckpt_subdir)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Checkpoint saved: {path}")
    return None
