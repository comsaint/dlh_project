import sys
import time
import random
import os
import config
import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter
from utils import calculate_metric

sys.path.insert(0, '../src')

writer = SummaryWriter()

random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
os.environ["PYTHONHASHSEED"] = str(config.SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_model(m, train_loader, valid_loader, criterion, optimizer, num_epochs=config.NUM_EPOCHS, verbose=True):
    print(f"Training started") 
    print(f"    Mode          : {device}")
    print(f"    Model type    : {type(m)}")
    
    start_time = time.time()
    train_losses, val_losses, val_rocs = [], [], []
    best_val_loss, roc_at_best_val_loss = 0.0, 0.0
    best_model_path = ''
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        m.train()
        print(f"Epoch {epoch+1}")
        
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            # get the inputs: a list of [inputs, labels]
            # `labels` is an 15-dim list of tensors
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = m(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss = torch.mean(loss)
            running_loss += float(loss.item())

            # print statistics
            print(".", end="")
            if verbose:
                if i % 50 == 9:  # print every 50 mini-batches
                    print(f' Epoch: {epoch + 1:>2}, '
                          f' Batch: {i + 1:>3} , '
                          f' loss: {running_loss / (i + 1):>4} '
                          f' Average batch time: {(time.time() - start_time) / (i + 1)} secs')
        print(f"\nTraining loss: {running_loss / len(train_loader):>4f}")
        train_losses.append(running_loss / len(train_loader))  # keep trace of train loss in each epoch
        writer.add_scalar("Loss/train", running_loss / len(train_loader), epoch)  # write loss to TensorBoard
        print(f'Time elapsed: {(time.time()-start_time)/60.0:.1f} minutes.')

        if epoch % 5 == 0:  # save every 5 epochs
            save_model(m, epoch)

        # validate every epoch
        print("Validating...")
        val_loss, val_auc, _, _, _ = eval_model(m, valid_loader, criterion)
        print(f"Validation loss: {val_loss:>4f}")
        print(f"Validation ROC: {val_auc:>3f}")
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("ROCAUC/val", val_auc, epoch)
        val_losses.append(val_loss)
        val_rocs.append(val_auc)
        # save model if best ROC
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            roc_at_best_val_loss = val_rocs
            best_model_path = save_model(m, epoch, best=True)

    print(f'Finished Training. Total time: {(time.time() - start_time) / 60} minutes.')
    print(f"Best ROC achieved on validation set: {best_val_loss:3f}")
    writer.flush()
    writer.close()
    return m, train_losses, val_losses, best_val_loss, val_rocs, roc_at_best_val_loss, best_model_path


def eval_model(model, loader, criterion, threshold=0.50, verbose=True):
    """
    evaluate test data on model and criterion, based of macro-average ROCAUC.
    """
    loss = 0.0
    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        model.eval()
        # empty tensors to hold results
        y_prob, y_true, y_pred = [], [], []
        for inputs, labels in loader:
            probs = model(inputs.to(device))  # prediction probability
            labels = labels.to(device)  # true labels
            predicted = probs > threshold
            # stack results
            y_prob.append(probs)
            y_true.append(labels)
            y_pred.append(predicted)
            loss += float(torch.mean(criterion(probs, labels)))
    loss = loss/len(loader)

    # convert to numpy
    y_prob = torch.cat(y_prob).detach().cpu().numpy()
    y_pred = torch.cat(y_pred).detach().cpu().numpy()
    y_true = torch.cat(y_true).detach().cpu().numpy()

    # compute macro-average ROCAUC
    macro_roc_auc = calculate_metric(y_prob, y_true)

    # print result
    if verbose:
        print(classification_report(y_true, y_pred))

    return loss, macro_roc_auc, y_prob, y_pred, y_true


def save_model(model, num_epochs, root_dir=config.ROOT_PATH, model_dir=config.MODEL_DIR, best=False):
    if best:
        path = os.path.join(root_dir, model_dir, f'{config.MODEL_NAME}_best.pth')
    else:
        path = os.path.join(root_dir, model_dir, f'{config.MODEL_NAME}_{num_epochs}epoch.pth')
    torch.save(model.state_dict(), path)
    print(f"Model Saved at: {path}")
    return path
