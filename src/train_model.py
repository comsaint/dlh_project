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

writer = SummaryWriter()
sys.path.insert(0, '../src')

random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
os.environ["PYTHONHASHSEED"] = str(config.SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_model(m, train_loader, valid_loader, criterion, optimizer, num_epochs=config.NUM_EPOCHS, verbose=False):
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
        for i, data in enumerate(train_loader, 0):
            torch.cuda.empty_cache()
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.type(torch.LongTensor).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = m(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += float(loss.item())
            print(".", end="")
            if verbose:
                if i % 10 == 9:  # print every 10 mini-batches
                    print(f' Epoch: {epoch + 1:>2}, Bacth: {i + 1:>3} , loss: {running_loss / (i + 1)} Average batch time: {(time.time() - start_time) / (i + 1)} secs')
        train_losses.append(running_loss / len(train_loader))  # keep trace of train loss in each epoch
        writer.add_scalar("Loss/train", running_loss / len(train_loader), epoch)  # write loss to TensorBoard
        print(f'Time elapsed: {(time.time()-start_time)/60.0:.1f} minutes.')

        if epoch % 5 == 0:  # save every 5 epochs
            save_model(m, epoch)

        # validate every epoch
        print("Validating...")
        val_loss, val_auc, _, _, _ = eval_model(m, valid_loader, criterion)
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


def eval_model(model, loader, criterion):
    # validate every epoch
    loss = 0.0
    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        model.eval()
        # empty tensors to hold results
        y_prob, y_true, y_pred = [], [], []
        for inputs, labels in loader:
            probs = model(inputs.to(device))  # prediction probability
            labels = labels.type(torch.LongTensor).to(device)  # true labels
            _, predicted = torch.max(probs, dim=1)
            # stack results
            y_prob.append(probs[:, -1])  # probability of positive class
            y_true.append(labels)
            y_pred.append(predicted)

            loss += float(criterion(probs, labels))
    loss = loss/len(loader)
    # convert to numpy
    y_prob = torch.cat(y_prob).detach().cpu().numpy()
    y_pred = torch.cat(y_pred).detach().cpu().numpy()
    y_true = torch.cat(y_true).detach().cpu().numpy()

    # TODO: use other metrics here
    print(f"ROC: {roc_auc_score(y_true, y_prob):.3f}")
    auc = roc_auc_score(y_true, y_prob)
    print(classification_report(y_true, y_pred))

    return loss, auc, y_prob, y_pred, y_true


def save_model(model, num_epochs, root_dir=config.ROOT_PATH, model_dir=config.MODEL_DIR, best=False):
    if best:
        path = os.path.join(root_dir, model_dir, f'{config.MODEL_NAME}_{config.DISEASE}_best.pth')
    else:
        path = os.path.join(root_dir, model_dir, f'{config.MODEL_NAME}_{config.DISEASE}_{num_epochs}epoch.pth')
    torch.save(model.state_dict(), path)
    print(f"Model Saved at: {path}")
    return path

