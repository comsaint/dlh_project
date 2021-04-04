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
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, classification_report

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
    best_val_roc = 0.0
    best_model_path = ''
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        m.train()
        print("=" * 40)
        print(f"Epoch {epoch + 1}")

        running_loss = 0.0
        val_loss = 0.0
        for i, data in enumerate(train_data_loader, 0):
            if device != 'cpu':
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
        print(f'Time elapsed: {(time.time()-start_time)/60.0:.1f} minutes.')
        save_model(m, epoch)  # save every epoch

        # validate every epoch
        print("Validating...")
        val_loss, val_auc, _, _, _ = eval_model(m, valid_loader, criterion)
        val_losses.append(val_loss)
        val_rocs.append(val_auc)
        if val_auc > best_val_roc:
            best_val_roc = val_auc
            best_model_path = save_model(m, epoch, best=True)

    print(f'Finished Training. Total time: {(time.time() - start_time) / 60} minutes.')
    print(f"Best ROC achieved on validation set: {best_val_roc:3f}")
    return model, train_losses, val_losses, best_val_roc, val_rocs, best_model_path


def eval_model(model, loader, criterion):
    # validate every epoch
    loss = 0.0
    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        model.eval()
        # empty tensors to hold results
        Y_prob, Y_true, Y_pred = [], [], []
        for inputs, labels in loader:
            probs = model(inputs.to(device))  # prediction probability
            labels = labels.type(torch.LongTensor).to(device)  # true labels
            _, predicted = torch.max(probs, dim=1)
            # stack results
            Y_prob.append(probs[:, -1])  # probability of positive class
            Y_true.append(labels)
            Y_pred.append(predicted)

            loss += float(criterion(probs, labels))

    # convert to numpy
    Y_prob = torch.cat(Y_prob).detach().cpu().numpy()
    Y_pred = torch.cat(Y_pred).detach().cpu().numpy()
    Y_true = torch.cat(Y_true).detach().cpu().numpy()

    # TODO: use other metrics here
    print(f"ROC: {roc_auc_score(Y_true, Y_prob):.3f}")
    auc = roc_auc_score(Y_true, Y_prob)
    print(classification_report(Y_true, Y_pred))

    return loss, auc, Y_prob, Y_pred, Y_true


def save_model(model, num_epochs, root_dir=config.ROOT_PATH, model_dir=config.MODEL_DIR, best=False):
    if best:
        path = os.path.join(root_dir, model_dir, f'{config.MODEL_NAME}_{config.DISEASE}_best.pth')
    else:
        path = os.path.join(root_dir, model_dir, f'{config.MODEL_NAME}_{config.DISEASE}_{num_epochs}epoch.pth')
    torch.save(model.state_dict(), path)
    print(f"Model Saved at: {path}")
    return path


if __name__ == "__main__":
    # load labels
    df_data, lst_labels = load_data_file(sampling=2000)
    print(f"Number of images: {len(df_data)}")
    # split the finding (disease) labels, to a list
    print(f"Number of labels: {len(lst_labels)}")
    print(f"Labels: {lst_labels}")
    # split data into train, val and test set
    df_train_val, df_test = make_train_test_split(df_data)

    # use stratify, especially for imbalance dataset
    df_train, df_val = train_test_split(df_train_val, stratify_label=config.DISEASE)
    print(f"Number of train images: {len(df_train)}")
    print(f"Number of val images: {len(df_val)}")
    print(f"Number of test images: {len(df_test)}")
    assert (len(df_train) + len(df_val) + len(df_test)) == len(df_data), \
        "Total number of images does not equal to sum of train+val+test!"

    # Criterion, with class weight
    # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#class_weights
    num_neg = sum(df_train[config.DISEASE] == 0)
    num_pos = sum(df_train[config.DISEASE] == 1)
    assert num_neg + num_pos == len(df_train)
    print(f"# of negative/positive cases: {num_neg}:{num_pos}")
    class_weight = torch.FloatTensor([(1 / num_neg) * (len(df_train)) / 2.0, (1 / num_pos) * (len(df_train)) / 2.0]).to(
        device)
    print(f"Class weight: {class_weight}")
    criterion = nn.CrossEntropyLoss(weight=class_weight)  # note the class weight

    # Initialize the model for this run
    model, input_size = initialize_model(config.MODEL_NAME, config.NUM_CLASSES, config.FEATURE_EXTRACT, use_pretrained=True)
    model = model.to(device)
    print(model)  # print structure
    print(f"Input image size: {input_size}")

    # make data loader
    tfx = make_data_transform(input_size)
    train_data_loader = load_data(df_train, label=config.DISEASE, transform=tfx['train'], shuffle=True)
    val_data_loader = load_data(df_val, label=config.DISEASE, transform=tfx['test'], shuffle=False)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # train
    model, t_losses, v_losses, v_best_auc, v_rocs, best_model_pth = train_model(model, train_data_loader,
                                                                                val_data_loader, criterion, optimizer,
                                                                                num_epochs=config.NUM_EPOCHS,
                                                                                verbose=True)

    # load and test on the best model
    model.load_state_dict(torch.load(best_model_pth))
    test_data_loader = load_data(df_test, label=config.DISEASE, transform=tfx['test'], shuffle=False)
    test_loss, test_auc, _, _, _ = eval_model(model.to(device), test_data_loader, criterion)
    print(f"Test loss: {test_loss}; Test ROC: {test_auc}")
    print("End of script.")
