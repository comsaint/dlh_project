import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import writer, make_optimizer_and_scheduler
from train_model import train_model, eval_model
import config
from dataset import make_data_transform, load_data
from data_processing import load_data_file, make_train_test_split, make_train_val_split
from model import initialize_model


def main(model_name=config.MODEL_NAME, verbose=config.VERBOSE):
    # load labels
    df_data, lst_labels = load_data_file()
    print(f"Number of images: {len(df_data)}")
    
    # split the finding (disease) labels, to a list
    print(f"Number of labels: {len(lst_labels)}")
    print(f"Labels: {lst_labels}")
    
    # split data into train+val and test set
    df_train_val, df_test = make_train_test_split(df_data)
    # split train+val set into train and val set
    df_train, df_val = make_train_val_split(df_train_val)

    # make sure same patient does not exist in both train and val set
    assert set(df_train['Patient ID'].tolist()).isdisjoint(set(df_val['Patient ID'].tolist())), \
        "Same patient exist in train and validation set!"
    assert set(df_train['Patient ID'].tolist()).isdisjoint(set(df_test['Patient ID'].tolist())), \
        "Same patient exist in train and test set!"
    assert set(df_val['Patient ID'].tolist()).isdisjoint(set(df_test['Patient ID'].tolist())), \
        "Same patient exist in validation and test set!"

    # make sure all diseases appear at least 5 times in train and validation set
    if verbose:
        print(df_train[config.TEXT_LABELS].sum())
        print(df_val[config.TEXT_LABELS].sum())
    assert all(df_train[config.TEXT_LABELS].sum() >= 5), \
        "At least 1 disease appears less than 5 times in train set!"
    assert all(df_val[config.TEXT_LABELS].sum() >= 5), \
        "At least 1 disease appears less than 5 times in validation set!"

    if verbose:
        print(f"Number of train images: {len(df_train)}")
        print(f"Number of val images: {len(df_val)}")
        print(f"Number of test images: {len(df_test)}")
    assert (len(df_train) + len(df_val) + len(df_test)) == len(df_data), \
        "Total number of images does not equal to sum of train+val+test!"

    # Initialize the model for this run
    model, input_size, use_model_loss = initialize_model()
    print(f'Model selected: {model_name}')
    print(f"Input image size: {input_size}")

    # make data loaders
    tfx = make_data_transform(input_size)
    train_data_loader = load_data(df_train, transform=tfx['train'], shuffle=True)
    val_data_loader = load_data(df_val, transform=tfx['test'], shuffle=False)
    test_data_loader = load_data(df_test, transform=tfx['test'], shuffle=False)

    # Criterion
    # Sigmoid + BCE loss https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    # note we do the sigmoid here instead of inside model for numerical stability
    if config.USE_CLASS_WEIGHT:
        class_weight = 1 / np.mean(df_data[lst_labels]) - 1  # ratio of #pos:#neg samples
        class_weight = torch.FloatTensor(class_weight.tolist()).to(config.DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight)
        if verbose:
            print(f"Class weights:\n{class_weight}")
    else:
        criterion = nn.BCEWithLogitsLoss()

    # train
    model, t_losses, v_losses, v_best_loss, v_rocs, roc_at_best_v_loss, best_model_pth = \
        train_model(model, train_data_loader, val_data_loader, criterion, num_epochs=config.NUM_EPOCHS)
    
    # load and test on the best model
    model.load_state_dict(torch.load(best_model_pth))
    test_loss, test_auc, _, _, _ = eval_model(model, test_data_loader, criterion)
    
    print(f"Best Validation loss: {v_best_loss}; at which AUC = {roc_at_best_v_loss}")
    print(f"Test loss: {test_loss}; Test ROC: {test_auc}")
        
    writer.add_scalar("Loss/test", test_loss, 0)
    writer.add_scalar("ROCAUC/test", test_auc, 0)
    writer.flush()
    writer.close()
    print("End of script.")
    return None


if __name__ == "__main__":
    main(verbose=config.VERBOSE)
