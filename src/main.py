import numpy as np
import torch
from torch import nn
from utils import writer
from train_model import train_model, eval_models
import config
from dataset import make_data_transform, load_data
from data_processing import load_data_file, make_train_test_split, make_train_val_split
from model import initialize_model, SimpleCLF


def main(verbose=config.VERBOSE):
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
    '''
    if verbose:
        print(df_train[config.TEXT_LABELS].sum())
        print(df_val[config.TEXT_LABELS].sum())
    assert all(df_train[config.TEXT_LABELS].sum() >= 5), \
        "At least 1 disease appears less than 5 times in train set!"
    assert all(df_val[config.TEXT_LABELS].sum() >= 5), \
        "At least 1 disease appears less than 5 times in validation set!"
    '''
    if verbose:
        print(f"Number of train images: {len(df_train)}")
        print(f"Number of val images: {len(df_val)}")
        print(f"Number of test images: {len(df_test)}")
    assert (len(df_train) + len(df_val) + len(df_test)) == len(df_data), \
        "Total number of images does not equal to sum of train+val+test!"

    # make data loaders
    tfx = make_data_transform(config.GLOBAL_IMAGE_SIZE)
    train_data_loader = load_data(df_train, transform=tfx['train'], shuffle=True)
    val_data_loader = load_data(df_val, transform=tfx['test'], shuffle=False)
    test_data_loader = load_data(df_test, transform=tfx['test'], shuffle=False)

    # Criterion
    # Sigmoid + BCE loss https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    # note we do the sigmoid here instead of inside model for numerical stability
    if config.USE_CLASS_WEIGHT:
        class_weight = 1 / np.mean(df_data[lst_labels]) - 1  # ratio of #pos:#neg samples
        class_weight = torch.FloatTensor(class_weight.tolist()).to(config.DEVICE)
        g_criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight)
        l_criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight)
        f_criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight)
        if verbose:
            print(f"Class weights:\n{class_weight}")
    else:
        g_criterion = nn.BCEWithLogitsLoss()
        l_criterion = nn.BCEWithLogitsLoss()
        f_criterion = nn.BCEWithLogitsLoss()

    criterions = [g_criterion, l_criterion, f_criterion]

    # train
    models, t_losses, v_losses, v_best_loss, v_rocs, roc_at_best_v_loss, best_model_paths = \
        train_model(
            train_loader=train_data_loader,
            val_loader=val_data_loader,
            criterions=criterions
            )
    print(f"Best Validation loss: {v_best_loss}; at which AUC = {roc_at_best_v_loss}")

    ##############################################
    # Test on the best models
    ##############################################
    g_best_path, l_best_path, f_best_path = best_model_paths
    g_model, g_input_size, _, g_fm_name, g_pool_name, g_fm_size = initialize_model(config.GLOBAL_MODEL_NAME)
    l_model, l_input_size, _, _, l_pool_name, l_fm_size = initialize_model(config.LOCAL_MODEL_NAME)
    if config.USE_EXTRA_INPUT:
        s = g_fm_size + l_fm_size + 3
    else:
        s = g_fm_size + l_fm_size
    f_model = SimpleCLF(input_size=s).to(config.DEVICE)

    g_model.load_state_dict(torch.load(g_best_path))
    l_model.load_state_dict(torch.load(l_best_path))
    f_model.load_state_dict(torch.load(f_best_path))
    models = [g_model, l_model, f_model]

    test_loss, test_auc, _, _, _ = eval_models(models, test_data_loader, criterions)

    print(f"Test loss: {test_loss[2]}; Test ROC: {test_auc[2]}")

    writer.add_scalar("Loss/test", test_loss[2], 0)
    writer.add_scalar("ROCAUC/test", test_auc[2], 0)
    writer.flush()
    writer.close()
    print("End of script.")
    return None


if __name__ == "__main__":
    main(verbose=config.VERBOSE)
