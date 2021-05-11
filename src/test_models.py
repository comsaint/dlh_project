"""
A script for quick testing of a trained model.
"""
from dataset import load_data, make_data_transform
from data_processing import load_data_file, make_train_test_split, make_train_val_split
from model import initialize_model, SimpleCLF
from train_model import eval_models
import torch
from torch import nn
import config
import os
from config import params
import numpy as np
from utils import calculate_metric
import functools

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_predictions(params, model_path, global_model_name, local_model_name, loader):
    params['GLOBAL_MODEL_NAME'], params['LOCAL_MODEL_NAME'] = global_model_name, local_model_name

    g_model_path = os.path.join(model_path, f'global_{global_model_name}_best.pth')
    l_model_path = os.path.join(model_path, f'local_{local_model_name}_best.pth')
    f_model_path = os.path.join(model_path, 'fusion_fusion_best.pth')

    # prepare model
    print("Loading models...")
    g_model, g_input_size, _, g_fm_name, g_pool_name, g_fm_size = initialize_model(params, global_model_name)
    l_model, l_input_size, _, _, l_pool_name, l_fm_size = initialize_model(params, local_model_name)
    if params['USE_EXTRA_INPUT']:
        s = g_fm_size + l_fm_size + 3
    else:
        s = g_fm_size + l_fm_size
    f_model = SimpleCLF(input_size=s).to(device)
    g_model.load_state_dict(torch.load(g_model_path))
    l_model.load_state_dict(torch.load(l_model_path))
    f_model.load_state_dict(torch.load(f_model_path))

    g_criterion = nn.BCEWithLogitsLoss()
    l_criterion = nn.BCEWithLogitsLoss()
    f_criterion = nn.BCEWithLogitsLoss()
    criterions = [g_criterion, l_criterion, f_criterion]
    test_loss, test_auc, y_probs, y_preds, y_true = eval_models(params, [g_model, l_model, f_model], loader, criterions)
    #print(f"Global Test loss: {test_loss[0]}; Global Test ROC: {test_auc[0]}")
    #print(f"Local Test loss: {test_loss[1]}; Global Test ROC: {test_auc[1]}")
    print(f"Fusion Test loss: {test_loss[2]}; Global Test ROC: {test_auc[2]}")
    return y_probs[-1], y_preds[-1], y_true


# FIXME
def hard_voting(list_probs):
    """
    Inputs: list of Numpy.array objects.
    """
    t_list = np.stack(list_probs)
    max_tensor = functools.reduce(np.maximum, t_list)
    return max_tensor


def soft_voting(list_probs):
    """
    Inputs: list of Numpy.array objects.
    """
    # assume all classifiers have same weight
    #wavg_t = np.mean(list_probs, axis=0)
    wavg_t = sum(list_probs) / len(list_probs)
    return wavg_t

###############################################
# change the name and path to the model to load
model_dir_1 = '../best_models/final/R50-R50'
model_dir_2 = '../best_models/final/Rx50-Rx50'
model_dir_3 = '../best_models/final/Rx101-Rx101'
###############################################

if __name__ == "__main__":
    # prepare data
    print("Loading test data...")
    df_data, lst_labels = load_data_file(sampling=0)
    # Map gender and view position to {0,1}
    gender_dict = {'M': 0, 'F': 1}
    view_dict = {'PA': 0, 'AP': 1}
    df_data.replace({"Patient Gender": gender_dict,
                     "View Position": view_dict},
                    inplace=True)
    df_train_val, df_test = make_train_test_split(df_data)
    df_train, _ = make_train_val_split(params, df_train_val)
    # standardize the age
    age_mean = np.mean(df_train['Patient Age'])  # use training data to prevent leakage
    age_std = np.std(df_train['Patient Age'])
    df_test['Patient Age'] = (df_test['Patient Age'] - age_mean) / age_std

    tfx = make_data_transform(config.GLOBAL_IMAGE_SIZE)
    test_loader = load_data(params,
                            df_test,
                            transform=tfx['test'],
                            shuffle=False,
                            num_workers=7)
    print("Evaluating...")
    y_probs_1, _, y_true = get_predictions(params, model_dir_1, 'resnet50', 'resnet50', test_loader)
    y_probs_2, _, _ = get_predictions(params, model_dir_2, 'resnext50', 'resnext50', test_loader)
    y_probs_3, _, _ = get_predictions(params, model_dir_3, 'resnext101', 'resnext101', test_loader)
    #hard_pred = hard_voting([y_probs_1, y_probs_2, y_probs_3])
    soft_prob = soft_voting([y_probs_1, y_probs_2, y_probs_3])
    # print results
    #hard_macro_roc_auc, hard_roc_aucs = calculate_metric(hard_pred, y_true)
    soft_macro_roc_auc, soft_roc_aucs = calculate_metric(soft_prob, y_true)


    print(f"Disease:{'':<22}AUROC")
    for i, lb in enumerate(config.TEXT_LABELS):
        print(f"{lb:<30}: {soft_roc_aucs[i]:.4f}")
    print(f"\nROCAUC (Macro): {soft_macro_roc_auc}")

    print("End of script.")
