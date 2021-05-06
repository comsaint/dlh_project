"""
A script for quick testing of a trained model.
"""
from dataset import load_data, make_data_transform
from data_processing import load_data_file, make_train_test_split
from model import initialize_model, SimpleCLF
from train_model import eval_models
import torch
from torch import nn
import config
import os
from config import params

device = 'cuda' if torch.cuda.is_available() else 'cpu'

###############################################
# change the name and path to the model to load
g_model_name = 'mobilenet'
l_model_name = 'mobilenet'

model_dir = '../models/model_experiment_fullmodel_mobilenet_mobilenet_5.0_16_70.0_classw_extra_tune_rot-hflip_trial2'
###############################################
g_model_path = os.path.join(model_dir, f'global_{g_model_name}_best.pth')
l_model_path = os.path.join(model_dir, f'local_{l_model_name}_best.pth')
f_model_path = os.path.join(model_dir, 'fusion_fusion_best.pth')

if __name__ == "__main__":
    # prepare model
    print("Loading model...")
    g_model, g_input_size, _, g_fm_name, g_pool_name, g_fm_size = initialize_model(params, g_model_name)
    l_model, l_input_size, _, _, l_pool_name, l_fm_size = initialize_model(params, l_model_name)
    if params['USE_EXTRA_INPUT']:
        s = g_fm_size + l_fm_size + 3
    else:
        s = g_fm_size + l_fm_size
    f_model = SimpleCLF(input_size=s).to(device)
    g_model.load_state_dict(torch.load(g_model_path))
    l_model.load_state_dict(torch.load(l_model_path))
    f_model.load_state_dict(torch.load(f_model_path))

    # prepare data
    print("Loading test data...")
    df_data, lst_labels = load_data_file(sampling=0)
    # Map gender and view position to {0,1}
    gender_dict = {'M': 0, 'F': 1}
    view_dict = {'PA': 0, 'AP': 1}
    df_data.replace({"Patient Gender": gender_dict,
                     "View Position": view_dict},
                    inplace=True)
    _, df_test = make_train_test_split(df_data)
    tfx = make_data_transform(config.GLOBAL_IMAGE_SIZE)
    test_loader = load_data(params,
                            df_test,
                            transform=tfx['test'],
                            shuffle=False,
                            num_workers=7)
    print("Evaluating...")
    g_criterion = nn.BCEWithLogitsLoss()
    l_criterion = nn.BCEWithLogitsLoss()
    f_criterion = nn.BCEWithLogitsLoss()
    criterions = [g_criterion, l_criterion, f_criterion]
    test_loss, test_auc, _, _, _ = eval_models(params, [g_model, l_model, f_model], test_loader, criterions)
    print(f"Global Test loss: {test_loss[0]}; Global Test ROC: {test_auc[0]}")
    print(f"Local Test loss: {test_loss[1]}; Global Test ROC: {test_auc[1]}")
    print(f"Fusion Test loss: {test_loss[2]}; Global Test ROC: {test_auc[2]}")

    print("End of script.")