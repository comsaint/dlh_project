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
import torch.backends.cudnn as cudnn
from utils import calculate_metric

device = 'cuda' if torch.cuda.is_available() else 'cpu'

###############################################
# change the name and path to the model to load
model_name = 'resnext101'
model_path = '../best_models/resnext101_best.pth'
###############################################

if __name__ == "__main__":
    # prepare model
    print("Loading model...")
    model, input_size, _, _, _, _ = initialize_model(params, model_name)
    model.load_state_dict(torch.load(model_path))

    # prepare data
    print("Loading test data...")
    df_data, lst_labels = load_data_file()
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
    criterion = nn.BCEWithLogitsLoss()

    y_true = []
    y_prob, y_pred = [], []
    loss = 0.0

    with torch.no_grad():
        model.eval()
        cudnn.benchmark = True
        for i, (inputs, labels, _) in enumerate(test_loader):
            labels = labels.to(config.DEVICE)  # true labels
            y_true.append(labels)
            probs = model(inputs.to(config.DEVICE))  # pass through
            loss += float(torch.mean(criterion(probs, labels)))

            predicted = probs > 0.5
            y_prob.append(probs)
            y_pred.append(predicted)
    loss /= len(test_loader)

    # convert to numpy
    y_prob = torch.cat(y_prob).detach().cpu().numpy()
    y_pred = torch.cat(y_pred).detach().cpu().numpy()
    y_true = torch.cat(y_true).detach().cpu().numpy()

    macro_roc_auc, roc_aucs = calculate_metric(y_prob, y_true)

    # print results
    print("'" * 20)
    print(f"Disease:{'':<22}AUROC")
    for i, lb in enumerate(config.TEXT_LABELS):
        print(f"{lb:<30}: {roc_aucs[i]:.4f}")
    print(f"\nROCAUC (Macro): {macro_roc_auc:.4f}")
    print("'" * 20)
    print("End of script.")
