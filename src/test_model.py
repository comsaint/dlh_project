"""
A script for quick testing of a trained model.
"""
from dataset import load_data, make_data_transform
from data_processing import load_data_file, make_train_test_split
from model import initialize_model
from train_model import eval_model
import torch
from torch import nn
import config

###############################################
# change the name and path to the model to load
model_name = 'resnext50'
model_path = '..\\models\\model_experiment_resnext_2\\resnext50_best.pth'
###############################################

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    # prepare model
    print("Loading model...")
    m, input_size, _ = initialize_model(model_name,
                                        num_classes=config.NUM_CLASSES,
                                        fine_tune=config.FINE_TUNE)
    m.load_state_dict(torch.load(model_path))

    # prepare data
    print("Loading test data...")
    df_data, lst_labels = load_data_file(sampling=0)
    _, df_test = make_train_test_split(df_data)
    tfx = make_data_transform(input_size)
    test_loader = load_data(df_test,
                            batch_size=config.BATCH_SIZE,
                            transform=tfx['test'],
                            shuffle=False,
                            num_workers=7)
    print("Evaluating...")
    eval_model(m, test_loader, criterion=nn.BCEWithLogitsLoss(), verbose=True)
    print("End of script.")
