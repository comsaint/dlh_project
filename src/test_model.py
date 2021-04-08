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

model_name = 'alexnet'
model_path = '..\\models\\alexnet_Atelectasis_best.pth'  #

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    # prepare model
    m, input_size = initialize_model(model_name, num_classes=2, use_pretrained=False, feature_extract=False)
    m.load_state_dict(torch.load(model_path))

    # prepare data
    df_data, lst_labels = load_data_file(sampling=config.SAMPLING)
    _, df_test = make_train_test_split(df_data)
    tfx = make_data_transform(input_size)
    test_loader = load_data(df_test, config.DISEASE, batch_size=config.BATCH_SIZE, transform=tfx['test'], shuffle=False,
                            num_workers=0)

    eval_model(m.to(device), test_loader, criterion=nn.CrossEntropyLoss())
