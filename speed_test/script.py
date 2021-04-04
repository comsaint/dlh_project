# user-specific setting
PROJECT = 'mcsds-dlh'  # CHANGE: billing project name (since the dataset is user-to-pay)
DATA_FOLDER = '../data/'
disease = 'Atelectasis'
model_name = 'densenet'
num_classes = 2
feature_extract = True
learning_rate = 0.005
num_epochs = 1
batch_size = 64
num_workers = 0

# import libraries
import pandas as pd
import numpy as np
import random
import os
import time
import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from functions import load_data, train_model, eval_model, initialize_model

# set seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

if __name__ == '__main__':
    # check if CUDA is available (GPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # load train test split
    with open('train_val_list.txt') as f: 
        train_val_list = [x.strip() for x in f.readlines()]
    with open('test_list.txt') as f:
        test_list = [x.strip() for x in f.readlines()]

    # load labels
    df_labels = pd.read_csv('Data_Entry_2017_v2020.csv')

    ##################Sample the full dataset#########################
    # COMMENT OUT IN PRODUCTION!!!
    df_labels = df_labels.sample(1000).reset_index()
    ##################################################################

    print(f"Number of images: {len(df_labels)}")
    # split the finding (disease) labels, to a list
    df_labels['targets'] = df_labels['Finding Labels'].str.split("|", expand = False)
    # look at available labels
    labels = set([item for sublist in df_labels['targets'].tolist() for item in sublist])

    print(f"Number of labels: {len(labels)}")
    print(f"Labels: {labels}")

    # one-hot encode labels to columns
    mlb = MultiLabelBinarizer(sparse_output=True)

    df_labels = df_labels.join(
                pd.DataFrame.sparse.from_spmatrix(
                    mlb.fit_transform(df_labels.pop('targets')),
                    index=df_labels.index,
                    columns=mlb.classes_))
    df_labels[list(labels)]=df_labels[list(labels)].sparse.to_dense()  # for easy .describe()

    # split into train_val and test sets
    df_train_val = df_labels[df_labels['Image Index'].isin(train_val_list)]
    df_test = df_labels[df_labels['Image Index'].isin(test_list)].reset_index()

    print(f"Number of train/val images: {len(df_train_val)}")
    print(f"Number of test images: {len(df_test)}")

    assert (len(df_train_val) + len(df_test)) == len(df_labels), "Total number of images does not equal to sum of train/val and test!"

    # 2 notes about train-val split:
    # 1. make sure the same patient NEVER appears in both sets, to avoid data leakage
    # 2. Stratify the sampling process to avoid bias, especially for imbalance class
    # TODO: how to cater for these 2 objectives at the same time?
    df_train, df_val = train_test_split(df_train_val, test_size=0.1, stratify=df_train_val[disease], random_state=seed)  # 10% val set, about half the size of test set
    df_train.reset_index(inplace=True)
    df_val.reset_index(inplace=True)

    assert len(df_train) + len(df_val) == len(df_train_val)

    # this statistics was computed from a small sample of training set
    sample_mean = np.repeat(np.array([129.76628483/255]), 3)
    sample_std = np.repeat(np.array([59.70063891/255]), 3)

    # Initialize the model for this run
    model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    model = model.to(device)

    # Print the model we just instantiated
    print(model)
    print(f"Input image size: {input_size}")

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),  # data augmentation
            transforms.ToTensor(),  # note that .ToTensor() scales input to [0.0,1.0]
            transforms.Normalize(sample_mean, sample_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize((256,256)),  # FIXME: how to cater for different `input_size`?
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(sample_mean, sample_std)
        ]),
    }

    # Class weight
    # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#class_weights
    num_neg = sum(df_train[disease] == 0)
    num_pos = sum(df_train[disease] == 1)
    assert num_neg + num_pos == len(df_train)
    print(f"# of negative/positive cases: {num_neg}:{num_pos}")
    class_weight = torch.FloatTensor([(1 / num_neg)*(len(df_train))/2.0, (1 / num_pos)*(len(df_train))/2.0]).to(device)
    print(f"Class weight: {class_weight}")
    criterion = nn.CrossEntropyLoss(weight=class_weight)  # note the class weight
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # data loader
    train_data_loader = load_data(df_train, DATA_FOLDER, disease, transform=data_transforms['train'], shuffle=True, batch_size=batch_size, num_workers=num_workers)
    val_data_loader = load_data(df_val, DATA_FOLDER, disease, transform=data_transforms['test'], shuffle=False, batch_size=batch_size, num_workers=num_workers)

    print(f"Training start. Mode: {device}")
    start_time = time.time()
    model, t_losses, v_losses, v_best_auc, v_roc, best_model_pth = train_model(model, train_data_loader, val_data_loader, criterion, optimizer, num_epochs=num_epochs, verbose=False)
    print(f"Best ROC achieved on validation set: {v_best_auc:3f}")
    print(f'Finished Training. Total time: {(time.time()-start_time)/60} minutes.')

    # load and test on the best model
    model.load_state_dict(torch.load(best_model_pth))
    model.eval()

    test_data_loader = load_data(df_test, DATA_FOLDER, disease, transform=data_transforms['test'], shuffle=False, batch_size=32, num_workers=num_workers)
    test_loss, test_auc, _, _, _ = eval_model(model.to(device), test_data_loader, criterion)
    print(f"Test loss: {test_loss}; Test ROC: {test_auc}")
    print("End of script.")
