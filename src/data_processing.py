#!/usr/bin/env python
# coding: utf-8

# # Notebook for preprocessing the data

# import necessary Libraries
#
# It is important to set path for the python source code


import sys
import os
import pandas as pd
import random
from glob import glob
import config
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import model_selection

sys.path.insert(0, '../src')
random.seed(config.SEED)


def load_data_file(root_dir=config.ROOT_PATH, data_dir=config.PROCESSED_DATA_DIR, index_file=config.INDEX_FILE, sampling=config.SAMPLING):
    """
    load the data file (with image indices, image path, labels etc.)
    """
    df = pd.read_csv(os.path.join(root_dir, data_dir, index_file))

    # rename some columns
    df.rename(columns={
        "OriginalImage[Width": "Width", 
        "Height]": "Height", 
        "OriginalImagePixelSpacing[x": "X", 
        "y]": "Y", 
        "Follow-up #": "Follow Up",
        }, inplace=True)

    # parse image paths
    image_paths = glob(os.path.join(root_dir, data_dir, '*.png'))
    assert len(df) == len(image_paths), f'Mismatch in index file and image count. {len(df)};{len(image_paths)}'
    df['Image_path'] = df['Image Index'].map(
        {os.path.basename(image_path): os.path.relpath(image_path) for image_path in image_paths})

    # one-hot encode labels to columns
    df['targets'] = df['Finding Labels'].str.split("|", expand=False)  # some images have >1 labels, split to list
    diseases = set([item for sublist in df['targets'].tolist() for item in sublist])  # get the set of labels
    diseases = sorted(list(diseases))
    # move the 'no finding' label to tail
    ###########################################################
    # This part controls which labels are used
    if config.NUM_CLASSES == 15:
        diseases.append(diseases.pop(diseases.index('No Finding')))  # 15 labels
    elif config.NUM_CLASSES == 14:
        diseases.pop(diseases.index('No Finding'))  # 14 labels (remove No Finding)
    else:
        raise NotImplementedError(f"Number of classes must be 14 or 15. Got {config.NUM_CLASSES}")
    ###########################################################
    # make each label column
    mlb = MultiLabelBinarizer(sparse_output=True)
    df = df.join(
                pd.DataFrame.sparse.from_spmatrix(
                    mlb.fit_transform(df.pop('targets')),
                    index=df.index,
                    columns=mlb.classes_))

    # make a label vector
    df['labels'] = df[diseases].values.tolist()

    if sampling > 0:
        df = df.sample(sampling).reset_index()

    return df, diseases


def make_train_test_split(df_data, train_val_list_file=config.TRAIN_VAL_FILE, test_list_file=config.TEST_FILE,
                          root_dir=config.ROOT_PATH, data_dir=config.PROCESSED_DATA_DIR):
    with open(os.path.join(root_dir, data_dir, train_val_list_file)) as f:
        train_val_list = [x.strip() for x in f.readlines()]

    with open(os.path.join(root_dir, data_dir, test_list_file)) as f:
        test_list = [x.strip() for x in f.readlines()]

    df_train = df_data[df_data['Image Index'].isin(train_val_list)]
    df_valid = df_data[df_data['Image Index'].isin(test_list)]
    df_train.reset_index(drop=True)
    df_valid.reset_index(drop=True)

    return df_train.reset_index(drop=True), df_valid.reset_index(drop=True)


def train_test_split(df, test_size=config.VAL_SIZE):
    df_train, df_test = model_selection.train_test_split(df, test_size=test_size, random_state=config.SEED)
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


