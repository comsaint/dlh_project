#!/usr/bin/env python
# coding: utf-8

# # Notebook for preprocessing the data

# import necessary Libraries
# 
# It is important to set path for the python source code


import sys  
sys.path.insert(0, '../src')

import os
import numpy as np
import pandas as pd
import random

from glob import glob

import config

import matplotlib.pyplot as plt
import matplotlib.image as img

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import model_selection 

random.seed(config.SEED)


def load_image_indices (root_dir=config.ROOT_PATH, data_dir=config.INPUT_DATA_DIR, index_file=config.INDEX_FILE) :
  indices = pd.read_csv(os.path.join(root_dir, data_dir, index_file))
  indices.drop(columns=['Unnamed: 11'], axis=1, inplace=True)
  indices.rename(columns={"OriginalImage[Width": "Width", "Height]": "Height", 
                            "OriginalImagePixelSpacing[x": "X", "y]": "Y", "Follow-up #": "Follow Up"}, inplace=True)
  return indices


def load_image_paths (df_indides=None, root_dir=config.ROOT_PATH, data_dir=config.INPUT_DATA_DIR) :
  if not df_indides:
    df_indices = load_image_indices(root_dir=root_dir, data_dir=data_dir)
  image_paths = glob(os.path.join(root_dir, data_dir, 'images', '*.png'))
  if len(df_indices) == len(image_paths):
    df_indices['Image_path'] = df_indices['Image Index'].map({os.path.basename(image_path): os.path.relpath(image_path) for image_path in image_paths})
  else:
    raise Exception('miamatch in index file and image count')
  return df_indices
  

def load_image_labels (df_indices):
  labels = list(set([label for label_list in df_indices['Finding Labels'].str.split("|", expand = False) for label in label_list]))
  return sorted(labels)



def draw_image (df, i=random.randint(0,99999)) :
  if i >= len(df):
    i = len(df)-1
  plt.imshow(img.imread(df['Image_path'][i]), cmap = 'bone')
  plt.title(df['Finding Labels'][i])



def load_data(disease_filter_list=None, position_filter_list=None):
    df_indices = load_image_paths()
    lst_labels = load_image_labels(df_indices)
    lst_labels.remove('No Finding')
    
    df_data  = df_indices[['Patient ID','Patient Age','Patient Gender','Follow Up','View Position','Image Index','Image_path','Finding Labels']]

    if disease_filter_list:
        lst_labels = disease_filter_list
        #df_data = df_data[df_data['Finding Labels'].isin(disease_filter_list)]
          
    if position_filter_list:
        df_data = df_data[df_data['View Position'].isin(position_filter_list)]
        
    return df_data, lst_labels

    
def multi_hot_label(df_data, labels):
  mb = MultiLabelBinarizer(sparse_output=True)
  df_data = df_data.join(pd.DataFrame.sparse.from_spmatrix(mb.fit_transform(df_data['Finding Labels'].str.split("|", expand = False)),columns=mb.classes_,index=df_data.index))
  df_data.drop(columns=['Finding Labels'], axis=1, inplace=True)
  df_data['Labels'] = list(df_data[labels].values)
  df_data.drop(columns=labels, axis=1, inplace=True)    
  
  dict_labels = {i: x for i,x in enumerate(labels)}
  
  return df_data, dict_labels


def make_train_test_split(df_data, train_val_list_file=config.TRAIN_VAL_FILE, test_val_list_file=config.TEST_VAL_FILE, root_dir=config.ROOT_PATH, data_dir=config.INPUT_DATA_DIR):
    with open(os.path.join(root_dir, data_dir, train_val_list_file)) as f:
        train_val_list = [x.strip() for x in f.readlines()]
    
    with open(os.path.join(root_dir, data_dir, test_val_list_file)) as f:
        test_val_list  = [x.strip() for x in f.readlines()]
        
    df_train = df_data[df_data['Image Index'].isin(train_val_list)]
    df_valid = df_data[df_data['Image Index'].isin(test_val_list)]    
    df_train.reset_index(drop=True)
    df_valid.reset_index(drop=True)
    
    return df_train.reset_index(drop=True), df_valid.reset_index(drop=True)
  
def train_test_split(df, test_size=config.TEST_SIZE):
    df_train, df_test = model_selection.train_test_split(df, test_size=test_size, random_state=config.SEED)
    
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


def main():
    df_data, labels = load_data()
    df_data, dict_labels = multi_hot_label(df_data, labels)
    df_train, df_test = train_test_split(df_data)
    
if __name__ == "__main__":
    main()
    
    