import pandas as pd
import sys
import random
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from glob import glob
from torchvision import models
import torch.nn as nn
from pytorch_grad_cam import CAM, GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image
import cv2
import numpy as np
from model import initialize_model,set_parameter_requires_grad

bbox_list = pd.read_csv("../data/processed/BBox_List_2017.csv")

def load_data_file(df,data_dir='../data/processed/'):
    """
    load the data file (with image indices, image path, labels etc.)
    """

    # parse image paths
    image_paths = glob(os.path.join(data_dir, '*.png'))
#     print(image_paths)
    # assert len(df) == len(image_paths), f'Mismatch in index file and image count. {len(df)};{len(image_paths)}'
    df['Image_path'] = df['Image Index'].map(
        {os.path.basename(image_path): os.path.relpath(image_path) for image_path in image_paths})


    return df,df['Finding Label']

df_data, label= load_data_file(bbox_list)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def make_data_transform(input_size):
    return {
                'train': transforms.Compose([
                    transforms.Resize(input_size), 
                    # transforms.RandomResizedCrop(input_size), # usually 224
                    # transforms.RandomHorizontalFlip(),  # data augmentation
                    transforms.ToTensor(),
                ]),
                'test': transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                ]),
            }


class NihDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # via .getband(), some images are know to have 4 channels. Here we convert them to 3-channel RGB.
        image = Image.open(self.dataframe.loc[idx, 'Image_path']).convert('RGB')
        target = self.dataframe.loc[idx, 'Finding Label']  # label is a 15-dim vector
#         target = torch.FloatTensor(target)

        if self.transform:
            image = self.transform(image)

        return (image,target,self.dataframe.loc[idx, 'Image Index'])


def load_data(dataframe, batch_size=4, transform=None, shuffle=True, num_workers=7):
    """
    Data Loader with batch loading and transform.
    """
    image_data= NihDataset(dataframe, transform=transform)
    pin = device == 'cpu'
    loader = torch.utils.data.DataLoader(image_data,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         num_workers=num_workers,
                                         pin_memory=pin)

    return loader


tfx = make_data_transform(224)

train_data_loader = load_data(df_data,transform=tfx['test'] ,shuffle=False)

model, input_size,use_model_loss = initialize_model("resnext50", num_classes=14, use_pretrained=False, feature_extract=False)
model.load_state_dict(torch.load('resnext50_best_valloss_auc7216.pth'))
# target_layer = model.features[-1]
target_layer = model.layer4[-1]


Path="../heatmap/"

cam = CAM(model=model, target_layer=target_layer,use_cuda=True)

labels_dict = {
    "Atelectasis": 0,
    "Cardiomegaly": 1,
    "Consolidation": 2,
    "Edema": 3,
    "Effusion": 4,
    "Emphysema": 5,
    "Fibrosis": 6,
    "Hernia": 7,
    "Infiltrate": 8,
    "Mass": 9,
    "Nodule": 10,
    "Pleural_Thickening": 11,
    "Pneumonia": 12,
    "Pneumothorax": 13
    }
size_upsample = (224, 224)


for (inputs, labels,name) in train_data_loader:
    bz, nc, h, w = inputs.shape
    
    for i in range(0, bz):
        image = inputs[i].cpu().numpy().reshape(224,224,3)
        image = image[int(224*0.334):int(224*0.667),int(224*0.334):int(224*0.667),:]
        image = cv2.resize(image, size_upsample)

        
        target_category = labels_dict[labels[i]]

        input_tensor = preprocess_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        grayscale_cam = cam(input_tensor=input_tensor,method='gradcam',target_category=target_category)

        mask =(grayscale_cam > .7).astype(int)*255
        
        cam_image = show_cam_on_image(image, grayscale_cam)
        
        # cv2.imwrite(Path+f'{name[i]}_cam_{labels[i]}.jpg', cam_image)  
        # cv2.imwrite(Path+f'{name[i]}_mask_{labels[i]}.jpg', mask)
        ind = np.argwhere(mask != 0)
        minh = min(ind[:,0])
        minw = min(ind[:,1])
        maxh = max(ind[:,0])
        maxw = max(ind[:,1])
        
        b_x = minw
        b_y = minh
        b_w = (maxw - minw)
        b_h = (maxh - minh)
        
        file1 = open(Path+"bbox_list_resnext50.txt","a")
        file1.write(name[i]+","+labels[i]+","+str(b_x)+","+str(b_y)+","+str(b_w)+","+str(b_h)+"\n") 
        file1.close()
    # break        

    
#         cv2.imwrite(Path+f'{name[i]}_cam_{labels[i]}.jpg', cam_image)  
#         cv2.imwrite(Path+f'.{name[i]}_mask_{labels[i]}.jpg', mask)
