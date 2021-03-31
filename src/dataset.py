import sys  
sys.path.insert(0, '../src')
import config
import util

import random
import os
import torch
from torch.utils.data import Dataset

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from PIL import Image

random.seed(config.SEED)
torch.manual_seed(config.SEED)
os.environ["PYTHONHASHSEED"] = str(config.SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop((112,112)),
        transforms.RandomHorizontalFlip(),  # data augmentation
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((128,128)),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class NihDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        image  = Image.open(self.dataframe.loc[idx, 'Image_path']).convert('L')  # via .getband(), some images are know to have 4 channels. Here we convert them to 1-channel grayscale.
        target = self.dataframe.loc[idx, 'Labels']
            
        if self.transform:
            image = self.transform(image)
        
        return image, target

def load_data(dataframe, transform=None, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, num_workers=config.TRAIN_WORKERS):
    '''
    Data Loader with batch loading and transform.
    '''
    image_data = NihDataset(dataframe, transform=transform)
    pin = device=='cpu'
    num_workers=num_workers*int(device!='cpu')
    loader = torch.utils.data.DataLoader(image_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin)
    return loader


