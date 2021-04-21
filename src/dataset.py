import sys
import config

import random
import os
import torch
from torch.utils.data import Dataset

import torchvision.transforms as transforms
from PIL import Image

sys.path.insert(0, '../src')


random.seed(config.SEED)
torch.manual_seed(config.SEED)
os.environ["PYTHONHASHSEED"] = str(config.SEED)


def make_data_transform(input_size=224):
    m = [0.485, 0.456, 0.406]
    sd = [0.229, 0.224, 0.225]
    return {
                'train': transforms.Compose([
                    transforms.Resize(input_size),
                    #transforms.RandomResizedCrop(input_size),
                    #transforms.RandomCrop(input_size),
                    transforms.CenterCrop(input_size),
                    #transforms.RandomHorizontalFlip(),  # data augmentation
                    transforms.ToTensor(),
                    transforms.Normalize(mean=m, std=sd)
                ]),
                'test': transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=m, std=sd)
                ]),
            }


class NihDataset(Dataset):
    def __init__(self, dataframe, transform=None, greyscale=False):
        self.dataframe = dataframe
        self.transform = transform
        self.greyscale = greyscale

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # via .getband(), some images are know to have 4 channels. Here we convert them to 3-channel RGB.
        image = Image.open(self.dataframe.loc[idx, 'Image_path'])
        
        if self.greyscale:
            image = image.convert('L')
        else:
            image = image.convert('RGB')
            
        target = self.dataframe.loc[idx, 'labels']  # 'labels' is a 14/15-dim vector
        target = torch.FloatTensor(target)

        if self.transform:
            image = self.transform(image)

        return image, target


def load_data(dataframe, batch_size=config.BATCH_SIZE, transform=None, shuffle=True, num_workers=config.NUM_WORKERS, greyscale=False):
    """
    Data Loader with batch loading and transform.
    """
    image_data = NihDataset(dataframe, transform=transform, greyscale=greyscale)
    pin = config.DEVICE == 'cpu'
    loader = torch.utils.data.DataLoader(image_data,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         num_workers=num_workers,
                                         pin_memory=pin)
    return loader

