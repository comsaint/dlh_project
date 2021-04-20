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

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_data_transform(input_size=224):
    return {
                'train': transforms.Compose([
                    transforms.Resize(input_size),
                    #transforms.RandomResizedCrop(input_size),  # usually 224
                    #transforms.RandomCrop(input_size),  # usually 224
                    transforms.RandomHorizontalFlip(),  # data augmentation
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),
                'test': transforms.Compose([
                    transforms.Resize(input_size),
                    #transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        target = self.dataframe.loc[idx, 'labels']  # label is a 15-dim vector
        target = torch.FloatTensor(target)

        if self.transform:
            image = self.transform(image)

        return image, target


def load_data(dataframe, batch_size=config.BATCH_SIZE, transform=None, shuffle=True, num_workers=config.NUM_WORKERS):
    """
    Data Loader with batch loading and transform.
    """
    image_data = NihDataset(dataframe, transform=transform)
    pin = device == 'cpu'
    loader = torch.utils.data.DataLoader(image_data,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         num_workers=num_workers,
                                         pin_memory=pin)
    return loader
