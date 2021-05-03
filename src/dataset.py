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


def make_data_transform(input_size=224, additional_transforms=None):
    m = [0.485, 0.456, 0.406]
    sd = [0.229, 0.224, 0.225]

    code_map = {
        'rot': transforms.RandomRotation(15),
        'hflip': transforms.RandomHorizontalFlip(p=0.5),
        'vflip': transforms.RandomVerticalFlip(p=0.5),
        'rcp': transforms.RandomResizedCrop(size=224),
    }

    train_transform = transforms.Compose([
                        transforms.Resize(input_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=m, std=sd)])
    if additional_transforms:
        for _t in additional_transforms[::-1]:
            train_transform.transforms.insert(0, code_map[_t])

    test_transform = transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=m, std=sd)
                ])
    return {
                'train': train_transform,
                'test': test_transform
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
        else:
            tmp_tf = transforms.Compose([
                    transforms.ToTensor(),
                ])
            image = tmp_tf(image)

        extra_features = torch.FloatTensor(
            self.dataframe.loc[idx, ['Patient Age', 'Patient Gender', 'View Position']].values
        )
        return image, target, extra_features


def load_data(params, dataframe, transform=None, shuffle=True, num_workers=config.NUM_WORKERS, greyscale=config.GREY_SCALE):
    """
    Data Loader with batch loading and transform.
    """
    image_data = NihDataset(dataframe, transform=transform, greyscale=greyscale)
    pin = config.DEVICE == 'cpu'
    loader = torch.utils.data.DataLoader(image_data,
                                         batch_size=params['BATCH_SIZE'],
                                         shuffle=shuffle,
                                         num_workers=num_workers,
                                         pin_memory=pin)
    return loader
