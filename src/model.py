import sys
import torch.nn as nn
from torchvision import models
from config import FINE_TUNE, NUM_CLASSES, DEVICE, USE_EXTRA_INPUT
import torch
from caps_net import CapsNet, CapsNetworks

sys.path.insert(0, '../src')


def get_hook_names(model_name):
    if model_name in ['resnet50', 'resnext50', 'resnext101']:
        fm_name = "layer4.2.relu"  # TODO: or layer4.2.conv3 ?
        pool_name = "avgpool"
    elif model_name == 'densenet':
        fm_name = "features.norm5"
        pool_name = "features.norm5"
    elif model_name == 'fc':
        fm_name, pool_name = '', ''
    elif model_name.startswith('capsnet'):
        fm_name = "conv"
        pool_name = "digit"
    else:
        raise NotImplementedError(f"Feature map and pooling hooks not implemented for model '{model_name}'.")
    return fm_name, pool_name


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name,
                     num_classes=NUM_CLASSES,
                     fine_tune=FINE_TUNE):
    use_model_loss = False
    if model_name == "resnet":
        """ 
        Resnet18
        """
        if fine_tune:  # Tune cls layer, then all
            model_ft = models.resnet18(pretrained=True)
            set_parameter_requires_grad(model_ft, feature_extracting=True)  # extract
        else:  # train from scratch
            model_ft = models.resnet18(pretrained=False)
            set_parameter_requires_grad(model_ft, feature_extracting=False)  # extract
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    elif model_name == "densenet":
        """ 
        Densenet
        """
        if fine_tune:  # Tune cls layer, then all
            model_ft = models.densenet121(pretrained=True)
            set_parameter_requires_grad(model_ft, feature_extracting=True)  # extract
        else:  # train from scratch
            model_ft = models.densenet121(pretrained=False)
            set_parameter_requires_grad(model_ft, feature_extracting=False)  # extract
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        fm_size = 1024 * 7 * 7

    elif model_name == "resnext50":
        """
        ResNeXt-50
        """
        if fine_tune:  # Tune cls layer, then all
            model_ft = models.resnext50_32x4d(pretrained=True)
            set_parameter_requires_grad(model_ft, feature_extracting=True)  # extract
        else:  # train from scratch
            model_ft = models.resnext50_32x4d(pretrained=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        fm_size = 2048 * 1 * 1

    elif model_name == "resnet50":
        """
        ResNet-50
        """
        if fine_tune:  # Tune cls layer, then all
            model_ft = models.resnet50(pretrained=True)
            set_parameter_requires_grad(model_ft, feature_extracting=True)  # extract
        else:  # train from scratch
            model_ft = models.resnet50(pretrained=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        fm_size = 2048 * 1 * 1

    elif model_name == "resnext101":
        """
        ResNeXt-101-32x8d
        """
        if fine_tune:  # Tune cls layer, then all
            model_ft = models.resnext101_32x8d(pretrained=True)
            set_parameter_requires_grad(model_ft, feature_extracting=True)  # extract
        else:  # train from scratch
            model_ft = models.resnext101_32x8d(pretrained=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        fm_size = 2048 * 1 * 1

    # FIXME: add hooks to models.
    elif model_name == "capsnet":
        """ 
        Capsnet 28x28 3 Chnl in, 16 chnl out
        """
        input_size = 28
        use_model_loss = True
        model_ft = CapsNet(img_size=input_size, img_channels=3,conv_out_channels=256, out_channels=16, num_classes=num_classes,conv_kernel_size=9,use_model_loss=use_model_loss)         
        fm_size = (num_classes) * 16 * 1

    elif model_name == "capsnet28_3_16":
        """ 
        Capsnet 28x28 3 Chnl in, 16 chnl out
        """
        input_size = 28
        model_ft = CapsNet(img_size=input_size, img_channels=3,conv_out_channels=256, out_channels=16, num_classes=num_classes,conv_kernel_size=9) 
        use_model_loss = True
        fm_size = (num_classes) * 16 * 1

    elif model_name == "capsnet28_3_32":
        """ 
        Capsnet
        """
        input_size = 28
        model_ft = CapsNet(img_size=input_size, img_channels=3,conv_out_channels=256, out_channels=32, num_classes=num_classes,conv_kernel_size=9) 
        use_model_loss = True
        fm_size = 32 * (num_classes) * 1

    elif model_name == "capsnet56_3_32":
        """ 
        Capsnet
        """
        input_size = 56
        model_ft = CapsNet(img_size=input_size, img_channels=3,conv_out_channels=256, out_channels=32, num_classes=num_classes,conv_kernel_size=9) 
        use_model_loss = True       
        fm_size = 32 * (num_classes) * 1

    elif model_name == "capsnet+densenet":
        """ 
        Capsnet + densenet
        """
        input_size = 224
        image_factor = 8
        img_size=int(input_size/image_factor)

        if fine_tune:  # Tune cls layer, then all
            model_ft = models.densenet121(pretrained=True)
            set_parameter_requires_grad(model_ft, feature_extracting=True)  # extract
        else:  # train from scratch
            model_ft = models.densenet121(pretrained=False)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, img_size*img_size*3 )
        model_ft = CapsNet(img_size=img_size, img_channels=3,conv_out_channels=256, out_channels=16, num_classes=num_classes,conv_kernel_size=9,conv_unit_model=model_ft, image_factor=image_factor)        
        use_model_loss = True
        fm_size = (num_classes) * 16 * 1       

    elif model_name == "capsnet+resnext50":
        """ 
        Capsnet + resnext50
        """
        input_size = 224
        image_factor = 4
        img_size=int(input_size/(image_factor))
        if fine_tune:  # Tune cls layer, then all
            model_ft = models.resnext50_32x4d(pretrained=True)
            set_parameter_requires_grad(model_ft, feature_extracting=True)  # extract
        else:  # train from scratch
            model_ft = models.resnext50_32x4d(pretrained=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        model_ft.load_state_dict(torch.load('../models/resnext50_10epoch.pth'))
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, (img_size*img_size*3) )
        model_cp = CapsNet(img_size=img_size, img_channels=3,conv_out_channels=256, out_channels=16, num_classes=num_classes,conv_kernel_size=9, image_factor=image_factor)
        model_cp.load_state_dict(torch.load('../models/capsnet56_3_32_5epoch.pth'))
        
        model_ft = CapsNetworks(preNet=model_ft, capsNet=model_cp)
        use_model_loss = False              
        fm_size = 16 * (num_classes) * 1       

    else:
        raise Exception(f"Invalid model name '{model_name}'")

    fm_name, pool_name = get_hook_names(model_name)
    model_ft = model_ft.to(DEVICE)

    return model_ft, input_size, use_model_loss, fm_name, pool_name, fm_size


class SimpleCLF(nn.Module):

    def __init__(self, input_size, output_size=NUM_CLASSES):
        super(SimpleCLF, self).__init__()
        # an affine operation: y = Wx + b
        if USE_EXTRA_INPUT:
            input_size += 3
        self.fc = nn.Linear(input_size, NUM_CLASSES)

    def forward(self, global_pool, local_pool, extra_features):
        if extra_features is None:
            fusion = torch.cat((global_pool, local_pool), 1)
        else:
            fusion = torch.cat((global_pool, local_pool, extra_features), 1)
        x = self.fc(fusion)
        return x
