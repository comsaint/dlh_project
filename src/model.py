import sys
import torch.nn as nn
from torchvision import models
from config import USE_PRETRAIN, FEATURE_EXTRACT, DEVICE
import torch

from caps_net import CapsNet, CapsNetworks

sys.path.insert(0, '../src')


def set_parameter_requires_grad(model, feature_extracting=FEATURE_EXTRACT):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes=14, use_pretrained=USE_PRETRAIN, feature_extract=FEATURE_EXTRACT):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    use_model_loss = False
    
    if model_name == "resnet":
        """ 
        Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ 
        Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ 
        VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ 
        Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ 
        Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """
        Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    elif model_name == "resnext50":
        """
        ResNeXt-50
        """
        model_ft = models.resnext50_32x4d(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet50":
        """
        ResNet-50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnext101":
        """
        ResNeXt-101-32x8d
        """
        model_ft = models.resnext101_32x8d(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "capsnet28_3_16":
        """ 
        Capsnet 28x28 3 Chnl in, 16 chnl out
        """
        input_size = 28
        model_ft = CapsNet(img_size=input_size, img_channels=3,conv_out_channels=256, out_channels=16, num_classes=num_classes,conv_kernel_size=9) 
        model_ft.load_state_dict(torch.load(f'../models/model_capsnet28_3_16/capsnet28_3_16_6epoch.pth', map_location=torch.device(DEVICE)))
        use_model_loss = True

    elif model_name == "capsnet28_3_32":
        """ 
        Capsnet
        """
        input_size = 28
        model_ft = CapsNet(img_size=input_size, img_channels=3,conv_out_channels=256, out_channels=32, num_classes=num_classes,conv_kernel_size=9) 
        use_model_loss = True

    elif model_name == "capsnet56_3_32":
        """ 
        Capsnet
        """
        input_size = 56
        model_ft = CapsNet(img_size=input_size, img_channels=3,conv_out_channels=256, out_channels=32, num_classes=num_classes,conv_kernel_size=9) 
        
        use_model_loss = True       

    elif model_name == "capsnet+densenet":
        """ 
        Capsnet + densenet
        """
        input_size = 224
        image_factor = 8
        img_size=int(input_size/image_factor)
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes )
        model_ft.load_state_dict(torch.load(f'../models/densenet_best.bpt', map_location=torch.device(DEVICE)))
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, img_size*img_size*3 )
        model_cp = CapsNet(img_size=img_size, img_channels=3,conv_out_channels=256, out_channels=16, num_classes=num_classes,conv_kernel_size=9, image_factor=image_factor)
        model_cp.load_state_dict(torch.load(f'../models/capsnet28_3_16_best.bpt', map_location=torch.device(DEVICE)))
        model_ft = CapsNetworks(preNet=model_ft, capsNet=model_cp)
        
        use_model_loss = True                       

    elif model_name == "capsnet+resnext50":
        """ 
        Capsnet + resnext50
        """
        input_size = 224
        image_factor = 8
        img_size=int(input_size/(image_factor))
        model_ft = models.resnext50_32x4d(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        model_ft.load_state_dict(torch.load(f'../models/resnext50_best.bpt', map_location=torch.device(DEVICE)))
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, (img_size*img_size*3) )
        model_cp = CapsNet(img_size=img_size, img_channels=3,conv_out_channels=256, out_channels=16, num_classes=num_classes,conv_kernel_size=9, image_factor=image_factor)
        model_cp.load_state_dict(torch.load(f'../models/capsnet28_3_16_best.bpt', map_location=torch.device(DEVICE)))
        
        model_ft = CapsNetworks(preNet=model_ft, capsNet=model_cp)
        use_model_loss = True    
        
    elif model_name == "densenet+capsnet":
        """ 
        Capsnet + densenet
        """
        input_size = 224
        image_factor = 8
        img_size=int(input_size/image_factor)
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes )
        #model_ft.load_state_dict(torch.load(f'../models/densenet_best.bpt', map_location=torch.device(DEVICE)))
        for param in model_ft.parameters():
            param.requires_grad = True
        for param in model_ft.features.norm5.parameters():
            param.requires_grad = True
        for param in model_ft.features.denseblock4.parameters():
            param.requires_grad = True
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier.weight.requires_grad = True
        model_ft.classifier.bias.requires_grad = True
        model_ft.classifier = nn.Linear(num_ftrs, img_size*img_size*3 )
        model_cp = CapsNet(img_size=img_size, img_channels=3,conv_out_channels=256, out_channels=16, num_classes=num_classes,conv_kernel_size=9, image_factor=image_factor)
        #model_cp.load_state_dict(torch.load(f'../models/model_capsnet28_3_16/capsnet28_3_16_best.pth', map_location=torch.device(DEVICE)))
        model_ft = CapsNetworks(preNet=model_ft, capsNet=model_cp)
        model_ft.load_state_dict(torch.load(f'../models/model_densenet+capsnet/densenet+capsnet_40epoch.pth', map_location=torch.device(DEVICE)))
        for child in model_ft.children():
            if str(type(child)) == "<class 'torchvision.models.densenet.DenseNet'>":
                for param in child.features.norm5.parameters():
                    param.requires_grad = True
                for param in child.features.denseblock4.parameters():
                    param.requires_grad = True
                for param in child.features.transition3.parameters():
                    param.requires_grad = True
                for param in child.features.denseblock3.parameters():
                    param.requires_grad = True
                for param in child.features.transition2.parameters():
                    param.requires_grad = True
                for param in child.features.denseblock2.parameters():
                    param.requires_grad = True
                for param in child.features.transition1.parameters():
                    param.requires_grad = True
        for param in model_ft.parameters():
            param.requires_grad = True
                        
        use_model_loss = True   
        
    elif model_name == "densenet+capsnet2":
        """ 
        Capsnet + densenet
        """
        input_size = 224
        image_factor = 8
        img_size=int(input_size/image_factor)
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes )
        model_ft.load_state_dict(torch.load(f'../models/densenet_best.bpt', map_location=torch.device(DEVICE)))
        model_ft.classifier = nn.Identity( 3 )
        model_cp = CapsNet(img_size=30, img_channels=1,conv_out_channels=256, out_channels=16, num_classes=num_classes,conv_kernel_size=9, image_factor=image_factor)
        #model_cp.load_state_dict(torch.load(f'../models/model_capsnet28_3_16/capsnet28_3_16_best.pth', map_location=torch.device(DEVICE)))
        model_ft = CapsNetworks(preNet=model_ft, capsNet=model_cp)
        use_model_loss = False
    
    elif model_name == "resnext101+capsnet":
        """ 
        Capsnet + resnext101
        """
        input_size = 224
        image_factor = 8
        img_size=int(input_size/image_factor)
        model_ft = models.resnext101_32x8d(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        model_ft.load_state_dict(torch.load(f'../models/resnext101/resnext101_best.pth', map_location=torch.device(DEVICE)))
        set_parameter_requires_grad(model_ft, False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, img_size*img_size*3 )
        model_ft.fc.weight.requires_grad = True
        model_ft.fc.bias.requires_grad = True
        model_cp = CapsNet(img_size=img_size, img_channels=3,conv_out_channels=256, out_channels=16, num_classes=num_classes,conv_kernel_size=9, image_factor=image_factor)
        model_cp.load_state_dict(torch.load(f'../models/model_capsnet28_3_16/capsnet28_3_16_best.pth', map_location=torch.device(DEVICE)))
        model_ft = CapsNetworks(preNet=model_ft, capsNet=model_cp)
        #model_ft.load_state_dict(torch.load(f'../models/model_resnext101+capsnet/resnext101+capsnet_best.pth', map_location=torch.device(DEVICE)))
        #set_parameter_requires_grad(model_ft, False)
        
        use_model_loss = True   
    
    else:
        raise Exception(f"Invalid model name '{model_name}'")

    return model_ft, input_size, use_model_loss

