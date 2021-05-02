# Include src directory as source
import sys  
sys.path.insert(0, '../src')

# Import configuration an utility functions
import config
import utils

# PyTorch Libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

# Other libs
import numpy as np
import random
import os

# Seeds
random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
os.environ["PYTHONHASHSEED"] = str(config.SEED)
np.random.seed(1)


# Input a 4D tensor (batch, caps, vecs, prob ) 
def non_linear_squashing(s, dim=-1):
    norm_sqr = s.pow(2).sum(dim=dim, keepdim=True)
         
    scale    = norm_sqr.pow(0.5)+(1e-10)
        
    mag      = (norm_sqr/(1+norm_sqr))
    s        = s * mag / scale
    return  s 
    

# Replace with Image Model
class CapsConvUnit(nn.Module):
    def __init__(self, in_channels=1, out_channels=256,kernel_size=9,stride=1,padding=0,nl_activation=True,bias=True):
        
        super(CapsConvUnit, self).__init__()

        self.nl_activation = nl_activation # Apply non linear activation?
        
        self.conv = nn.Conv2d(in_channels  = in_channels ,
                              out_channels = out_channels,
                              kernel_size  = kernel_size ,
                              stride       = stride      ,
                              padding      = padding     ,
                              bias         = True
                             )
        
        if self.nl_activation:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        x = self.conv(x) # Basic Conv Cell
        
        if self.nl_activation:
            x = self.relu(x) # non linearity
            
        return x

class CapsConvLayer(nn.Module):
    
    def __init__(self, 
                 in_channels       = 256 , 
                 out_channels      = 32  , 
                 kernel_size       = 9   ,
                 stride            = 1   ,
                 padding           = 0   ,
                 num_capsules      = 8   , 
                 input_size        = 32 * 6 * 6  ,
                 num_classes       = 10  ,
                 #input_channel     = 1   , # what if input channel is > 1                 
                 routing           = True,
                 is_convUnit       = False,
                 num_iterations    = 3,
                 conv_unit_model   = None,      
                 img_size          = 28           
                ):
        
        super(CapsConvLayer, self).__init__()
        #assert input_channel == 1 # fixme
                
        self.routing      = routing
        self.iterations   = num_iterations
        self.num_classes  = num_classes
        self.num_capsules = num_capsules
        self.is_convUnit  = is_convUnit
        self.input_size   = input_size
        self.img_size     = img_size
        self.in_channels  = in_channels
        
        
        if conv_unit_model:
            self.conv_pre_model = True
        else:
            self.conv_pre_model = False
        
        
        if self.routing:
            self.weights = nn.Parameter(torch.randn(1, self.input_size , self.num_classes, out_channels, self.num_capsules))
            
        else:
            if self.is_convUnit:
        
                if self.conv_pre_model:
                    self.net = conv_unit_model
                
                self.capsules = CapsConvUnit(
                            in_channels=in_channels, 
                            out_channels=out_channels, 
                            kernel_size=kernel_size, 
                            stride=stride, 
                            padding=0, 
                            nl_activation=True
                        )
                
            else:
                self.capsules = nn.ModuleList(
                [
                    CapsConvUnit(
                            in_channels=in_channels, 
                            out_channels=out_channels, 
                            kernel_size=kernel_size, 
                            stride=stride, 
                            padding=0, 
                            nl_activation=is_convUnit
                    )
                    for i in range(self.num_capsules)
                ])
                
            #else:
            #    self.capsules = nn.ModuleList([ primary_cap_class(in_channels=in_channels,out_channels=out_channels)
            #                                                    for _ in range(num_capsules)])
        
    def forward(self, x):
        # x <- (batch, num_capsules, input_size)
        
        batch_size = x.size(0)
        
        if self.is_convUnit:
            if self.conv_pre_model:
                x = self.net(x)
                x = x.view(batch_size, self.in_channels, self.img_size, self.img_size)
                
            return self.capsules(x)
        
        if self.routing:
            # u <- [batch, input_size, num_classes, num_capsules, 1]
            u_i  = torch.stack([x.transpose(1, 2)] * self.num_classes, dim=2).unsqueeze(4)
            
            # w <- [batch, input_size, num_classes, out_channels, num_capsules]            
            w_ij = torch.cat([self.weights] * batch_size)
            
            # b <= [1 , input_size, num_classes, num_capsules]
            b_ij = Variable(torch.zeros(1, self.input_size, self.num_classes, 1)).to(config.DEVICE)
            
            # u_j|i = w_ij * u_i
            u_j_i = torch.matmul(w_ij, u_i)

            for iteration in range(self.iterations):
                                                                  
                c_ij = F.softmax(b_ij, dim=1)
                c_ij = torch.cat([c_ij] * batch_size).unsqueeze(4)
        
                s_j  = (c_ij * u_j_i).sum(dim=1, keepdim=True)
                
                out = non_linear_squashing(s_j, dim=2)
                
                v_j  = torch.cat([out] * self.input_size, dim=1)
                
                u_v_j= torch.matmul(u_j_i.transpose(3, 4), v_j).squeeze(4).mean(dim=0, keepdim=True)
                
                b_ij = b_ij + u_v_j
            
            out = out.squeeze(1)
                            
        else:
            
            # # u <- [batch, num_capsules, (input_size)]
            u_i = torch.stack([self.capsules[i](x) for i in range(self.num_capsules)], dim=1)
            
            # # u <- [batch, num_capsules, input_size]
            u_i = u_i.view(batch_size, self.num_capsules, -1)
            
            v_j = non_linear_squashing(u_i)            
            
            out = v_j
    
        return out
        
        
class CapsNet(nn.Module):
    def __init__(self,
                 img_size          = 28  , # 28
                 img_channels      = 1   , # 1
                 conv_out_channels = 256 , # 256
                 prim_out_channels = 32  , # 256
                 num_capsules      = 8   , # 8
                 num_classes       = 10  , # 10
                 out_channels      = 16  , # 16
                 conv_kernel_size  = 9   ,
                 conv_unit_model   = None,
                 image_factor      = 1
                 ):
        
        super(CapsNet, self).__init__()
        
        self.image_size        = img_size
        self.image_channels    = img_channels
        self.num_classes       = num_classes
        self.conv_out_channels = conv_out_channels
        self.conv_kernel_size  = conv_kernel_size
        self.image_factor      = image_factor
        self.transform         = transforms.Compose([transforms.Resize(int(img_size))])
                
        self.conv = CapsConvLayer (
                 in_channels       = img_channels      , # *3
                 out_channels      = conv_out_channels , # *3 
                 kernel_size       = conv_kernel_size  ,
                 stride            = 1                 ,
                 padding           = 0                 ,
                 routing           = False             ,
                 is_convUnit       = True              ,
                 conv_unit_model   = conv_unit_model   ,
                 img_size          = img_size
                )
                
                
        conv_output_volume = utils.conv_output_volume(img_size,conv_kernel_size,1,0)
        #print(f'conv_output_volume = {conv_output_volume}')
        
        if conv_output_volume > 50:
            self.pool = nn.MaxPool2d(conv_kernel_size, stride=3)
            conv_output_volume = utils.conv_output_volume(conv_output_volume,conv_kernel_size,3,0)
        else:
            self.pool = nn.MaxPool2d(1, stride=1)        
            
        #print(f'conv_output_volume = {conv_output_volume}')
        
        self.primary = CapsConvLayer (
                 in_channels       = conv_out_channels , # *3
                 out_channels      = prim_out_channels , # *3
                 kernel_size       = conv_kernel_size  ,
                 stride            = 2   ,
                 padding           = 0   ,
                 routing           = False,
                 is_convUnit       = False,
                 num_capsules      = num_capsules
                )
        
        prim_output_volume = utils.conv_output_volume(conv_output_volume,conv_kernel_size,2,0)
        
        self.digit = CapsConvLayer (
                 out_channels      = out_channels  , # *3
                 num_classes       = num_classes   , # *3 ### Binary = 2
                 input_size        = prim_out_channels * prim_output_volume * prim_output_volume, # *3
                 routing           = True,
                 is_convUnit       = False,
                 num_capsules      = num_capsules
                )
        
        num_out_channels = self.image_channels * self.image_size * self.image_size
        
        self.decoder = nn.Sequential(
            nn.Linear(out_channels * num_classes, int(2 * num_out_channels / 3)),
            nn.ReLU(inplace=True),
            nn.Linear(int(2 * num_out_channels / 3), int(3 * num_out_channels / 2)),
            nn.ReLU(inplace=True),
            nn.Linear(int(3 * num_out_channels / 2), (img_size) * (img_size) * img_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        
        out = x
        out = self.conv(out)
        out = self.pool(out)
        out = self.primary(out)
        out = self.digit(out)
        
        return out 
        
    def loss(self, x, y_hat, y, mean_error=True, reconstruct=True):
        
        margin_err      = self.margin_loss(x,y,y_hat,mean_error)
        reconstruct_err = 0
        reconstructions = None
        
        if reconstruct:
            reconstruct_err, reconstructions = self.reconstruct_loss(x,y_hat,mean_error)
            return (margin_err + 0.0005 + (reconstruct_err/self.image_factor), reconstructions)
            
        return margin_err + 0.0005
        
    def margin_loss(self, x, y, y_hat, mean_error=True):    
        
        batch = x.size(0)
        
        y_hat = torch.sqrt((y_hat ** 2).sum(dim=2, keepdim=True)).to(config.DEVICE)
        zero = Variable(torch.zeros(1)).to(config.DEVICE)
        
        # Margin Loss
        left  = F.relu(0.9 - y_hat, zero).view(batch, -1)**2
        right = F.relu(y_hat - 0.1, zero).view(batch, -1)**2
        
        y_bar = y.view(batch,self.num_classes).to(config.DEVICE)
        
        margin_error = (y_bar * left + 0.5  * (1. - y_bar) * right).sum(dim=1)
        
        if mean_error:
            margin_error = margin_error.mean()
        
        return margin_error

    def reconstruct_loss(self, x, y_hat, mean_error=True):    
    
        batch = x.size(0)
        
        out = ((y_hat ** 2).sum(dim=2) ** 0.5)
        
        _, index = out.max(dim=1)
        index = index.data
        
        masks = [None] * batch
        imges = [None] * batch
        
        for idx in range(batch):
            data = y_hat[idx]
            
            mask = Variable(torch.zeros(data.size())).to(config.DEVICE)
            mask[index[idx]] = data[index[idx]]
            masks[idx] = mask
            
            imges[idx] = self.transform(x[idx])
            
        y_hat = torch.stack(masks, dim=0)
        x = torch.stack(imges, dim=0)
        
        #reconstructions = self.decoder(y_hat.view(batch, -1)).view(batch, self.image_channels, int(self.image_size*self.image_factor), int(self.image_size*self.image_factor))
        reconstructions = self.decoder(y_hat.view(batch, -1)).view(batch, self.image_channels, int(self.image_size), int(self.image_size))
        
        # Reconstruction Loss
        reconstruct_error = torch.sum(((reconstructions - x).view(batch, -1) ** 2), dim=1) * 0.0005
        
        if mean_error:
            reconstruct_error = reconstruct_error.mean()    
        
        return reconstruct_error, reconstructions
        
    def get_preduction(self, out):
        length = ((out ** 2).sum(dim=2) ** 0.5)
        
        y_hat = (length.data.max(2)[0])
        
        return y_hat
        
    def get_probabilities(self, out):
        return ((out ** 2).sum(dim=2, keepdim=True) ** 0.5).squeeze().squeeze()

class CapsNetworks(nn.Module):
    def __init__(self,
                 preNet,
                 capsNet
                 ):
        
        super(CapsNetworks, self).__init__()
        
        self.preNet     = preNet
        self.capsNet    = capsNet
        self.pool       = nn.AvgPool2d(kernel_size=3, stride=1, padding=0)
       
    def forward(self, x):
        
        hidden = None
        out = x
        if self.preNet:
            out = self.preNet(out)
            out = out.view(x.size(0), 1, 32, 32)
            out = self.pool(out)
            hidden = out
        
        out = self.capsNet(out)
        
        out = self.get_probabilities(out)
        out = torch.sigmoid(out)
        
        return (out,hidden)
        
    def loss(self, x, y_hat, y, mean_error=True, reconstruct=False):
        #if reconstruct:
        #    return (self.capsNet.loss(x,y_hat, y, mean_error=mean_error, reconstruct=False), None)
        return self.capsNet.loss(x,y_hat, y, mean_error=mean_error, reconstruct=False)

    def get_preduction(self, out):
        return self.capsNet.get_preduction(out)

    def get_probabilities(self, out):
        return ((out ** 2).sum(dim=2, keepdim=True) ** 0.5).squeeze(-1).squeeze(-1)
