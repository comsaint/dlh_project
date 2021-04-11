# Include src directory as source
import sys  
sys.path.insert(0, '../src')

# Import configuration an utility functions
import config
import util

# PyTorch Libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
                 primary_cap_class = CapsConvUnit,                 
                ):
        
        super(CapsConvLayer, self).__init__()
        #assert input_channel == 1 # fixme
                
        self.routing      = routing
        self.iterations   = num_iterations
        self.num_classes  = num_classes
        self.num_capsules = num_capsules
        self.is_convUnit  = is_convUnit
        self.input_size   = input_size
        
        if self.routing:
            self.weights = nn.Parameter(torch.randn(1, self.input_size , self.num_classes, out_channels, self.num_capsules))
            
        else:
            if self.is_convUnit:
        
                if primary_cap_class.__name__ == 'CapsConvUnit':
                    self.capsules = CapsConvUnit(
                            in_channels=in_channels, 
                            out_channels=out_channels, 
                            kernel_size=kernel_size, 
                            stride=stride, 
                            padding=0, 
                            nl_activation=True
                        )
                else:
                    self.capsules = primary_cap_class(in_channels=in_channels,out_channels=out_channels)
                
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
                 primary_cap_class = CapsConvUnit
                 ):
        
        super(CapsNet, self).__init__()
        
        self.image_size     = img_size
        self.image_channels = img_channels
        self.num_classes    = num_classes
                
        self.conv = CapsConvLayer (
                 in_channels       = img_channels      , # *3
                 out_channels      = conv_out_channels , # *3 
                 kernel_size       = conv_kernel_size  ,
                 stride            = 1                 ,
                 padding           = 0                 ,
                 routing           = False             ,
                 is_convUnit       = True              ,
                 primary_cap_class = primary_cap_class
                )
                
                
        conv_output_volume = util.conv_output_volume(img_size,conv_kernel_size,1,0)
        
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
        
        prim_output_volume = util.conv_output_volume(conv_output_volume,conv_kernel_size,2,0)
        
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
            nn.Linear(int(3 * num_out_channels / 2), img_size * img_size * img_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        
        out = x
        out = self.conv(out)
        out = self.primary(out)
        out = self.digit(out)
        
        return out 
        
    def loss(self, x, y_hat, y, mean_error=True, reconstruct=True):
        
        margin_err      = self.margin_loss(x,y,y_hat,mean_error)
        reconstruct_err = 0
        
        if reconstruct:
            reconstruct_err = self.reconstruct_loss(x,y_hat,mean_error)
            
        return margin_err + reconstruct_err
        
    def margin_loss(self, x, y, y_hat, mean_error=True):    
        
        batch = x.size(0)
        
        y_hat = ((y_hat ** 2).sum(dim=2, keepdim=True) ** 0.5).to(config.DEVICE)
        
        # Margin Loss
        left  = torch.max(0.9 - y_hat.unsqueeze(-1), Variable(torch.zeros(1)).to(config.DEVICE)).view(batch, -1)**2
        right = torch.max(y_hat.unsqueeze(-1) - 0.1, Variable(torch.zeros(1)).to(config.DEVICE)).view(batch, -1)**2
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
        
        for idx in range(batch):
            data = y_hat[idx]

            mask = Variable(torch.zeros(data.size())).to(config.DEVICE)
            mask[index[idx]] = data[index[idx]]
            masks[idx] = mask
        y_hat = torch.stack(masks, dim=0)
        
        reconstructions = self.decoder(y_hat.view(batch, -1)).view(batch, self.image_channels, self.image_size, self.image_size)
        
        # Reconstruction Loss
        reconstruct_error = torch.sum(((reconstructions - x).view(batch, -1) ** 2), dim=1) * 0.0005
        if mean_error:
            reconstruct_error = reconstruct_error.mean()    
        
        return reconstruct_error
            
    def get_preduction(self, out):
        length = ((out ** 2).sum(dim=2, keepdim=True) ** 0.5)
        
        y_hat = length.data.max(1)[1].cpu()
        
        return y_hat
