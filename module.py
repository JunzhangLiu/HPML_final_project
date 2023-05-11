import numpy as np
import torch
import torch.nn as nn
from typing import Tuple,List
class Positional_embedding(nn.Module):
    def __init__(self,vertical_feat,horizontal_feat,mean=0,std=0.02):
        super(Positional_embedding,self).__init__()
        self.embedding=torch.nn.Parameter(torch.normal(mean,std,(1,vertical_feat,horizontal_feat)))
        self.embedding.requires_grad=True
    def forward(self,inputs:torch.Tensor)->torch.Tensor:
        return inputs+self.embedding
class Gaussian_noise(nn.Module):
    def __init__(self):
        super(Gaussian_noise,self).__init__()
    def forward(self,inputs,mean=0,std=1):
        noise = torch.normal(mean,std,size=inputs.shape,device=inputs.device)
        return inputs+noise
    
class Residual_block(nn.Module):
    def __init__(self,config):
        super(Residual_block,self).__init__()
        embed_dim = config['hidden_channels']
        activation1 = config['activation1']
        activation2 = config['activation2']
        normalization = config['normalization']
        self.module = nn.Sequential(nn.Linear(embed_dim,embed_dim),
                                    activation1(inplace=True),
                                    nn.Linear(embed_dim,embed_dim))
        self.normalize = normalization(embed_dim)
        self.activation = activation2(inplace=True)
    def forward(self,inputs:torch.Tensor)->torch.Tensor:
        x = self.module(inputs)
        return self.activation(self.normalize(x+inputs))
    
class Attention_block(nn.Module):
    def __init__(self,config):
        super(Attention_block,self).__init__()       
        embed_dim = config['hidden_channels']
        activation1 = config['activation1']
        activation2 = config['activation2']
        normalization = config['normalization']
        num_heads = config['num_heads']
        self.atten_head = nn.MultiheadAttention(embed_dim,num_heads=num_heads,batch_first=True)
        self.norm = normalization(embed_dim)
        self.act = activation2(inplace=True)
        self.fc = Time_distributed(nn.Linear(embed_dim,embed_dim),
                                  activation1(inplace=True),
                                  nn.Linear(embed_dim,embed_dim))
    def forward(self,inputs)->torch.Tensor:
        x = inputs
        atten_out = self.norm(self.atten_head(x,x,x)[0]+x)
        x = self.fc(atten_out)
        x = self.norm(x+atten_out)
        x = self.act(x)
        return x

class Loss_func():
    def __init__(self,loss_metrics,weights=None):
        super(Loss_func,self).__init__()
        self.loss_callbacks = [f() for f in loss_metrics]
        if weights is None:
            self.weights = [1/len(loss_metrics) for _ in range(len(loss_metrics))]
        else:
            self.weights = weights
    def __call__(self,y_pred,y_true):
        loss = 0
        for l,weight in zip(self.loss_callbacks,self.weights):
            loss = loss+l(y_pred,y_true)/weight
        return loss

class Time_distributed(nn.Module):
    def __init__(self, *module):
        super(Time_distributed, self).__init__()
        self.module = nn.Sequential(*module)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        if len(x.size()) <= 2:
            return self.module(x)

        x_reshape = x.contiguous().view((-1,x.shape[-1]))

        y = self.module(x_reshape)

        y = y.contiguous().view(x.size(0), -1, y.size(-1))
        return y


