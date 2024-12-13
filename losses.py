# Differentiable FDN for Colorless Reverberation 
# custom loss functions 

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

class mse_loss(nn.Module):
    '''Means squared error between abs(x1) and x2'''
    def forward(self, y_pred, y_true):
        loss = 0.0
        N = y_pred.size(dim=-1)
        # loss on channels' output
        for i in range(N):
            loss = loss + torch.mean(torch.pow(torch.abs(y_pred[:,i])-torch.abs(y_true), 2*torch.ones(y_pred.size(0))))  

        # loss on system's output
        y_pred_sum = torch.sum(y_pred, dim=-1)
        loss = loss/N + torch.mean(torch.pow(torch.abs(y_pred_sum)-torch.abs(y_true), 2*torch.ones(y_pred.size(0)))) 

        return loss

class sparsity_loss(nn.Module):
    ''''''
    def forward(self, A):
        N = A.shape[-1]
        return -(torch.sum(torch.abs(A)) - N*np.sqrt(N))/(N*(np.sqrt(N)-1))