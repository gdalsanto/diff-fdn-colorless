# Differentiable FDN for Colorless Reverberation 
# custom loss functions 

import torch 
import torch.nn as nn 
import torch.nn.functional as F

class asy_p_loss(nn.Module):
    '''Means squared error between abs(x1) and x2'''
    def forward(self, y_pred, y_true):
        loss = 0.0
        # loss on channels' output
        for i in range(y_pred.size(dim=-1)):
            loss = loss + self.p_loss(y_pred[:,:,i], y_true)

        # loss on system's output
        y_pred_sum = torch.sum(y_pred, dim=-1)
        loss = loss + self.p_loss(y_pred_sum, y_true)*torch.sqrt(torch.tensor(y_pred.size(0)))

        return loss
    
    def p_loss(self, y_pred, y_true):
        gT = 2*torch.ones((y_pred.size(0),y_pred.size(1)))
        gT = gT + 2*torch.gt((torch.abs(y_pred) - torch.abs(y_true)),0).type(torch.uint8)
        loss = torch.mean(torch.pow(torch.abs(y_pred)-torch.abs(y_true),gT))   

        return loss   

class sparsity_loss(nn.Module):
    '''ratio between l2 and l1 norm for ir sparsity'''
    def forward(self, y_pred, y_true):
        batch_size = y_pred.size(dim=0)
        loss = 0.0
        for i in range(batch_size): # loop over batch elements
            loss = loss + torch.norm(y_pred[i,:], 2)/torch.norm(y_pred[i,:], 1)
        loss = loss / batch_size
        return loss

class neg_l1_norm(nn.Module):
    '''negative l1 norm '''
    def forward(self, y_pred, y_true):
        batch_size = y_pred.size(dim=0)
        loss = 0.0
        for i in range(batch_size): # loop over batch elements
            loss = loss - torch.norm(y_pred[i,:], 1)
        loss = loss / batch_size
        return loss 