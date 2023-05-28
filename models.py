# Differentiable FDN for Colorless Reverberation 
# modules

import torch 
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize 

class Skew(nn.Module):
    def forward(self, X):
        A = X.triu(1)
        return A - A.transpose(-1, -2)

class MatrixExponential(nn.Module):
    def forward(self, X):
        return torch.matrix_exp(X)

class DiffFDN(nn.Module):
    # recursive neural netwrok 
    def __init__(self, delays, gain_per_sample, device):
        super().__init__()
        self.device = device
        # input parameters
        self.gain_per_sample = gain_per_sample

        if isinstance(delays, (list)):
            self.m = torch.tensor(delays).squeeze()
            self.mStd = torch.std(self.m)
            self.mAvr = torch.mean(self.m)
            self.m = (self.m - self.mAvr)/self.mStd 
            self.N = len(delays) # size of the FDN 
        else:
            raise ValueError('Value for delay line lengths is not valid')

        # learnable parameters
        self.B = nn.Parameter(torch.randn(1,self.N,1)/self.N)
        self.C = nn.Parameter(torch.randn(1,1,self.N)/self.N)
        # feedback matrix
        self.A = nn.Linear(self.N,self.N)
        # save parametetrization for orthogonality 
        parametrize.register_parametrization(self.A, "weight", Skew())
        parametrize.register_parametrization(self.A, "weight", MatrixExponential())
        X = self.A.weight
        print(torch.dist(X.T @ X, torch.eye(self.N)))
        
    def forward(self, x):
        # output system's impulse response h and channels' frequency response
        B = torch.complex(self.B, torch.zeros(1,self.N,1))
        C = torch.complex(self.C, torch.zeros(1,1,self.N))
        A = self.A.weight.unsqueeze(0)
        m = self.m * self.mStd + self.mAvr 
        D = torch.diag_embed(torch.unsqueeze(x, dim=-1) ** m)
        Gamma = torch.diag(self.gain_per_sample**m)
        Hchannel = torch.matmul(torch.inverse(D - torch.matmul(A,Gamma)), B)
        H = Hchannel.squeeze()*C.squeeze()  
        H_mask = torch.ones((torch.sum(H, dim=-1)).size())
        H_mask[(x == 0).squeeze()] = 0 # pad on the right side 
        h = torch.fft.irfft(torch.mul(torch.sum(H, dim=-1),H_mask), n=48000*8)
        h = h/torch.max(torch.abs(h))
        param = (A.detach(), B.detach(), C.detach(), Gamma.detach(), m.detach())
        return H, h, param

    def print(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data)

