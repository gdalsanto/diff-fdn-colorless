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
    def __init__(self, N, gain_per_sample, delays, device):
        super().__init__()
        self.device = device
        # input parameters
        self.N = N  # size of the FDN 
        self.gain_per_sample = gain_per_sample
        # learnable parameters
        self.B = nn.Parameter(torch.randn(1,N,1)/N)
        self.C = nn.Parameter(torch.randn(1,1,N)/N)

        if delays == 'init':
            delaysVal = torch.tensor([809., 877., 937., 1049., 1151., 1249., 1373., 1499.])
            self.mStd = torch.std(delaysVal)
            self.mAvr = torch.mean(delaysVal)
            delaysVal = (delaysVal - self.mAvr)/self.mStd 
            self.m = nn.Parameter(delaysVal)
        elif isinstance(delays, (list)):
            self.m = torch.tensor(delays).squeeze()
            self.mStd = torch.std(self.m)
            self.mAvr = torch.mean(self.m)
            self.m = (self.m - self.mAvr)/self.mStd 
        else:
            raise ValueError('Value for third argument is not valid')

        # feedback matrix
        self.A = nn.Linear(N,N)
        # save parametetrization for orthogonality 
        parametrize.register_parametrization(self.A, "weight", Skew())
        parametrize.register_parametrization(self.A, "weight", MatrixExponential())
        X = self.A.weight
        print(torch.dist(X.T @ X, torch.eye(N)))
        
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

