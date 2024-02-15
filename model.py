# Differentiable FDN for Colorless Reverberation 
# modules

import torch 
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize 
import numpy as np 
from scattering.utils import *
from utility import to_complex
from filters.utils import biquad_to_tf

class Skew(nn.Module):
    def forward(self, X):
        A = X.triu(1)
        return A - A.transpose(-1, -2)
    
class MatrixExponential(nn.Module):
    def forward(self, X):
        return torch.matrix_exp(X)

class DiffFDN(nn.Module):
    # recursive neural netwrok 
    def __init__(self, delays, gain_per_sample, device, scattering = False):
        super().__init__()
        self.device = device
        self.scattering = scattering
        # input parameters
        self.gain_per_sample = gain_per_sample
        self.m = torch.tensor(delays).squeeze()
        self.N = len(delays) # size of the FDN 

        # learnable parameters
        self.B = nn.Parameter(torch.randn(1,self.N,1)/self.N)
        self.C = nn.Parameter(torch.randn(1,1,self.N)/self.N)

        # feedback matrix
        if self.scattering: 
            self.K = 4  # number of stages
            self.sparsity = 3
            self.A = nn.Parameter(2*torch.rand(self.K, self.N, self.N)/np.sqrt(self.N) - 1/np.sqrt(self.N))
            self.m_L = torch.randint(low=1, high=550,size=[self.N])
            self.m_R = torch.randint(low=1, high=550,size=[self.N])
        else:
            self.A = nn.Parameter(2*torch.rand(self.N, self.N)/np.sqrt(self.N) - 1/np.sqrt(self.N))

        self.ortho_param = nn.Sequential(Skew(),
                                        MatrixExponential())
        self.rir_synthesis = False
        self.TC_SOS = None  # it will be set by another function 
        self.G_SOS = None
        
    def forward(self, x):

        B = to_complex(self.B)
        C = to_complex(self.C)

        m = self.m  
        if self.scattering:
            A = self.ortho_param(self.A).permute(1, 2, 0)
            V = cascaded_paraunit_matrix(
                self.N, 
                self.K-1, 
                gain_per_sample = 1, 
                sparsity=self.sparsity, 
                matrix=A)
            self.V = V.detach().clone() # for logging
            # put part of main delay to left and right delays (to break the syncrony of paths)
            V = shift_matrix(V, self.m_L, direction='left')
            V = shift_matrix(V, self.m_R, direction='right')
            V = to_complex(V)
            V = torch.einsum('jim, mn -> jimn', V,  (x.view(-1,1)**-torch.arange(0, V.shape[-1])).permute(1,0))
            V = torch.sum(V, dim=2).permute(2, 0, 1)           
        else:
            V = to_complex(self.ortho_param(self.A))

        D = torch.diag_embed(torch.unsqueeze(x, dim=-1) ** m)
        Gamma = to_complex(torch.diag(self.gain_per_sample**m))

        
        if self.rir_synthesis:
            self.set_tone_control(x, self.TC_SOS)
            self.set_attenuation_filter(x, self.G_SOS)
            C = torch.matmul(self.TC.reshape(x.shape[0], 1), C).reshape(x.shape[0], 1, self.N)
            H = torch.matmul(C, torch.matmul(torch.inverse(torch.squeeze(D) - torch.matmul(self.G, V)), B[0,:,:]))
            H = torch.squeeze(H, -1)
        else:
            Hchannel = torch.matmul(torch.inverse(D - torch.matmul(V,Gamma)), B)
            H = Hchannel.squeeze()*C.squeeze()  
        
        return H

    def print(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data)
    
    def get_parameters(self):
        B = torch.complex(self.B, torch.zeros(1,self.N,1))
        C = torch.complex(self.C, torch.zeros(1,1,self.N))
        A = to_complex(self.ortho_param(self.A))        
        m = self.m  
        Gamma = torch.diag(self.gain_per_sample**m)

        return (A, B, C, Gamma, m)
    
    @torch.no_grad()
    def get_param_dict(self):
        param_np = {}
        param_np['feedbackMatrix'] = self.V.squeeze().cpu().numpy() 
        param_np['delayLeft'] = self.m_L.squeeze().cpu().numpy() 
        param_np['delayRight'] = self.m_R.squeeze().cpu().numpy() 
        param_np['delays'] = self.m.squeeze().cpu().numpy()  
        param_np['inputGain'] = self.B.squeeze().cpu().numpy() 
        param_np['outputGain'] = self.C.squeeze().cpu().numpy() 
        return param_np

    def set_attenuation_filter(self, z, G_SOS):
        num = len(z)
        Gch = (1j*np.zeros((num, self.N))).astype('complex64')
        for ch in range(self.N):
            Gch[:, ch] = biquad_to_tf(np.reshape(z.numpy(), (num, 1)), G_SOS[ch,:,:,0:3].squeeze(), G_SOS[ch,:,:,3:6].squeeze())

        # this is so ugly, but there no python version of torch.diag_embed
        self.G = torch.diag_embed(torch.tensor(Gch))

    def set_tone_control(self, z, TC_SOS):
        num = len(z)
        self.TC = biquad_to_tf(np.reshape(z.numpy(), (num, 1)), TC_SOS[:, 0:3], TC_SOS[:, 3:6])
        self.TC = torch.tensor(self.TC)