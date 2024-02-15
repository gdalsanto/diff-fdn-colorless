import torch 
import numpy as np
import soundfile as sf
from filters.eq import *
from filters.utils import * 
from decayfit import * 

class FilterDesigner:
    def __init__(self, net, ir_path, octave=1, method='DecayFitNet', filter='GEQ'): 
        '''
        Estimates attenuation and tone control filter for DiffFDN
            net (nn.Model):     DiffFDN model
            ir_path (string):   path to the reference impulse response 
            method (string):    one of "DecayFinNet" and "BDA" - method use to 
                                extract EDC parameters 
            octave (int):       number of band in one octave
        '''
        self.net = net
        self.ir_path = ir_path
        self.method = method
        self.octave = octave
        self.filter = filter
        self.sr = 48000
        self.n_slopes = 1       # number of slopes in the EDC 

    def get_edc(self):
        # read impulse response 
        rir, sr = sf.read(self.ir_path, dtype='float32')
        if sr != self.sr:
            ValueError('Input samplerate does not correspond to that of the system')
            # TODO implement resampling 
        
        # convert stereo/multichannel to mono by taking first channel only
        if len(rir.shape) > 1:
            rir = rir[:, 0]
        rir = rir / np.linalg.norm(rir)        
        
        if self.method == 'DecayFitNet':
            assert self.octave == 1, 'DecayFitNet supports only one octave-band filters'
            self.T, self.A, self.N, self.level = get_fdn_EDCparam(rir, self.f_bands, self.n_slopes, sr, self.net.device)
        elif self.method == 'BDA':
            self.T, self.A, self.N, self.level = get_BDA_param(rir, self.f_bands, self.n_slopes, sr)
        else:
            ValueError('EDC estimation method must be one between DecayFitNet and BDA')

    def get_center_freq(self):
        # TODO compute them for a given min ad max frequency
        if self.octave == 1:
            self.f_bands = [63, 125, 250, 500, 1000, 2000, 4000, 8000]    
        elif self.octave == 3:
            self.f_bands = [63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000]
        else:
            ValueError('Resolutions different form one or one-third octave bands are not supported at the moment')
        
    def get_filter(self):
        if self.filter == 'GEQ':
            assert self.octave == 1, "Graphic EQ is supported only for 1 octave filters"
            self.T = np.pad(self.T, ((1,1), (0, 0)), 'edge')    # used by the shelving filters 
            # get absorption filter
            G = absorptionGEQ(self.T, self.f_bands, self.net.m.numpy(), self.sr) # SOS 
            self.G_SOS = G / np.reshape(G[:,:,:,3], (len(self.net.m), 1, len(self.f_bands)+3, 1))   # a0 = 1
            # initial level filter, attenuate top and bottom bands
            target_level = mag2db(np.pad(self.level, ((1,1), (0, 0)), 'edge'))
            target_level = target_level.squeeze() - np.array([5, 0, 0, 0, 0, 0, 0, 0, 5, 30])
            self.TC_SOS = designGEQ(target_level, self.f_bands, self.sr)

    def run_designer(self):
        self.get_center_freq()
        self.get_edc()
        self.get_filter()       
