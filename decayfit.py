import torch 
import numpy as np
from filters.utils import *
from DecayFitNet.python.toolbox.DecayFitNetToolbox import DecayFitNetToolbox
from DecayFitNet.python.toolbox.utils import calc_mse
from DecayFitNet.python.toolbox.core import discard_last_n_percent, decay_model, PreprocessRIR
from DecayFitNet.python.toolbox.BayesianDecayAnalysis import BayesianDecayAnalysis


def getEDCparam(rir, filter_frequencies, n_slopes = 1, sr=48000, device='cpu'):
    '''
    Using DecayFitNet, extract EDC parameters A, T, C 
    The RIR onset time is detected 
    args:
        rir (array):    input room impulse response
        n_slopes (int): number of EDC slopes. 0 = number of active slopes is 
                        determined by network (between 1 and 3)
        sr (int):       sampling rate
        filter_frequencies (list):  frequency bands
        
    '''
    n_slopes = n_slopes 

    # Init Preprocessing
    rir_preprocessing = PreprocessRIR(sample_rate=sr, filter_frequencies=filter_frequencies)

    # Schroeder integration, analyse_full_rir: if RIR onset should be detected, set this to False
    true_edc, __ = rir_preprocessing.schroeder(rir, analyse_full_rir=False)
    time_axis = (torch.linspace(0, true_edc.shape[2] - 1, true_edc.shape[2]) / sr)

    # Permute into [n_bands, n_batches, n_samples]
    true_edc = true_edc.permute(1, 0, 2)

    # Prepare the model
    decayfitnet = DecayFitNetToolbox(n_slopes=n_slopes, sample_rate=sr, filter_frequencies=filter_frequencies)
    estimated_parameters_decayfitnet, norm_vals_decayfitnet = decayfitnet.estimate_parameters(rir, analyse_full_rir=False)
    # Get fitted EDC from estimated parameters
    print('Device = ' + device)
    fitted_edc_decayfitnet = decay_model(torch.from_numpy(estimated_parameters_decayfitnet[0]).to(device),
                                        torch.from_numpy(estimated_parameters_decayfitnet[1]).to(device),
                                        torch.from_numpy(estimated_parameters_decayfitnet[2]).to(device),
                                        time_axis=time_axis,
                                        compensate_uli=True,
                                        backend='torch',
                                        device=device)
    # Discard last 5% for MSE evaluation
    true_edc = discard_last_n_percent(true_edc, 5)
    fitted_edc_decayfitnet = discard_last_n_percent(fitted_edc_decayfitnet, 5)
    # Calculate MSE between true EDC and fitted EDC
    mse_per_frequencyband = calc_mse(true_edc, fitted_edc_decayfitnet)
    return estimated_parameters_decayfitnet, norm_vals_decayfitnet.numpy(), mse_per_frequencyband


def decayFitNet2InitialLevel(T, A, N, normalization, fs, rirLen, fBands):
    '''
    convert decayFitNet estimation to initial level as used in FDNs
    Sebastian J. Schlecht, Saturday, 28 January 2023 (adapted to python)
    '''
    # Denormalize the amplitudes of the EDC
    if normalization.shape[0] != A.shape[0]:
        normalization = normalization.transpose(1, 0)
    A = A * normalization
    N = N * normalization
    # Estimate the energy of the octave filters
    inpulse = np.zeros((fs, 1))
    inpulse[0] = 1
    rirFBands = octave_filtering(inpulse, fs, fBands)
    bandEnergy = np.expand_dims(sum(rirFBands**2), -1)  
    # Cumulative energy is a geometric series of the gain per sample
    gainPerSample = db2mag(rt2slope(T, fs))
    decayEnergy = 1 / (1 - gainPerSample**2)
    # initial level
    level = np.sqrt( A / bandEnergy / decayEnergy * rirLen)
    # there is an offset because, the FDN is not energy normalized
    # The rirLen factor is due to the normalization in schroederInt (in DecayFitNet)
    return level, A, N

def get_fdn_EDCparam(rir, f_bands, n_slopes, sr, device):
    ''' Use DecayFitNet to extract the EDC parameters to be used in FDN design 
    args:
        rir (array):    input room impulse response
        f_bands (list): band center frequencies
        n_slopes (int): number of EDC slopes. 0 = number of active slopes is 
                        determined by network (between 1 and 3)
        sr (int):       sampling rate
    '''

    # extract EDC parameters
    edc_param, norm_vals, _ = getEDCparam(rir, f_bands, n_slopes = n_slopes, sr=sr, device=device)
    T, A, N = edc_param[0], edc_param[1], edc_param[2]
    # convert decayFitNet estimation to initial level as used in FDNs
    level, A, N = decayFitNet2InitialLevel(T, A, N, norm_vals, sr, len(rir), f_bands)
    return T, A, N, level

def get_BDA_param(rir, f_bands, n_slopes, sr):
    # Bayesian paramters: a_range and n_range are both exponents, i.e., actual range = 10^a_range or 10^n_range
    parameter_ranges = {'t_range': [0.1, 3.5],
                        'a_range': [-3, 0],
                        'n_range': [-10, -2]}
    n_iterations = 100
    # Init Bayesian decay analysis
    bda = BayesianDecayAnalysis(n_slopes, sr, parameter_ranges, n_iterations, filter_frequencies=f_bands)

    # Estimate decay parameters
    edc_param, norm_vals = bda.estimate_parameters(rir)
    T, A, N = edc_param[0], edc_param[1], edc_param[2]
    # convert Bayesian Analysis estimation to initial level as used in FDNs
    level, A, N = decayFitNet2InitialLevel(T, A, N, norm_vals, sr, len(rir), f_bands)
    return T, A, N, level
