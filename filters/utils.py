import numpy as np
from scipy.signal import butter, sosfilt, zpk2sos, sosfreqz

def rt2slope(rt60, fs):
    '''convert time in seconds of 60db decay to energy decay slope'''
    return -60/(rt60*fs)

def hertz2unit(hertz, fs):
    '''Convert frequency from cycles per second to normalized'''
    return np.divide(hertz, fs//2).tolist()

def hertz2rad(hertz, fs):
    '''Convert frequency from cycles per second to rad'''
    return np.divide(hertz, fs)*2*np.pi

def db2mag(ydb):
    return 10**(ydb/20)

def mag2db(ylin):
    return 20*np.log10(ylin)

def get_frequency_samples(num):
    '''
    get frequency samples (in radians) sampled at linearly spaced points along the unit circle
    Args    num (int): number of frequency samples
    Output  frequency samples in radians between [0, pi]
    '''
    angle = np.linspace(0, 1, num)
    abs = np.ones(num)
    return abs * np.exp(1j * angle * np.pi) 

def octave_filtering(input_signal, fs, f_bands):
    num_bands = len(f_bands)
    out_bands = np.zeros((len(input_signal), num_bands))

    for b_idx in range(num_bands):
        if f_bands[b_idx] == 0:
            f_cutoff = (1 / np.sqrt(1.5)) * f_bands[b_idx + 1]
            z, p, k = butter(5, f_cutoff / (fs / 2), output='zpk')
        elif f_bands[b_idx] == fs / 2:
            f_cutoff = np.sqrt(1.5) * f_bands[b_idx - 1]
            z, p, k = butter(5, f_cutoff / (fs / 2), btype='high', output='zpk')
        else:
            this_band = f_bands[b_idx] * np.array([1 / np.sqrt(1.5), np.sqrt(1.5)])
            z, p, k = butter(5, this_band / (fs // 2), btype='band', output='zpk')

        sos = zpk2sos(z, p, k)

        w, h = sosfreqz(sos, worN = len(input_signal) // 2 + 1)
        out_bands[:, b_idx] = np.fft.irfft(h)
        # somehow this does not work when the input signal is an impulse 
        # out_bands[:, b_idx] = sosfilt(sos, input_signal).squeeze()

    return out_bands

def biquad_to_tf(x, beta, alpha):
    x_pow = np.power(x, np.array([0, -1, -2]), dtype=x.dtype)
    
    band = 0    
    H = ((x_pow @ beta[band, :]) / (x_pow @ alpha[band, :]))
    for band in range(1, beta.shape[0]):
        H *= ((x_pow @ beta[band, :]) / (x_pow @ alpha[band, :]))
    
    return H.astype(x.dtype)