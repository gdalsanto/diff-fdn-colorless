import numpy as np
from filters.utils import *
from scipy.signal import sosfreqz
from scipy.optimize import minimize

def absorptionGEQ(RT, center_freq, delays, fs):
    # absorptionGEQ - Design GEQ absorption filters according to specified T60
    #
    # Schlecht, S., Habets, E. (2017). Accurate reverberation time control in
    # feedback delay networks Proc. Int. Conf. Digital Audio Effects (DAFx)
    # adapted to python 

    # Define the number of delays
    numberOfDelays = len(delays)

    # Convert T60 to magnitude response
    targetG = rt2slope(RT, fs)  

    # Design delay proportional filters
    numberOfBands = len(targetG) + 1 
    sos = np.zeros((numberOfDelays, 1, numberOfBands, 6))

    for it in range(numberOfDelays):
        optimalSOS = designGEQ(targetG * delays[it], center_freq, fs)
        sos[it, 0, :, :] = optimalSOS

    return sos

def designGEQ(targetG, center_freq, fs=48000):
    # Initialization
    fftLen = 2**16

    # center_freq = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000])  # Hz
    shelving_crossover = np.array([46, 11360])  # TODO these has to be made functions of the center freqs
    num_freq = len(center_freq) + len(shelving_crossover)
    shelving_omega = hertz2rad(shelving_crossover, fs)
    center_omega = hertz2rad(center_freq, fs)
    R = 2.7

    # Control frequencies are spaced logarithmically
    num_control = 100
    control_freq = np.round(np.logspace(np.log10(1), np.log10(fs/2.1), num_control+1))

    targetF = np.concatenate(([1], center_freq, [fs/2.1]))
    # targetF = center_freq
    targetInterp = np.interp(control_freq, targetF, targetG.squeeze())

    # Design prototype of the biquad sections
    prototype_gain = 10  # dB
    prototype_gain_array = np.full(num_freq + 1, prototype_gain)
    prototypeSOS = graphicEQ(center_omega, shelving_omega, R, prototype_gain_array)
    G, _, _ = probeSOS(prototypeSOS, control_freq, fftLen, fs)
    G = G / prototype_gain  # dB vs control frequencies

    # Define the optimization bounds
    upperBound = [np.inf] + [2 * prototype_gain] * num_freq
    lowerBound = [-val for val in upperBound]

    # Optimization
    result = minimize(
        lambda x: np.linalg.norm(G.dot(x) - targetInterp),
        np.zeros(num_freq + 1),
        bounds=list(zip(lowerBound, upperBound))
    )
    optG = result.x

    # Generate the SOS coefficients
    sos = graphicEQ(center_omega, shelving_omega, R, optG)

    return sos

def graphicEQ(center_omega, shelving_omega, R, gaindB):
    num_freq = len(center_omega) + len(shelving_omega) + 1
    assert len(gaindB) == num_freq
    SOS = np.zeros((num_freq, 6))

    for band in range(num_freq):
        if band == 0:
            b = np.array([db2mag(gaindB[band]), 0, 0])
            a = np.array([1, 0, 0])
        elif band == 1:
            b, a = shelving_filter(shelving_omega[0], db2mag(gaindB[band]), 'low')
        elif band == num_freq - 1:
            b, a = shelving_filter(shelving_omega[1], db2mag(gaindB[band]), 'high')
        else:
            Q = np.sqrt(R) / (R - 1)
            b, a = bandpass_filter(center_omega[band-2], db2mag(gaindB[band]), Q )
            
        sos = np.hstack((b, a))
        SOS[band, :] = sos

    return SOS

def shelving_filter(omegaC, gain, type):
    b = np.ones(3)
    a = np.ones(3)

    t = np.tan(omegaC / 2)
    t2 = t ** 2
    g2 = gain ** 0.5
    g4 = gain ** 0.25

    b[0] = g2 * t2 + np.sqrt(2) * t * g4 + 1
    b[1] = 2 * g2 * t2 - 2
    b[2] = g2 * t2 - np.sqrt(2) * t * g4 + 1

    a[0] = g2 + np.sqrt(2) * t * g4 + t2
    a[1] = 2 * t2 - 2 * g2
    a[2] = g2 - np.sqrt(2) * t * g4 + t2

    b = g2 * b

    if type == 'high':
        tmp = b.copy()
        b = a * gain
        a = tmp

    return b, a

def bandpass_filter(omegaC, gain, Q):
    b = np.ones(3)
    a = np.ones(3)

    bandWidth = omegaC / Q
    t = np.tan(bandWidth / 2)

    b[0] = np.sqrt(gain) + gain * t
    b[1] = -2 * np.sqrt(gain) * np.cos(omegaC)
    b[2] = np.sqrt(gain) - gain * t

    a[0] = np.sqrt(gain) + t
    a[1] = -2 * np.sqrt(gain) * np.cos(omegaC)
    a[2] = np.sqrt(gain) - t

    return b, a


def probeSOS(SOS, controlFrequencies, fftLen, fs):
    ''' Probe the frequency / magnitude response of a cascaded SOS filter at the points
    specified by the control frequencies '''
    numFreq = SOS.shape[0]
    
    H = np.zeros((fftLen, numFreq), dtype=complex)
    W = np.zeros((fftLen, numFreq))
    G = np.zeros((len(controlFrequencies), numFreq))

    for band in range(numFreq):
        SOS[band, :] = SOS[band, :] / SOS[band, 3]

        w, h = sosfreqz(SOS[band, :], worN=fftLen, fs=fs)
        g = np.interp(controlFrequencies, w, 20 * np.log10(np.abs(h)))

        G[:, band] = g
        H[:, band] = h
        W[:, band] = w

    return G, H, W