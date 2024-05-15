import argparse
import warnings
import os 
import scipy 
from glob import glob
import numpy as np
import soundfile as sf
from filters.eq import *
from filters.utils import * 
from decayfit import * 
import matplotlib.pyplot as plt 

def main(args):
    """
    Main function for analyzing room impulse responses (RIRs) and estimating parameters.
    The function reads the RIRs from the directory specified in the command-line arguments, 
    estimates the EDC parameters using the specified method and saves the estimated parameters in .mat format.

    Args:
        args (object): Command-line arguments.
    Returns:
        None
    """
    device = 'cpu'
    # read room impulse responses
    pathlist = [y for x in os.walk(args.dir_path) for y in glob(os.path.join(x[0], '*.wav'))] 

    for filepath in pathlist:
        # read the rir
        rir, sr = sf.read(filepath, dtype='float32')
        # convert stereo/multichannel to mono by taking first channel only
        if len(rir.shape) > 1:
            rir = rir[:, 0]
        # get center frequencies and correction factors
        f_bands, correction = get_center_freq(args) 
        # get the decay model
        n_slopes = 1 # number of slopes in the decay model
        if args.edc_est_method == 'DecayFitNet':
            edc_param, norm_vals, _ = getEDCparam(rir, f_bands, n_slopes = n_slopes, sr=sr, device=device)
            T, A, N = edc_param[0], edc_param[1], edc_param[2]
        elif args.edc_est_method == 'BDA':
            parameter_ranges = {'t_range': [0.1, 3.5],
                                'a_range': [-3, 0],
                                'n_range': [-10, -2]}
            n_iterations = 100
            # Init Bayesian decay analysis
            bda = BayesianDecayAnalysis(n_slopes, sr, parameter_ranges, n_iterations, filter_frequencies=f_bands)
            edc_param, norm_vals = bda.estimate_parameters(rir)
            T, A, N = edc_param[0], edc_param[1], edc_param[2]
        else:
            ValueError('EDC estimation method must be one between DecayFitNet and BDA')
        
        # Init Preprocessing for Schroeder integration (reference EDC calculation)
        rir_preprocessing = PreprocessRIR(sample_rate=sr, filter_frequencies=f_bands)

        # Schroeder integration, analyse_full_rir: if RIR onset should be detected, set this to False
        true_edc, __ = rir_preprocessing.schroeder(rir, analyse_full_rir=True)
        time_axis = (torch.linspace(0, true_edc.shape[2] - 1, true_edc.shape[2]) / sr)

        # Permute into [n_bands, n_batches, n_samples]
        true_edc = true_edc.permute(1, 0, 2)
        fitted_edc = decay_model(torch.from_numpy(T).to(device),
                                    torch.from_numpy(A).to(device),
                                    torch.from_numpy(N).to(device),
                                    time_axis=time_axis.to(device),
                                    compensate_uli=True,
                                    backend='torch',
                                    device=device)

        # Discard last 5% for MSE evaluation
        true_edc = discard_last_n_percent(true_edc, 5)
        fitted_edc = discard_last_n_percent(fitted_edc, 5)

        plot_edc = False   # Set to True to print the EDCs
        # Calculate MSE between true EDC and fitted EDC
        mse_per_frequencyband = get_mse(true_edc, fitted_edc, f_bands)
        # plot the fitter edc vs the true edc 
        if plot_edc:
            for i, band in enumerate(f_bands):
                if i == len(f_bands)-2:
                    print('ok')
                plt.plot(discard_last_n_percent(time_axis, 5), 20*torch.log10(true_edc[i, :, :]).squeeze())
                plt.plot(discard_last_n_percent(time_axis, 5), 20*torch.log10(fitted_edc[i, :, :]).squeeze())
                plt.title("EDC band " + str(band) + " MSE Error {:.02f} dB".format( mse_per_frequencyband[i].item()))
                plt.legend(['true', 'fitted'])
                plt.show()
                plt.clf()
        
        # save estimated parameters 
        est = {}
        est['T'] = T
        est['A'] = A
        est['N'] = N 
        est['norm_vals'] = norm_vals
        filename = os.path.join(os.path.basename(filepath)).split('.')[0] + "_est.mat"
        if args.out_dir_path is None:
            args.out_dir_path = args.dir_path
        scipy.io.savemat(os.path.join(args.out_dir_path, filename),
                        est)

def get_mse(ground_truth_edc, estimated_edc, f_bands):
    """
    Calculates the mean squared error (MSE) between the ground truth energy decay curve (EDC) and the estimated EDC.

    Args:
        ground_truth_edc (torch.Tensor): The ground truth energy decay curve.
        estimated_edc (torch.Tensor): The estimated energy decay curve.
        f_bands (list): A list of frequency bands.
    Returns:
        torch.Tensor: The MSE between the input EDCs and estimated fits.
    """
    
    loss_fn = torch.nn.MSELoss(reduction='none')
    this_mse = torch.sqrt(torch.mean(loss_fn(10 * torch.log10(ground_truth_edc), 10 * torch.log10(estimated_edc)), 2))
    print('==== Average MSE between input EDCs and estimated fits: {:.02f} dB ===='.format(float(torch.mean(this_mse))))
    this_mse_bands = this_mse.squeeze().tolist()
    if len(f_bands) == 9:
        print('MSE between input EDC and estimated fit for different frequency bands: 64 Hz: {:.02f} dB -- 125 Hz: {:.02f} dB -- '
            '250 Hz: {:.02f} dB -- 500 Hz: {:.02f} dB -- 1 kHz: {:.02f} dB -- 2 kHz: {:.02f} dB -- '
            '4 kHz: {:.02f} dB -- 8 kHz: {:.02f} dB -- 16 kHz: {:.02f} dB'.format(*this_mse_bands))
    elif len(f_bands) == 25:
        print('MSE between input EDC and estimated fit for different frequency bands: 64 Hz: {:.02f} dB -- 80 Hz: {:.02f} dB -- 100 Hz: {:.02f} dB -- 125 Hz: {:.02f} dB -- '
            '160 Hz: {:.02f} dB -- 200 Hz: {:.02f} dB -- 250 Hz: {:.02f} dB -- 315 Hz: {:.02f} dB -- 400 Hz: {:.02f} dB -- 500 Hz: {:.02f} dB -- 630 Hz: {:.02f} dB -- 800 Hz: {:.02f} dB -- 1 kHz: {:.02f} dB -- 1.25 kHz: {:.02f} dB -- 1.6 kHz: {:.02f} dB -- 2 kHz: {:.02f} dB -- '
            '2.5 kHz: {:.02f} dB -- 3.15 kHz: {:.02f} dB -- 4 kHz: {:.02f} dB -- 5 kHz: {:.02f} dB -- 6.3 kHz: {:.02f} dB -- 8 kHz: {:.02f} dB -- 10 kHz: {:.02f} dB -- 10.25 kHz: {:.02f} dB -- 16 kHz: {:.02f} dB'.format(*this_mse_bands))
    else:
        warnings.warn('Unsupported fBands for printout')

    if torch.mean(this_mse) > 5:
        warnings.warn('High MSE value detected. The obtained fit may be bad.')
        print('!!! WARNING !!!: High MSE value detected. The obtained fit may be bad. You may want to try:')
        print('1) Increase fadeout_length. This decreases the upper limit of integration, thus cutting away more from '
              'the end of the EDC. Especially if your RIR has fadeout windows or very long silence at the end, this can'
              'improve the fit considerably.')
        print('2) Manually cut away direct sound and potentially strong early reflections that would cause the EDC to '
              'drop sharply in the beginning.')

    return this_mse
    

def get_center_freq(args):
    """
    Compute the center frequencies and correction factors for a given octave band resolution.
    Args:
        args (Namespace): The command line arguments.
    Returns:
        tuple: A tuple containing the center frequencies and correction factors.
    """
    if args.octave_bands == 1:
        f_bands = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]   
        correction = np.array([0.9, 1, 1, 1, 1, 1, 1, 1, 1, 0.9, 0.5]) 
    elif args.octave_bands == 3:
        f_bands = [63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000]
        correction = np.array([0.9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  0.9, 0.5])
    else:
        raise ValueError('Resolutions different from one or one-third octave bands are not supported at the moment')
    
    return f_bands, correction

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type=str, default=None,
        help='path to the directory containing the rirs to analyse')
    parser.add_argument('--out_dir_path', type=str, default=None,
        help='path to the directory where to save parameters to')
    parser.add_argument('--edc_est_method', type=str, default='DecayFitNet', 
        help='Method used for the estimation of the EDC curves. One between DecayFitNet and BDA (Bayesian Decay Analysis).')
    parser.add_argument('--octave_bands', type=int, default=1, 
        help='Number of bands in one octave. One between 1 and 3.')
    args = parser.parse_args()

    main(args)