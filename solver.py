import torch
import time
import os
import argparse

from utility import * 
from dataloader import *
from model import DiffFDN
from trainer import Trainer
from eq_design import FilterDesigner
# set fixed random seed for reproducibility
torch.manual_seed(130798)

def main(args, train_dataset, valid_dataset):
    """
    Main function for training and saving parameters and IR.

    Args:
        args: The command line arguments.
        train_dataset: The training dataset.
        valid_dataset: The validation dataset.
    """
    # initialize network 
    net = DiffFDN(args.delays, args.gain_per_sample, args.device, args.scattering, args.householder)
    # parameters initialization 
    net.apply(weights_init_normal)  
    # construct trainer
    trainer = Trainer(net, args)

    # save initial parameters and ir 
    save_parametes(trainer.net, args.train_dir, 'parameters_init.mat', scattering=args.scattering)
    trainer.save_ir(args.train_dir, filename='ir_init.wav')
    # TRAIN
    trainer.train(train_dataset, valid_dataset)
    # save final parameters and ir 
    save_parametes(trainer.net, args.train_dir, 'parameters.mat', scattering=args.scattering)
    trainer.save_ir(args.train_dir, filename='ir.wav')
    # save loss evolution
    save_loss(trainer.train_loss, trainer.valid_loss, args.train_dir, save_plot=False)

    with torch.no_grad():
        if args.reference_ir:
            # design filter form reference IR
            trainer.net.to('cpu')
            trainer.net.device = 'cpu'
            filter_designer = FilterDesigner(trainer.net, args.reference_ir, octave=args.octave_bands, method=args.edc_est_method)
            filter_designer.run_designer()

            trainer.net.rir_synthesis = True
            trainer.net.TC_SOS = filter_designer.TC_SOS
            trainer.net.G_SOS = filter_designer.G_SOS
            
            # save impulse respnse and filters 
            trainer.save_ir(args.train_dir, filename='ir_synthesis.wav')
            save_filters(trainer.net,filter_designer, args.train_dir, 'synthesis_filters.mat')
   
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--samplerate', type=int, default=48000,
        help ='sample rate')
    parser.add_argument('--train_dir',
        help ='path to output directory')
    parser.add_argument('--device', default='cuda',
        help='training device')
    # dataset 
    parser.add_argument('--num', type=int, default=480000,
        help = 'dataset size') 
    parser.add_argument('--split', type=float, default=0.8,
        help='training / validation split')
    parser.add_argument('--shuffle', action='store_false',
        help='if true, shuffle the data in the dataset at every epoch')
    # training
    parser.add_argument('--batch_size', type=int, default=2000,
        help='batch size')
    parser.add_argument('--max_epochs', type=int, default=20, 
        help='maximum number of training epochs')
    parser.add_argument('--log_epochs', action='store_true',
        help='Store met parameters at every epoch')
    # optimizer 
    parser.add_argument('--lr', type=float, default=1e-3,
        help='learning rate')
    parser.add_argument('--alpha', type=float, default=2.,
        help='temporal loss scaling factor. Suggested values 2 or 1 (if scattering == true)')
    # netowrk
    parser.add_argument('--gain_per_sample', type=float, default=0.9999, 
        help='gain per sample value gamma')
    parser.add_argument('--delays',  nargs='+', type=float,
        help='list of delay lengths')
    # frequency-dependent attenuation 
    parser.add_argument('--reference_ir', type=str, default=None,
        help='filepath to the reference ir, used to design an attenuation filter for rt60 matching')
    parser.add_argument('--edc_est_method', type=str, default='DecayFitNet', 
        help='Method used for the estimation of the EDC curves. One between DecayFitNet and BDA (Bayesian Decay Analysis).')
    parser.add_argument('--octave_bands', type=int, default=1, 
        help='Number of bands in one octave. One between 1 and 3. NOTE, this is still under development.')
    # scattering feedback matrix 
    parser.add_argument('--scattering', action='store_true', default=False,
        help='If true use the scattering FDN configuaration')
    parser.add_argument('--householder', action='store_true', default=False,
        help='If true use the householder feedback matrix')
    parser.add_argument('--transpose', action='store_true', default=False,
        help='If true use transposed configuration')
    args = parser.parse_args()

    # make output directory
    if args.train_dir is not None:
        if not os.path.isdir(args.train_dir):
            os.makedirs(args.train_dir)
    else:
        args.train_dir = os.path.join('output', time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(args.train_dir)

    # save arguments 
    with open(os.path.join(args.train_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    
    args.device = set_device(args.device)

    train_dataset, valid_dataset = load_dataset(args)

    main(args, train_dataset, valid_dataset)