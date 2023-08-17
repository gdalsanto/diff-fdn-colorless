# Differentiable FDN for Colorless Reverberation 
# main scropt for the training of DiffFDN 
# training and FDN parameters are set in .config.py 
# FDN parameters values are stored in \output
# 
# If you would like to use this code, please cite the related DAFx conference 
# paper using the following reference:
# Dal Santo, Gloria, Karolina Prawda, Sebastian J. Schlecht, and Vesa Välimäki. 
# "Differentiable Feedback Delay Network for colorless reverberation." 
# International Conference on Digital Audio Effects (DAFx23), Copenhagen, 
# Denmark, Sept. 4-7 2023 

import torch
import time
import torchaudio 
import os
import uuid
import shutil
import argparse
import sys
import glob

from utility import * 
from dataloader import split_dataset, get_dataloader, Dataset
from losses import asy_p_loss, sparsity_loss
from models import DiffFDN

def set_device(device):
    if (device == 'cuda') & torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    else:
        device = 'cpu'
    return device

def load_dataset(args):
    # get training and valitation dataset
    dataset = Dataset(args.num, args.min_nfft, args.max_nfft, args.device)
    # split data into training and validation set 
    train_set, valid_set = split_dataset(
        dataset, args.split)

    # dataloaders
    train_loader = get_dataloader(
        train_set,
        batch_size=args.batch_size,
        shuffle = args.shuffle,
    )
    
    valid_loader = get_dataloader(
        valid_set,
        batch_size=args.batch_size,
        shuffle = args.shuffle,
    )
    return train_loader, valid_loader 

def train(args, train_dataset, valid_dataset):
    # initialize network 
    net = DiffFDN(args.delays, args.gain_per_sample, args.device)
    # parameters initialization 
    net.apply(weights_init_normal)

    # ----------- TRAINING CONFIGURATION
    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr) 
    # spectral loss
    criterionFreq = asy_p_loss()
    criterionFreq = criterionFreq.to(args.device)
    # temporal loss 
    criterionTime = sparsity_loss()
    criterionTime = criterionTime.to(args.device)
    # scheduler 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1 )

    # initialize training variables
    x = get_frequency_samples(args.samplerate*2)
    x = torch.tensor(x.view(x.size(0), -1))
    train_loss, valid_loss = [], []

    # energy normalization
    with torch.no_grad():
        inputs, _ = next(iter(train_dataset))
        _, _, param = net(inputs) 
        A, B, C, Gamma, m = param 
        D = torch.diag_embed(x ** m)
        H = torch.matmul(torch.matmul(C, torch.inverse(D - torch.matmul(A,Gamma))), B).squeeze()
        energyH = torch.sum(torch.pow(torch.abs(H),2)) / torch.tensor(H.size())

        # apply energy normalization on input and output gains only
        for name, param in net.named_parameters():
            if name == 'B' or name == 'C':    
                param.data.copy_(torch.div(param.data, torch.pow( energyH, 1/4)))

    # save initial parameters and config file
    param_filename = 'parameters_init.mat' 

    save_parametes(net, args.train_dir, param_filename)

    # training start time
    st = time.time()     

    # ----------- TRAINING LOOP
    for epoch in range(args.max_epochs):
        epoch_loss = 0
        st_epoch = time.time()

        for i, data in enumerate(train_dataset):
            # batch processing
            inputs, labels = data 
            optimizer.zero_grad()
            H, h, param = net(inputs)
            loss = criterionFreq(H, labels) + args.alpha*criterionTime(torch.abs(h), torch.ones(h.size(-1)))

            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss.append(epoch_loss/len(train_dataset))
        scheduler.step()

        # ----------- VALIDATION
        epoch_loss = 0
        for i, data in enumerate(valid_dataset):
            inputs, labels = data
            
            optimizer.zero_grad()
            H, h, _ = net(inputs)
            loss = criterionFreq(H, labels) + args.alpha*criterionTime(torch.abs(h), torch.ones(480000))
            
            epoch_loss += loss.item()
        
        valid_loss.append(epoch_loss/len(valid_dataset))
        et_epoch = time.time()
        to_print = get_str_results(
            epoch=epoch, 
            train_loss=train_loss, 
            valid_loss=valid_loss, 
            time=et_epoch-st_epoch)
        print(to_print)

    # end time 
    et = time.time()    
    print('Training time: {:.3f}s'.format(et-st))

    net.print()
    # save optimized parameters and loss values
    save_output(net, args.train_dir)
    # save_parametes(net, args.train_dir)
    save_loss(train_loss, valid_loss, args.train_dir, save_plot=False)

    # save final impulse response  
    A, B, C, Gamma, m = param 
    D = torch.diag_embed(x ** m)
    H = torch.matmul(torch.matmul(C, torch.inverse(D - torch.matmul(A,Gamma))), B).squeeze()
    h = torch.fft.irfft(H)
    h_norm = torch.div(h, torch.max(torch.abs(h)))
    filename = os.path.join(args.train_dir,'ir.wav')
    torchaudio.save(filename,
                    torch.stack((h_norm.squeeze(0),h_norm.squeeze(0)),1).cpu(),
                    48000,
                    bits_per_sample=32,
                    channels_first=False)

if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument('--samplerate', type=int, default=48000,
        help ='sample rate')
    parser.add_argument('--train_dir',
        help ='path to output directory')
    parser.add_argument('--device', default='cuda',
        help='training device')
    # dataset 
    parser.add_argument('--num', type=int, default=256,
        help = 'dataset size')
    parser.add_argument('--min_nfft', type=int, default=384000,
        help='min number of fft point M_min')
    parser.add_argument('--max_nfft', type=int, default=480000,
        help='max number of fft point M_max')    
    parser.add_argument('--split', type=float, default=0.8,
        help='training / validation split')
    parser.add_argument('--shuffle', action='store_false',
        help='if true, shuffle the data in the dataset at every epoch')
    # training
    parser.add_argument('--batch_size', type=int, default=4,
        help='batch size')
    parser.add_argument('--max_epochs', type=int, default=15, 
        help='maximum number of training epochs')
    # optimizer 
    parser.add_argument('--lr', type=float, default=1e-3,
        help='learning rate')
    parser.add_argument('--alpha', type=int, default=300,
        help='temporal loss scaling factor')
    # netowrk
    parser.add_argument('--gain_per_sample', type=float, default=0.9999, 
        help='gain per sample value gamma')
    parser.add_argument('--delays',  nargs='+', type=float,
        help='list of delay lengths')

    args = parser.parse_args()
    print(args.delays)
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

    train(args, train_dataset, valid_dataset)