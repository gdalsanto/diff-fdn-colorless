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

import config
from utility import * 
from dataloader import split_dataset, get_dataloader, Dataset
from losses import asy_p_loss, sparsity_loss
from models import DiffFDN

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == torch.device('cuda'):
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)

# ----------- DATASET
# get training and valitation dataset
dataset = Dataset(config.num, config.min_nfft, config.max_nfft, device)
# split data into training and validation set 
train_set, valid_set = split_dataset(
    dataset, config.split)

# dataloaders
train_loader = get_dataloader(
    train_set,
    batch_size=config.batch_size,
    shuffle = config.shuffle,
)
  
valid_loader = get_dataloader(
    valid_set,
    batch_size=config.batch_size,
    shuffle = config.shuffle,
)

# initialize network 
net = DiffFDN(config.N, config.gain_per_sample, config.learnDelays, device)
# parameters initialization 
net.apply(weights_init_normal)

# ----------- TRAINING CONFIGURATION
# optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=config.lr) 
# spectral loss
criterionFreq = asy_p_loss()
criterionFreq = criterionFreq.to(device)
# temporal loss 
criterionTime = sparsity_loss()
criterionTime = criterionTime.to(device)
# scheduler 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1 )

# initialize training variables
x = get_frequency_samples(config.samplerate*2)
x = torch.tensor(x.view(x.size(0), -1))
H_t = []       
train_loss, valid_loss = [], []
save_lines = True    # if true saves in .wav the output of the delay lines

# energy normalization
with torch.no_grad():
    inputs, _ = next(iter(train_loader))
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
param_filename = 'parameters_init.mat' + str(uuid.uuid4())
config_filename = 'config_' + str(uuid.uuid4()) + '.py'
shutil.copy('config.py', config_filename)

save_parametes(net, 'output', filename=param_filename)

# training start time
st = time.time()     

# ----------- TRAINING LOOP
for epoch in range(config.max_epochs):
    epoch_loss = 0
    st_epoch = time.time()

    for i, data in enumerate(train_loader):
        # batch processing
        inputs, labels = data 
        optimizer.zero_grad()
        H, h, param = net(inputs)
        loss = criterionFreq(H, labels) + config.alpha*criterionTime(torch.abs(h), torch.ones(480000))

        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss.append(epoch_loss/len(train_loader))
    scheduler.step()

    # ----------- VALIDATION
    epoch_loss = 0
    for i, data in enumerate(valid_loader):
        inputs, labels = data
        
        optimizer.zero_grad()
        H, h, _ = net(inputs)
        loss = criterionFreq(H, labels) + config.alpha*criterionTime(torch.abs(h), torch.ones(480000))
        
        epoch_loss += loss.item()
    
    valid_loss.append(epoch_loss/len(valid_loader))
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
output_dir, _, _= save_output(net, config_filename = config_filename)
save_parametes(net, output_dir, filename='parameters.mat')
move_item('output/' + param_filename,
    output_dir+'/parameters_init.mat')
save_loss(train_loss, valid_loss, output_dir, save_plot=False)

# save final impulse response  
A, B, C, Gamma, m = param 
D = torch.diag_embed(x ** m)
H = torch.matmul(torch.matmul(C, torch.inverse(D - torch.matmul(A,Gamma))), B).squeeze()
h = torch.fft.irfft(H)
h_norm = torch.div(h, torch.max(torch.abs(h)))
filename = os.path.join(output_dir,'ir.wav')
torchaudio.save(filename,
                torch.stack((h_norm.squeeze(0),h_norm.squeeze(0)),1).detach().cpu(),
                48000,
                bits_per_sample=32,
                channels_first=False)