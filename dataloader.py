# dataset and dataloader colorless fdn

import torch
import numpy as np 
import torch.utils.data as data

class Dataset(torch.utils.data.Dataset):
    ''' dataset of frequency samples (in rads) sampled at linearly spaced 
    points along the unit circle '''
    def __init__(self, num, min_nfft, max_nfft, device):
        # num: number of input pairs

        angle = torch.zeros((num, max_nfft))
        abs = torch.ones((num, max_nfft))
        for i in range(num):
            nfft = torch.randint(min_nfft, max_nfft, (1,))
            angle[i,0:nfft] = torch.linspace(0, 1, nfft.item())
            abs[i,nfft:] = 0

        self.input = torch.polar(abs, angle * np.pi)
        self.labels = torch.ones((num, max_nfft))
        self.input = self.input.to(device)
        self.labels = self.labels.to(device)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        # select sample
        y = self.labels[index,:]
        x = self.input[index,:]
        
        return x, y

def split_dataset(dataset, split):
    ''' randomly split a dataset into non-overlapping new datasets of 
    sizes given in 'split' argument'''
    # use split % of dataset for validation 
    train_set_size = int(len(dataset) * split)
    valid_set_size = len(dataset) - train_set_size
    
    seed = torch.Generator(device=get_device()).manual_seed(42)
    train_set, valid_set = data.random_split(dataset, [train_set_size, valid_set_size], generator=seed)

    return train_set, valid_set

def get_dataloader(dataset, batch_size, shuffle=True):
    ''' create torch dataloader form given dataset '''
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator(device=get_device()),
        drop_last = True
    )
    return dataloader

def get_device():
    ''' output 'cuda' if gpu is available, 'cpu' otherwise '''
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')