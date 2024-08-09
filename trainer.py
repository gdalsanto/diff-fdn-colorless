import torch
import time
import torchaudio 
from losses import mse_loss, sparsity_loss
from utility import *
from tqdm import trange, tqdm

class Trainer:
    def __init__(self, net, args):
        self.net = net
        self.device = args.device
        self.alpha = args.alpha
        self.scattering = args.scattering
        self.max_epochs = args.max_epochs
        self.patience = 5
        self.early_stop = 0
        self.train_dir = args.train_dir

        self.optimizer = torch.optim.Adam(net.parameters(), lr=args.lr) 
        self.criterion = [mse_loss(), sparsity_loss()]
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1 )
        self.z = get_frequency_samples(args.samplerate*2).view(args.samplerate*2, -1)
        self.z_batch = get_frequency_samples(args.batch_size).view(args.batch_size, -1)

        self.normalize()

    def to_device(self):
        for i, criterion in enumerate(self.criterion):
            self.criterion[i] = self.criterion[i].to(self.device)
        
    # MAIN TRIANING METHOD
    def train(self, train_dataset, valid_dataset):
        self.train_loss, self.valid_loss, self.density, loss_sparsity, loss_spectral = [], [], [], [], []
        
        st = time.time()    # start time
        for epoch in trange(self.max_epochs, desc='Training'):
            st_epoch = time.time()

            # training
            epoch_loss = 0
            epoch_density = 0
            epoch_loss_spectral = 0
            epoch_loss_sparsity = 0
            for data in train_dataset:
                e_loss, dens, e_spec, e_temp = self.train_step(data)
                epoch_loss += e_loss
                epoch_density += dens
                epoch_loss_spectral += e_spec
                epoch_loss_sparsity += e_temp
            self.scheduler.step()
            self.train_loss.append(epoch_loss/len(train_dataset))
            self.density.append(dens/len(train_dataset))
            loss_spectral.append(epoch_loss_spectral/len(train_dataset))
            loss_sparsity.append(epoch_loss_sparsity/len(train_dataset))

            # validation
            epoch_loss = 0
            for data in valid_dataset:
                epoch_loss += self.valid_step(data)
            self.valid_loss.append(epoch_loss/len(valid_dataset))
            et_epoch = time.time()

            self.print_results(epoch, et_epoch-st_epoch)
            self.save_model(epoch)

            # early stopping
            if (epoch >=1):
                if (abs(self.valid_loss[-2] - self.valid_loss[-1]) <= 0.0001):
                    self.early_stop += 1
                else: 
                    self.early_stop = 0
            if self.early_stop == self.patience:
                break

        et = time.time()    # end time 
        print('Training time: {:.3f}s'.format(et-st))
        import scipy.io as sio

        # Save density to mat file
        sio.savemat(os.path.join(self.train_dir,'density.mat'), {'density': self.density})
        sio.savemat(os.path.join(self.train_dir,'train_loss.mat'), {'train_loss': self.train_loss})
        sio.savemat(os.path.join(self.train_dir,'losses_partial.mat'), {'spectral': loss_spectral, 'sparsity': loss_sparsity})


    def train_step(self, data):
        # batch processing
        inputs, labels = data 
        self.optimizer.zero_grad()
        H = self.net(inputs)
        loss = self.criterion[0](H, labels) + self.alpha*self.criterion[1](self.net.ortho_param(self.net.A))
        
        loss.backward()
        self.optimizer.step()

        h = self.get_ir()
        density = torch.norm(h, p=1, dim=-1) / torch.norm(h, p=2, dim=-1)
        return loss.item(), density.item(), self.criterion[0](H, labels).item(), self.criterion[1](self.net.ortho_param(self.net.A)).item()

    def valid_step(self, data):
        # batch processing
        inputs, labels = data 
        self.optimizer.zero_grad()
        H = self.net(inputs)
        loss = self.criterion[0](H, labels) + self.alpha*self.criterion[1](self.net.ortho_param(self.net.A))
        return loss.item()
    
    @torch.no_grad()
    def normalize(self):
        # average enery normalization
        H, _ = get_response(self.z_batch, self.net)
        energyH = torch.sum(torch.pow(torch.abs(H),2)) / torch.tensor(H.size(0))

        # apply energy normalization on input and output gains only
        for name, prm in self.net.named_parameters():
            if name == 'B' or name == 'C':    
                prm.data.copy_(torch.div(prm.data, torch.pow(energyH, 1/4)))

    def print_results(self, e, e_time):
        print(get_str_results(epoch=e, 
                              train_loss=self.train_loss, 
                              valid_loss=self.valid_loss, 
                              time=e_time))

    def save_model(self, e):
        dir_path = os.path.join(self.train_dir, 'checkpoints')
        # create checkpoint folder 
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)  
        # save model 
        torch.save(
            self.net.state_dict(), 
            os.path.join(dir_path, 'model_e' + str(e) + '.pt'))   
        
    @torch.no_grad()
    def save_ir(self, dir, filename='ir.wav', norm=False):
        if self.scattering:
            print('Cannot compute the impulse response for FDNs with scattering feedback matrices')
            return
        _, h = get_response(self.z, self.net)
        if norm:
            h = torch.div(h, torch.max(torch.abs(h)))
        filepath = os.path.join(dir, filename)
        torchaudio.save(filepath,
                        torch.stack((h.squeeze(0),h.squeeze(0)),1).cpu(),
                        48000,
                        bits_per_sample=32,
                        channels_first=False)        
    
    def get_ir(self):
        if self.scattering:
            print('Cannot compute the impulse response for FDNs with scattering feedback matrices')
            return
        _, h = get_response(self.z, self.net)
        return h