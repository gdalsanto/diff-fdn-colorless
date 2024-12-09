import torch
import argparse
import os
import time
import scipy
import shutil
import scipy.io as sio

from collections import OrderedDict

from flamo.optimize.dataset import DatasetColorless, load_dataset
from flamo.optimize.trainer import Trainer
from flamo.processor import dsp, system
from flamo.optimize.loss import masked_mse_loss, sparsity_loss
from flamo.utils import save_audio


torch.manual_seed(130709)

def main(args):
    """
    Example function that demonstrates the construction and training of a Feedback Delay Network (FDN) model
    with scattering feedback matrix and sparse marsking of the loss.
    Args:
        args: A dictionary or object containing the necessary arguments for the function.
    Returns:
        None
    """

    # FDN parameters
    N = args.N  # number of delays
    alias_decay_db = 0  # alias decay in dB
    if N == 4:
        delay_lengths = torch.tensor([1499, 1889, 2381, 2999])
    elif N == 6:
        delay_lengths = torch.tensor([997, 1153, 1327, 1559, 1801, 2099])
    elif N == 8:
        delay_lengths = torch.tensor([809, 877, 937, 1049, 1151, 1249, 1373, 1499])

    args.num = (args.nfft//2+1) // 2000
    gain_per_sample = torch.tensor(0.9999)
    
    ## ---------------- CONSTRUCT FDN ---------------- ##
    # Input and output gains
    input_gain = dsp.Gain(
        size=(N, 1), nfft=args.nfft, requires_grad=True, alias_decay_db=alias_decay_db, device=args.device
    )
    output_gain = dsp.Gain(
        size=(1, N), nfft=args.nfft, requires_grad=True, alias_decay_db=alias_decay_db, device=args.device
    )
    # Feedback loop with delays
    delays = dsp.parallelDelay(
        size=(N,),
        max_len=delay_lengths.max(),
        nfft=args.nfft,
        isint=True,
        requires_grad=False,
        alias_decay_db=alias_decay_db,
        device=args.device,
    )
    delays.assign_value(delays.sample2s(delay_lengths))

    if args.feedback_type == 'orthogonal':
        # Feedback path with orthogonal matrix
        mixing_matrix = dsp.Matrix(
            size=(N, N),
            nfft=args.nfft,
            matrix_type='orthogonal',
            requires_grad=True,
            device=args.device,
        )
        attenuation = dsp.parallelGain(
            size=(N, ),
            nfft=args.nfft,
            requires_grad=False,
            alias_decay_db=alias_decay_db,
            device=args.device
        )
        attenuation.assign_value(gain_per_sample**delay_lengths)
        # read the initial parameters from dafx23
        param = sio.loadmat(os.path.join('eurasip-plots','table3-4','dafx23', "N{:02d}".format(len(delay_lengths)), "N{}-{:03d}".format(len(delay_lengths), args.run_n), 'parameters_init.mat'))
        input_gain.assign_value(torch.tensor(param['B']).transpose(1,0))
        output_gain.assign_value(torch.tensor(param['C']))
        mixing_matrix.assign_value(torch.tensor(param['A']))
        
    elif args.feedback_type == 'scattering':
        # Feedback path with scattering matrix
        m_L =  torch.randint(low=1, high=int(torch.floor(min(delay_lengths)/2)), size=[N]) 
        m_R =  torch.randint(low=1, high=int(torch.floor(min(delay_lengths)/2)), size=[N]) 
        mixing_matrix = dsp.ScatteringMatrix(
            size=(4, N, N),
            nfft=args.nfft,
            gain_per_sample=gain_per_sample,
            sparsity=3,
            m_L=m_L,
            m_R=m_R,
            alias_decay_db=alias_decay_db,
            requires_grad=True,
            device=args.device,
        )
    elif args.feedback_type == 'householder':   
        # Feedback path with householder matrix
        mixing_matrix = dsp.HouseholderMatrix(
            size=(4, N, N),
            nfft=args.nfft,
            gain_per_sample=1,
            sparsity=3,
            requires_grad=True,
            device=args.device,
        )
        attenuation = dsp.parallelGain(
            size=(N, ),
            nfft=args.nfft,
            requires_grad=False,
            alias_decay_db=alias_decay_db,
            device=args.device
        )
        attenuation.assign_value(gain_per_sample**delay_lengths)

    
    if args.feedback_type == 'orthogonal' or args.feedback_type == 'householder':
        feedback = system.Series(OrderedDict({
            'mixing_matrix': mixing_matrix,
            'attenuation': attenuation
        }))
    else:
        feedback = mixing_matrix
        
    # Recursion
    feedback_loop = system.Recursion(fF=delays, fB=feedback)

    # Full FDN
    FDN = system.Series(OrderedDict({
        'input_gain': input_gain,
        'feedback_loop': feedback_loop,
        'output_gain': output_gain
    }))

    # Create the model with Shell
    input_layer = dsp.FFT(args.nfft)
    output_layer = dsp.Transform(transform=lambda x : torch.abs(x))
    model = system.Shell(core=FDN, input_layer=input_layer, output_layer=output_layer)

    # Get initial impulse response
    with torch.no_grad():
        ir_init =  model.get_time_response(identity=False, fs=args.samplerate).squeeze() 
        save_audio(os.path.join(args.train_dir, "ir_init.wav"), ir_init/torch.max(torch.abs(ir_init)), fs=args.samplerate)
        save_fdn_params(model, filename='parameters_init')

    ## ---------------- OPTIMIZATION SET UP ---------------- ##

    dataset = DatasetColorless(
        input_shape=(1, args.nfft // 2 + 1, 1),
        target_shape=(1, args.nfft // 2 + 1, 1),
        expand=args.num,
        device=args.device,
    )
    train_loader, valid_loader = load_dataset(dataset, batch_size=args.batch_size)

    # Initialize training process
    trainer = Trainer(model, max_epochs=args.max_epochs, lr=args.lr, train_dir=args.train_dir, device=args.device)
    trainer.register_criterion(masked_mse_loss(nfft=args.nfft, n_samples=2000, n_sets=1, device=args.device), 1)
    trainer.register_criterion(sparsity_loss(), 0.2, requires_model=True)
    ## ---------------- TRAIN ---------------- ##

    # Train the model
    trainer.train(train_loader, valid_loader)

    # remove checkpoints
    shutil.rmtree(os.path.join(args.train_dir, 'checkpoints'))

    # Get optimized impulse response
    with torch.no_grad():
        ir_optim =  model.get_time_response(identity=False, fs=args.samplerate).squeeze()
        save_audio(os.path.join(args.train_dir, "ir_optim.wav"), ir_optim/torch.max(torch.abs(ir_optim)), fs=args.samplerate)
        save_fdn_params(model, filename='parameters_optim')

def save_fdn_params(net, filename='parameters'):
    r"""
    Retrieves the parameters of a feedback delay network (FDN) from a given network and saves them in .mat format.

    **Parameters**:
        net (Shell): The Shell class containing the FDN.
        filename (str): The name of the file to save the parameters without file extension.
    **Returns**:
        dict: A dictionary containing the FDN parameters.
            - 'A' (ndarray): The feedback loop parameter A.
            - 'B' (ndarray): The input gain parameter B.
            - 'C' (ndarray): The output gain parameter C.
            - 'm' (ndarray): The feedforward parameter m.
    """

    core = net.get_core()
    param = {}
    param['A'] = core.feedback_loop.feedback.mixing_matrix.param.squeeze().detach().cpu().numpy()
    param['B'] = core.input_gain.param.squeeze().detach().cpu().numpy()
    param['C'] = core.output_gain.param.squeeze().detach().cpu().numpy()
    param['m'] = core.feedback_loop.feedforward.s2sample(core.feedback_loop.feedforward.map(core.feedback_loop.feedforward.param)).squeeze().detach().cpu().numpy()

    scipy.io.savemat(os.path.join(args.train_dir, filename + '.mat'), param)

    return param

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--nfft", type=int, default=480000*2, help="FFT size")
    parser.add_argument("--N", type=int, default=6, help="number of delays")
    parser.add_argument("--samplerate", type=int, default=48000, help="sampling rate")
    parser.add_argument('--device', type=str, default='cpu', help='device to use for computation')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('--max_epochs', type=int, default=20, help='maximum number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--train_dir', type=str, help='directory to save training results')
    parser.add_argument('--masked_loss', action='store_true', help='use masked loss')
    parser.add_argument('--feedback_type', type = str, default='orthogonal', help='Type of feedback matrix to use')
    parser.add_argument('--n_runs', type=int, default=1, help='number of runs')
    parser.add_argument('--testname', type=str, default='test')
    args = parser.parse_args()

    if args.n_runs == 1:
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
        args.run_n = None
        # lunch training
        main(args)
    else:
        for i in range(args.n_runs):
            # make output directory
            args.train_dir = os.path.join('eurasip-plots','table3-4','output', args.testname, "N{}-{:03d}".format(args.N, i))
            os.makedirs(args.train_dir)
            # save arguments 
            with open(os.path.join(args.train_dir, 'args.txt'), 'w') as f:
                f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
            args.run_n = i
            main(args)

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
    
class map_gamma(torch.nn.Module):

    def __init__(self, delays):
        super().__init__()
        self.delays = delays
        self.g_min = 0.99
        self.g_max = 1.0

    def forward(self, x):
        return (torch.sigmoid(x[0]) * (self.g_max - self.g_min) + self.g_min)**self.delays