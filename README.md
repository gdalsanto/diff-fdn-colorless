# diff-fdn-colorless
Demo for the differentiable feedback delay network (FDN) for colorless reverberation submitted to the 26th International Conference on Digital Audio Effects (DAFx 2023). 
## Overview 
In this work we introduce an optimization framework to tune a set of FDN parameters (feedback matrix, input gains, and output gains) to achieve a smoother and less colored reverberation. 
## Getting started 
To install the required packages using conda environments open the terminal at the repo directory and run the following command
```
conda env create -f colorless-fdn.yml
```
The optimization is coded in Pytorch. Set the configuration parameters in `config.py` and launch the training by running `solver.py`. The initial and optimized parameters values are saved in `output/.`.

The repository also contains a MATLAB demo `demo.m` that shows how to load the model parameters in matlab and uses Sebastian Schlecht's [fdnToolbox](https://github.com/SebastianJiroSchlecht/fdnToolbox) to compute the impulse response and modal decomposition. 

## References
Audio demos are published in: [Differentiable Feedback Delay Network for Colorless Reverberationg](http://research.spa.aalto.fi/publications/papers/dafx23-colorless-fdn/).  
If you would like to use this code, please cite the related DAFx conference paper (submitted) using the following reference:
```
Dal Santo Gloria, Karolina Prawda, Sebastian J. Schlecht, and Vesa Välimäki. "Differentiable Feedback Delay Network for colorless reverberation." International Conference on Digital Audio Effects (DAFx23), Copenhagen, Denmark, Sept. 4-7 2023 
```