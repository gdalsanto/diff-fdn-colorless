# diff-fdn-colorless  
In this work we introduce an optimization framework to tune a set of Feedback Delay Network (FDN) parameters (feedback matrix, input gains, and output gains) to achieve a smoother and less colored reverberation. 

This is the companion code to the **Feedback Delay Network Optimization** paper submitted for EURASIP Journal on Audio, Speech, and Music Processing, special issue on Digital Audio Effects [1].  
This is an extension to our previous work, which you can find in the branch `dafx23` of this repository and documented in the relative DAFx paper [2].  

Main updates from the previous framework:
* Faster training, we now concentrate solely on the frequency domain
* Support for scattering feedback matrix optimization, for improved temporal density
* Synthesis of real RIR using DecayFitNet to design attenuation and tone control filters
* Support for Householder feedback matrix optimization, for reduced computational costs (not yet available in this repo)
  
## Getting started 
When cloning this repository, make sure to clone all the submodules, namely [fdnToolbox](https://github.com/SebastianJiroSchlecht/fdnToolbox) and [DecayFitNet](https://github.com/georg-goetz/DecayFitNet/tree/01daf3e7bbfd637aa1269bbca0cab7f445db0d5d), by running
```
git clone --recurse-submodules git@github.com:gdalsanto/diff-fdn-colorless.git
```
To install the required packages using conda environments open the terminal at the repo directory and run the following command
```
conda env create -f diff-colorless-fdn-gpu.yml
```
Alternatively, use the CPU compatible environement `diff-colorless-fdn.yml`  

The optimization is coded in PyTorch. Run the `solver.py` file to launch training, the delay lines lengths must be given as arguments for the code to run. Check `solver.py` for the complete list of arguments. The initial and optimized parameters values are saved in `output/.`.  
## Demo 
The MATLAB demo code `inference.m` shows how to load the optimized FDN parameters on Sebastian Schlecht's [fdnToolbox](https://github.com/SebastianJiroSchlecht/fdnToolbox) 


## References
Audio demos are published in: [Feedback Delay Network Optimization](http://research.spa.aalto.fi/publications/papers/eurasip-colorless-fdn/).  
The paper is now on [arXiv](http://arxiv.org/abs/2402.11216)! 
```
[1] Dal Santo G., Prawda K., Schlecht S. J., and Välimäki V., "Feedback Delay Network Optimization." in EURASIP Journal on Audio, Speech, and Music Processing - sumbitted for reviews on 31.01.2024
[2] Dal Santo G., Prawda K., Schlecht S. J., and Välimäki V., "Differentiable Feedback Delay Network for colorless reverberation." in the 26th International Conference on Digital Audio Effects (DAFx23), Copenhagen, Denmark, Sept. 4-7 2023 
```
