% +------------------------------------------------------+
% |   Pink Noise Generation with MATLAB Implementation   | 
% |                                                      |
% | Author: Ph.D. Eng. Hristo Zhivomirov        07/31/13 | 
% +------------------------------------------------------+
%
% function: x = pinknoise(N)
%
% Input:
% N - number of samples to be returned in the noise column vector
% alpha - PSD spectral slope
% 
% Output:
% x - column vector of pink noise samples with unity  
%     standard deviation and zero mean value 
%
% The function generates a column vector of pink (flicker) noise 
% samples. In terms of power at a constant bandwidth, the pink  
% noise falls off by -3 dB/oct i.e., -10 dB/dec.

function x = pinknoise(N)

% input validation
validateattributes(N, {'double'}, ...
                      {'scalar', 'integer', 'nonnan', 'finite'}, ...
                      '', 'N', 1)

% set the PSD slope
alpha = -1; 

% convert from PSD (power specral density) slope 
% to ASD (amplitude spectral density) slope
alpha = alpha/2;

% generate AWGN signal
x = randn(1, N);

% calculate the number of unique fft points
NumUniquePts = ceil((N+1)/2);

% take fft of x
X = fft(x);

% fft is symmetric, throw away the second half
X = X(1:NumUniquePts);

% prepare a vector with frequency indexes 
n = 1:NumUniquePts;

% manipulate the left half of the spectrum so the spectral 
% amplitudes are proportional to the frequency by factor f^alpha
X = X.*(n.^alpha);

% perform ifft
if rem(N, 2)	% odd N excludes Nyquist point 
    % reconstruct the whole spectrum
    X = [X conj(X(end:-1:2))];
    
    % take ifft of X
    x = real(ifft(X));   
else            % even N includes Nyquist point  
    % reconstruct the whole spectrum
    X = [X conj(X(end-1:-1:2))];
    
    % take ifft of X
    x = real(ifft(X));  
end

% ensure zero mean value and unity standard deviation 
x = x - mean(x);
x = x/std(x, 1);
x = x(:);

end