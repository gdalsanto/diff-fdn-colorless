% Differentiable FDN for Colorless Reverberation 
% demo code 

clear; clc; close all;

addpath(genpath('fdnToolbox'))
addpath(genpath('DecayFitNet'))

results_dir = "./test";
output_dir = fullfile(results_dir, 'matlab');

%% Analyse reference RIR 

delays = [593., 743., 929., 1153., 1399., 1699.];
N = length(delays);
fs = 48000; 
irLen = 3*fs;

% load RIR
filename = "s3_r4_o";
rir = audioread(fullfile('rirs',filename + ".wav"));
rir = rir(:,1);
[~,onset] = max(abs(rir));
rir = rir(onset:end,:);
rir = rir ./ norm(rir);
rirLen = size(rir,1);

nSlopes = 1; % approximate RIR with a single slope
fBands = [63, 125, 250, 500, 1000, 2000, 4000, 8000];

% load and process estimations from decay fit net 
load(fullfile('rirs', filename + "_DecayFitNet_est.mat"))
est.T = double(T);  est.A = double(A); est.N = double(N); est.norm = double(norm); 
clear norm A T N
est = transposeAllFields(est);
[est.L, est.A, est.N] = decayFitNet2InitialLevel(est.T, est.A, est.N, est.norm, fs, rirLen, fBands);

% absorption filters, shorten top and bottom band
T60frequency = [1, fBands fs];
targetT60 = est.T([1 1:end end]);  % seconds

% initial level filter, attenuate top and bottom band
targetLevel = mag2db(est.L([1 1:end end]));  % dB
targetLevel = targetLevel - [5 0 0 0 0 0 0 0 5 30];
equalizationSOS = designGEQ(targetLevel);

% load FDN parameters
load(fullfile(results_dir,'parameters.mat'))
B = B(:);        % input gains as column
D = zeros(1:1);     % direct gains
C = zSOS(permute(equalizationSOS,[3 4 1 2]) .*  C);

%% Mode Distribution at different attenuations 

% FIRST test the  repeatability of the modal decomposition algorithm
% homogeneous attenatuion 
% RT = ones(1, 6);  % reverberation time (s)
% figure('Name', 'Test Modal Decomposition')
% for i_RT = 1:length(RT)
%     g = 10^(-3/fs/RT(i_RT));  % gain per sample 
%     Gamma = diag(g.^delays);
%     % make matrix A orthogonal 
%     U = double(expm(skew(A*Gamma)));
%     [residues, poles, ~, ~, ~] = dss2pr(delays, U, B, C, D);
%     % h = histogram(db(abs(residues)),'BinWidth',1);
%     [N, edges] = histcounts(db(abs(residues)),'BinWidth',1);
%     plot(edges(1:end-1), N)
%     hold on;
% end
% % frequnecy dependent attenuation 
% legend(string(RT))
% title('Modal excitation at different RT')
% xlabel('Residue Magnitude (dB)')
% ylabel('Number of Occurence')

% homogeneous attenatuion 
RT = [0.5, 1., 1.5, 2., 5., 10.];  % reverberation time (s)
figure('Name', 'Homogeneous Decay')

stdev_hm = zeros(1, length(RT));
mns_hm = zeros(1, length(RT));
for i_RT = 1:length(RT)
    g = 10^(-3/fs/RT(i_RT));  % gain per sample 
    Gamma = diag(g.^delays);

    % make matrix A orthogonal 
    U = double(expm(skew(A*Gamma)));
    [residues, poles, ~, ~, ~] = dss2pr(delays, U, B, C, D);
    % h = histogram(db(abs(residues)),'BinWidth',1);
    [N, edges] = histcounts(db(abs(residues)),'BinWidth',1);
    plot(edges(1:end-1), N, 'LineWidth',2); hold on;
    stdev_hm(i_RT) = std(residues);
    mns_hm(i_RT) = mean(residues);
end
xlim([-200, -100])
% frequnecy dependent attenuation 
legend(string(RT))
title('Modal excitation at different RT')
xlabel('Residue Magnitude (dB)')
ylabel('Number of Occurence')

%%

% frequency dependent  attenatuion 
figure('Name', 'Frequency Dependent Decay')
stdev_fd = zeros(1, length(RT));
mns_fd = zeros(1, length(RT));
for i_RT = 1:length(RT)
    targetT60_mod = targetT60 .* 10.*abs(randn(size(targetT60)));
    zAbsorption = zSOS(absorptionGEQ(targetT60_mod, delays, fs),'isDiagonal',true);
    % make matrix A orthogonal 
    U = double(expm(skew(A)));
    [residues, poles, ~, ~, ~] = dss2pr(delays, U, B, C, D, 'absorptionFilters', zAbsorption);
    % h = histogram(db(abs(residues)),'BinWidth',1);
    [N, edges] = histcounts(db(abs(residues)),'BinWidth',1);
    plot(edges(1:end-1), N, 'LineWidth',2); hold on;
    stdev_fd(i_RT) = std(residues);
    mns_fd(i_RT) = mean(residues);
end
xlim([-200, -100])
% frequnecy dependent attenuation 
title('Modal excitation at different RT')
xlabel('Residue Magnitude (dB)')
ylabel('Number of Occurence')

%% functions 
function Y = skew(X)
    X = triu(X,1);
    Y = X - transpose(X);
end

