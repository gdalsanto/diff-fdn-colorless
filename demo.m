% Differentiable FDN for Colorless Reverberation 
% demo code 
%
% Dal Santo, Gloria, Karolina Prawda, Sebastian J. Schlecht, and Vesa Välimäki. 
% "Differentiable Feedback Delay Network for colorless reverberation." 
% International Conference on Digital Audio Effects (DAFx23), Copenhagen, 
% Denmark, Sept. 4-7 2023 

clear; clc; close all;

addpath(genpath('fdnToolbox'))
addpath(genpath('utilities'))
results_date = ['20230419-094709'];
results_dir = fullfile('output',results_date);
rng(13);
mkdir(fullfile(results_dir,'ir'))

% general parameters
fs = 48000;         % sampling frequency
fbin = fs * 10;     % number of frequency bin
irLen = fs*2;       % ir length   
types = {'DiffFDN','initDiffFDN','Hadamard','Householder','random'};


%% construct FDNs 
RT = 1.44*2;        % reverberation time (s)
g = 10^(-3/fs/RT);  % gain per sample   (linear)

delays = [809, 877, 937, 1049, 1151, 1249, 1373, 1499];
% attenuaton matrix
Gamma = diag(g.^delays);
N = length(delays); 

for typeCell = types
    type = typeCell{1};
    switch type
        case 'DiffFDN' 
            % load parameters
            load(fullfile(results_dir,'parameters.mat'))
            temp = B; clear B
            B.(type) = temp(:);        % input gains as column
            temp = C; clear C
            C.(type) = temp;
            D.(type) = zeros(1:1);     % direct gains
            % make matrix A orthogonal 
            temp = A; clear A
            A.(type) = double(expm(skew(temp))*Gamma);
        case 'initDiffFDN'
            % load parameters
            load(fullfile(results_dir,'parameters_init.mat'))
            temp = B; clear B
            B.(type) = temp(:);        % input gains as column
            temp = C; clear C
            C.(type) = temp;
            D.(type) = zeros(1:1);     % direct gains
            % make matrix A orthogonal 
            temp = A; clear A
            A.(type) = double(expm(skew(temp))*Gamma);
        case 'Hadamard'
            A.(type) = fdnMatrixGallery(double(N),'Hadamard');
            B.(type) = ones(N,1);
            C.(type) = ones(1,N);
            D.(type) = zeros(1,1);
            A.(type) = A.(type)*Gamma;
        case 'random'
            A.(type) = fdnMatrixGallery(N,'orthogonal');
            B.(type) = ones(N,1);
            C.(type) = ones(1,N);
            D.(type) = zeros(1,1);
            A.(type) = A.(type)*Gamma;
        case 'Householder'
            A.(type) = fdnMatrixGallery(double(N),'Householder');
            B.(type) = ones(N,1);
            C.(type) = ones(1,N);
            D.(type) = zeros(1,1);
            A.(type) = A.(type)*Gamma;
    end

    % generate impulse response
    ir.(type) = dss2impz(...
        irLen, delays, A.(type), B.(type), C.(type), D.(type));
    % modal decomposition
    [residues.(type), poles.(type), ...
        direct.(type), isConjugatePolePair.(type), metaData.(type)] = ...
        dss2pr(delays,A.(type), B.(type), C.(type), D.(type));
    % transfer function
    [tfB.(type), tfA.(type)] = dss2tf(delays, A.(type), B.(type), C.(type), D.(type));
    % save impulse response
    name = ['ir_','g',num2str(g),'_N',num2str(N),'_',type,'.wav'];
    filename = fullfile(results_dir,'ir',name);
    audiowrite(filename, ir.(type)/max(abs(ir.(type))),fs);
end

%% plots 
% mode excitation
figure(); hold on; grid on;
for typeCell = types
    type = typeCell{1};
    plot(angle(poles.(type)), db(abs(residues.(type))),'LineStyle','none','Marker','.')
end
legend(types)
xlim([0, pi]);
ylim([-80,-60]);
xlabel('Pole Frequency (rad)')
ylabel('Residue Magnitude (dB)')

% modal excitation histogram 
figure(); hold on; grid on;
for typeCell = types
    type = typeCell{1};
    res = db(abs(residues.(type)));
    % res = res - mean(res);
    histogram(res,'FaceAlpha',0.1,'BinWidth',1)
end
legend(types)
title('Modal excitation')
xlabel('Residue Magnitude (dB)')
ylabel('Number of Occurence')
