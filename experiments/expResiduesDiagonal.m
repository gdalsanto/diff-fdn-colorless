% analysis of the distribution of the residues for the case of a diagonal
% feedback matrix 
%
% created 26.04.23 

clear; clc; close all;

addpath(genpath(fullfile('..','fdnToolbox')))
addpath(genpath(fullfile('..','utilities')))
rng(13);

% general parameters
fs = 48000;         % sampling frequency
fbin = fs * 10;     % number of frequency bin
irLen = fs*2;       % ir length   
types = {'random','diagonal'};


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
        case 'diagonal'
            A.(type) = diag(randn(N,1));
            B.(type) = randn(N,1);
            C.(type) = randn(1,N);
            D.(type) = zeros(1,1);
            A.(type) = A.(type)*Gamma;
    end

    % generate impulse response
    ir.(type) = dss2impz(...
        irLen, delays, A.(type), B.(type), C.(type), D.(type));
    % modal decomposition and matrix of the residues 
    [residues.(type), poles.(type), ...
        direct.(type), isConjugatePolePair.(type), metaData.(type), matrixRes.(type)] = ...
        dss2pr_resMatrix(delays,A.(type), B.(type), C.(type), D.(type));
    % transfer function
    [tfB.(type), tfA.(type)] = dss2tf(delays, A.(type), B.(type), C.(type), D.(type));
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
    res = res - mean(res);
    histogram(res,'FaceAlpha',0.1,'BinWidth',1)
end
legend(types)
title('Modal excitation')
xlabel('Residue Magnitude (dB)')
ylabel('Number of Occurence')


% plot matrix of resides 
for typeCell = types
    type = typeCell{1};
    figure;
    indx = 1; 
    for i = 1 : N 
        for j = 1 : N
            temp = matrixRes.(type);
            subplot(N, N, indx); 
            histogram(db(temp(:, i, j)))
            indx = indx + 1; 
        end
    end
end