% Differentiable FDN for Colorless Reverberation 
% demo code 

clear; clc; close all;

addpath(genpath('fdnToolbox'))

results_dir = "./output/20240828-154659";
output_dir = fullfile(results_dir, 'matlab');

%% Analyse reference RIR 

delays = [997., 1153., 1327., 1559., 1801., 2099.];
N = length(delays);
fs = 48000; 
irLen = 3*fs;
if isfile(fullfile(results_dir, 'synthesis_filters.mat'))
    load(fullfile(results_dir, 'synthesis_filters.mat'))
    zAbsorption = zSOS(G_SOS,'isDiagonal',true);
    equalizationSOS = TC_SOS;
else
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

    % Unless you have the M2 chio, you should be able to get the EDR
    % parameters from the DecayFitNet Toolbox. In this code I'm just
    % uploading a presetimated values 
    % net = DecayFitNetToolbox(nSlopes, fs, fBands(cInd));
    % est.T, est.A, est.N, est.norm] = net.estimateParameters(rir);
    load(fullfile('rirs', filename + "_DecayFitNet_est.mat"))
    est.T = double(T);  est.A = double(A); est.N = double(N); est.norm = double(norm); 
    clear norm A T N
    est = transposeAllFields(est);
    [est.L, est.A, est.N] = decayFitNet2InitialLevel(est.T, est.A, est.N, est.norm, fs, rirLen, fBands);

    % absorption filters, shorten top and bottom band
    T60frequency = [1, fBands fs];
    targetT60 = est.T([1 1:end end]);  % seconds
    targetT60 = targetT60 .* [0.9 1 1 1 1 1 1 1 0.9 0.5];
    zAbsorption = zSOS(absorptionGEQ(targetT60, delays, fs),'isDiagonal',true);
    
    % initial level filter, attenuate top and bottom band
    targetLevel = mag2db(est.L([1 1:end end]));  % dB
    targetLevel = targetLevel - [5 0 0 0 0 0 0 0 5 30];
    
    equalizationSOS = designGEQ(targetLevel);
end

%% construct FDN

if isfile(fullfile(results_dir, 'scat_parameters.mat'))
    types = {'SCAT', 'SCAT_INIT'};
else
    types = {'DFDN','RO'};
end

if ~exist(output_dir, 'dir')
    mkdir(output_dir)
end 
for typeCell = types
    type = typeCell{1};
    switch type
        case 'DFDN' 
            % load parameters
            load(fullfile(results_dir,'parameters.mat'))
            temp = B; clear B
            B.(type) = temp(:);        % input gains as column
            temp = C; clear C
            C.(type) = temp;
            D.(type) = zeros(1:1);     % direct gains
            % make matrix A orthogonal 
            temp = A; clear A
            A.(type) = double(expm(skew(temp)));
            C.(type) = zSOS(permute(equalizationSOS,[3 4 1 2]) .*  C.(type));
            ir.(type) = dss2impz(...
                irLen, delays, A.(type), B.(type), C.(type), D.(type), 'absorptionFilters', zAbsorption);
        case 'RO'
            % ugly quick fix
            tempB = B; tempC = C; tempD = D; tempA = A; 
            load(fullfile(results_dir,'parameters_init.mat'))
            % make matrix A orthogonal 
            temp = A; clear A
            Ainit = double(expm(skew(temp)));  
            temp = B; clear B 
            B = tempB; 
            B.(type) = temp';
            temp = C; clear C
            C = tempC; 
            C.(type) = temp;
            D = tempD; 
            D.(type) = zeros(1,1);
            A = tempA;
            A.(type) = Ainit; 
            C.(type) = zSOS(permute(equalizationSOS,[3 4 1 2]) .*  C.(type));
            ir.(type) = dss2impz(...
                irLen, delays, A.(type), B.(type), C.(type), D.(type), 'absorptionFilters', zAbsorption);
        case 'SCAT' 
            tempDelays = delays;
            load(fullfile(results_dir, 'scat_parameters.mat'))
            Adl = size(feedbackMatrix, 3);
            delayLeft = double(delayLeft);
            delayRight = double(delayRight);
            inputGain = inputGain(:);
            outputGain = outputGain(:)';
            outputGain = zSOS(permute(equalizationSOS,[3 4 1 2]) .*  outputGain);
            feedbackMatrix = shiftMatrix(double(feedbackMatrix), delayLeft, 'left');
            feedbackMatrix = shiftMatrix(feedbackMatrix, delayRight, 'right');
            mainDelay = double(delays - delayLeft - delayRight);
            matrixFilter = zFIR(double(feedbackMatrix));
            ir.(type) = dss2impzTransposed(irLen, mainDelay, matrixFilter, inputGain, outputGain, zeros(1,1), 'absorptionFilters', zAbsorption);
            delays = tempDelays;
        case 'SCAT_INIT' 
            tempDelays = delays;
            load(fullfile(results_dir, 'scat_parameters_init.mat'))
            Adl = size(feedbackMatrix, 3);
            delayLeft = double(delayLeft);
            delayRight = double(delayRight);
            inputGain = inputGain(:);
            outputGain = outputGain(:)';
            outputGain = zSOS(permute(equalizationSOS,[3 4 1 2]) .*  outputGain);
            feedbackMatrix = shiftMatrix(double(feedbackMatrix), delayLeft, 'left');
            feedbackMatrix = shiftMatrix(feedbackMatrix, delayRight, 'right');
            mainDelay = double(delays - delayLeft - delayRight);
            matrixFilter = zFIR(double(feedbackMatrix));
            ir.(type) = dss2impzTransposed(irLen, mainDelay, matrixFilter, inputGain, outputGain, zeros(1,1), 'absorptionFilters', zAbsorption);
            delays = tempDelays;
    end

    % generate impulse response

    % save impulse response
    name = ['ir_TR_',type,'.wav'];
    filename = fullfile(output_dir,name);
    audiowrite(filename, ir.(type)/max(abs(ir.(type))),fs);
end

%% functions 
function Y = skew(X)
    X = triu(X,1);
    Y = X - transpose(X);
end

