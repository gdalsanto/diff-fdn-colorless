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
types = {'DFDN','random','allpass','Hadamard','Householder','Schroeder'};
types = {'random'};
% types = {'allpass', 'random'};

%% construct FDNs 
RT = 1.44*2;      % reverberation time
g = 10^(-3/fs/RT); % g = 1; 
% g = 1;
delays_N1 = {[1499, 1889, 2381, 2999], ...
            [997., 1153., 1327., 1559., 1801., 2099.], ...
            [809, 877, 937, 1049, 1151, 1249, 1373, 1499], ...
            [241, 271, 293, 317, 359, 389, 433, 467, 523, 571, 619, 683, 757, 829, 911, 997]};
delays_N2 = {[797., 839., 2381., 2999.], ...
            [887, 911, 941, 1699, 1951, 2053], ...
            [739, 757, 761, 773, 1103, 1249, 1307, 1459], ...
            [241., 263., 281., 293., 1193., 1319., 1453., 1597.], ...
            [839, 859, 863, 877, 1103, 1249, 1307, 1459], ...
            [241, 271, 293, 317, 359, 389, 433, 467, 523, 571, 619, 683, 757, 829, 911, 997]};
delays = delays_N1{2};
N = length(delays); 

addFilter = false;
addDelays = false; 

% extraDelayIn = [149., 181., 233., 293.]';
% extraDelayOut = [223., 271., 349., 439.]; 
extraDelayIn = [97., 113., 131., 151., 179., 199.]'; 
extraDelayOut = [139., 167., 193., 223., 263., 293.];
% % absorption filters
% centerFrequencies = [ 63, 125, 250, 500, 1000, 2000, 4000, 8000]; % Hz
% T60frequency = [1, centerFrequencies fs];
% targetT60 = [2; 2; 2.2; 2.3; 2.1; 1.5; 1.1; 0.8; 0.7; 0.7];  % seconds

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
            mm = m * std(delays) + mean(delays);
            delays = double(round(mm));  % delay line lengths 
            % attenuaton matrix
            Gamma = diag(g.^delays);
            % make matrix A orthogonal 
            temp = A; clear A
            A.(type) = double(expm(skew(temp))*Gamma);
        case 'random_DFDN'
            % ugly quick fix
            tempB = B; tempC = C; tempD = D; tempA = A; 

            load(fullfile(results_dir,'parameters_init.mat'))
            delays = double(round(m));  % delay line lengths 
            % attenuaton matrix
            Gamma = diag(g.^delays);
            % make matrix A orthogonal 
            temp = A; clear A
            Ainit = double(expm(skew(temp))*Gamma);  

            B = tempB; C = tempC; D = tempD; A = tempA;             
            B.(type) = ones(N,1);
            C.(type) = ones(1,N);
            D.(type) = zeros(1,1);
            A.(type) = Ainit; 
        case 'DFDNhouseholder' 
            % load parameters
            load(fullfile(results_dir,'parameters.mat'))
            temp = B; clear B
            B.(type) = temp(:);        % input gains as column
            temp = C; clear C
            C.(type) = temp;
            D.(type) = zeros(1:1);     % direct gains
            delays = double(round(m));  % delay line lengths 
            % attenuaton matrix
            Gamma = diag(g.^delays);
            % make matrix A orthogonal 
            temp = double(v/norm(v)); clear v
            A.(type) = (eye(N,N) - 2*temp.'.*temp)*Gamma;  
        case 'nn_init' 
            % load parameters
            temp_struct = A;
            load(fullfile(results_dir,'parameters_init.mat'))
            temp = B; clear B
            B.(type) = temp(:);        % input gains as column
            temp = C; clear C
            C.(type) = temp;
            D.(type) = zeros(1:1);     % direct gains
            delays = double(round(m));  % delay line lengths 
            % attenuaton matrix
            Gamma = diag(g.^delays);
            % make matrix A orthogonal 
            temp = A; clear A
            A = temp_struct;
            A.(type) = double(expm(skew(temp))*Gamma);   
        case 'allpass'
            g_ap = g;
            if g == 1
                % function cannot produce fdn with g=1
                g_ap = g-1e-4;
            end
            G = diag(g_ap.^delays ); % gain matrix
            X = randAdmissibleHomogeneousAllpass(G, [0.8, 0.99]); % diagonal similarity
            [A.(type), B.(type), C.(type), D.(type), U] = ...
                homogeneousAllpassFDN(G, X,'verbose',true); % create allpass FDN
            if g == 1
                A.(type) = U; % remove absorption (not allpass anymore)
            end
            D.(type) = 0
        case 'Hadamard'
            A.(type) = fdnMatrixGallery(double(N),'Hadamard');
            B.(type) = ones(N,1);
            C.(type) = ones(1,N);
            D.(type) = zeros(1,1);
            Gamma = diag(g.^delays);
            A.(type) = A.(type)*Gamma;
        case 'random'
            A.(type) = fdnMatrixGallery(N,'orthogonal');
            B.(type) = ones(N,1);
            C.(type) = ones(1,N);
            D.(type) = zeros(1,1);
            Gamma = diag(g.^delays);
            A.(type) = A.(type)*Gamma;
        case 'Householder'
            A.(type) = fdnMatrixGallery(double(N),'Householder');
            B.(type) = ones(N,1);
            C.(type) = ones(1,N);
            D.(type) = zeros(1,1);
            Gamma = diag(g.^delays);
            A.(type) = A.(type)*Gamma;
    end

    if addFilter
        % absorption filter 
        zAbsorption = zSOS(absorptionGEQ(targetT60, delays, fs),'isDiagonal',true);
        
        % power correction filter
        targetPower = [5; 5; 5; 3; 2; 1; -1; -3; -5; -5];  % dB
        powerCorrectionSOS = designGEQ(targetPower);
        C.(type) = zSOS(permute(powerCorrectionSOS,[3 4 1 2]) .* C.(type));
         % generate impulse response
        ir.(type) = dss2impz(...
            irLen, delays, A.(type), B.(type), C.(type), D.(type),...
            'absorptionFilters', zAbsorption);
        % compute the residues
        [residues.(type), poles.(type), ...
            direct.(type), isConjugatePolePair.(type), metaData.(type)] = ...
            dss2pr(delays,A.(type), B.(type), C.(type), D.(type),...
            'absorptionFilters', zAbsorption); 
    elseif addDelays
        matrixDelays = 1 + extraDelayIn + extraDelayOut;
        extendedDelays = delays+extraDelayIn'+extraDelayOut;
        % generate impulse response
        ir.(type) = dss2impz(irLen + max(matrixDelays(:)), extendedDelays, A.(type), diag(B.(type)), diag(C.(type)), 0);
        % shift input and outputs by the extra delays
        ir.(type) = mcircshift(ir.(type),-matrixDelays.'+1);
        ir.(type) = sum(ir.(type),[2 3]);
        temp = ir.(type); 
        ir.(type) = temp(1:irLen,:,:); % shorten the response to the same lengts
        % compute the residues
        [residues.(type), poles.(type), ...
            direct.(type), isConjugatePolePair.(type), metaData.(type)] = ...
            dss2pr(extendedDelays, A.(type), B.(type), C.(type), D.(type));
        [tfB.(type), tfA.(type)] = dss2tf(extendedDelays, A.(type), B.(type), C.(type), D.(type));
    else
        % generate impulse response
        ir.(type) = dss2impz(...
            irLen, delays, A.(type), B.(type), C.(type), D.(type));
        % compute the residues
        [residues.(type), poles.(type), ...
            direct.(type), isConjugatePolePair.(type), metaData.(type)] = ...
            dss2pr(delays,A.(type), B.(type), C.(type), D.(type));
        [tfB.(type), tfA.(type)] = dss2tf(delays, A.(type), B.(type), C.(type), D.(type));
    end
    name = ['ir_','g',num2str(g),'_N',num2str(N),'_filt',num2str(int8(addFilter)),'_diff',num2str(int8(addDelays)),'_',type,'.wav'];
    filename = fullfile(results_dir,'ir',name);
    audiowrite(filename, ir.(type)/max(abs(ir.(type))),fs);
end

% %% listening test 
% t_pn = 0.3;
% burst = pinknoise(floor(t_pn*fs));
% % burst = burst/max(abs(burst));
% for typeCell = types
%     type = typeCell{1};
%     ir_burst.(type) = conv(burst,ir.(type)); 
%     name = ['ir_','g',num2str(g),'_N',num2str(N),'_filt',num2str(int8(addFilter)),'_',type,'_burst.wav'];
%     filename = fullfile(results_dir,'ir',name);
%     audiowrite(filename, ir_burst.(type)/max(abs(ir_burst.(type))),fs);
% end
% 
% % reference 
% wgn = randn(1,irLen);
% t = (1:1:irLen)/fs;
% alpha = -log(10^(-3))/RT;
% expDecay = exp(-t*alpha);
% ir.('wgn') = wgn.*expDecay;
% name = 'reference.wav';
% filename = fullfile(results_dir,'ir',name);
% audiowrite(filename, ir.('wgn')/max(abs(ir.('wgn'))),fs);
% 

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
%% 
% mode RT
% TODO: check correctess when g = 1
% figure(); hold on; grid on; 
% for typeCell = types
%     type = typeCell{1};
%     rt_poles = -60/fs/20./log10(abs(poles.(type)));
%     plot(angle(poles.(type)), rt_poles,'LineStyle','none','Marker','.')
% end
% plot(2*pi*T60frequency./fs, targetT60,'o','Color','k')
% legend(types)
% xlim([0 pi])
% xlabel('Pole Frequency (rad)')
% ylabel('Pole Reverberation Time (dB)')

%% objective evaluation
% histogram of frequency response bins
figure(); hold on; grid on; 
for typeCell = types
    type = typeCell{1};
    [h,w] = freqz(ir.(type),1,2^12);
    H = db(abs(h));
    H = H - mean(H);
    histogram(db(abs(H)),'FaceAlpha',0.1);
end
legend(types)
title('Magnitude response')
xlabel('Magnitude value (dB)')
ylabel('Number of Occurence')

% histogram of STFT bins 
figure(); hold on; grid on;
for typeCell = types 
    type = typeCell{1};
    [S,f,t] = stft(ir.(type), fs);
    histogram(db(abs(S)),'FaceAlpha',0.1)
end
legend(types)
title('STFT')
xlabel('Magnitude value (dB)')
ylabel('Number of Occurence')

% modal excitation histogram 
figure(); hold on; grid on;
for typeCell = types
    type = typeCell{1};
    histogram(db(abs(residues.(type))),'FaceAlpha',0.1,'BinWidth',1)
end
legend(types)
title('Modal excitation')
xlabel('Residue Magnitude (dB)')
ylabel('Number of Occurence')
% comparison between Gaussian noise 


%% Paper plots 
% set latex as default interpreter
set(groot, 'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');

typesB = {'DFDN','allpass'};
binWidth = 1;
% mode excitation
figure(); hold on; grid on;
for typeCell = types
    type = typeCell{1};
    res = db(abs(residues.(type))); 
    res = res - mean(res);
    [N, edges] = histcounts(res,'BinWidth',binWidth);
    pdf.(type) = (N/sum(N))*binWidth;
    centerBin = edges+binWidth/2;
    centerBin(end) = [];
    plot(centerBin,pdf.(type),'LineWidth',1);
end
set(gca,'Fontsize',12)
ylim([0, 0.5])
xlim([-30, 30])
legend(types)
xlabel('Magnitude [dB]')
ylabel('Probability density')