clear all; close all; clc

addpath(genpath('../fdnToolbox'))
addpath(genpath('utilities'))
set(groot, 'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');

rng(111)
results_dir = "../mushra/training-data/DFDN/N8D2B";
%% 
fs = 48000; 
irLen = 2*fs;
delays = [241.0, 263.0, 281.0, 293.0, 1871.0, 1973.0, 1999.0, 2027.0]; % [797.0, 839.0, 3547.0, 3581.0];
Gamma = diag(0.9999.^delays);
% construct FDN
N = length(delays); 
load(fullfile(results_dir,'parameters_init.mat'))
B = B(:);        % input gains as column
C = C(:)';
D = zeros(1:1);     % direct gains
A = double(expm(skew(A))*Gamma);
ir.('init') = dss2impz(...
    irLen, delays, A, B, C, D);
[residues.('init'), poles.('init'), ...
    direct.('init'), isConjugatePolePair.('init'), metaData.('init')] = ...
    dss2pr(delays,A, B, C, D);
[tfB.('init'), tfA.('init')] = dss2tf(delays, A*Gamma, B, C, D);

load(fullfile(results_dir,'parameters.mat'))
B = B(:);        % input gains as column
C = C(:)';
D = zeros(1:1);     % direct gains
A = double(expm(skew(A))*Gamma);
ir.('optim') = dss2impz(...
    irLen, delays, A, B, C, D);
[residues.('optim'), poles.('optim'), ...
    direct.('optim'), isConjugatePolePair.('optim'), metaData.('optim')] = ...
    dss2pr(delays,A, B, C, D);
[tfB.('optim'), tfA.('optim')] = dss2tf(delays, A*Gamma, B, C, D);
[h.('init'),w] = freqz(squeeze(tfB.('init')),squeeze(tfA.('init')),2^15, fs);
[h.('optim'),w] = freqz(squeeze(tfB.('optim')),squeeze(tfA.('optim')),2^15, fs);

% plotting
figure('Position', [0 0 1000 400]); 
hold on; grid on;
plot(w, db(abs(h.('init'))),'Color', [0,135,255]./255, 'LineWidth',2); 
plot(w, db(abs(h.('optim'))),'Color', [239,95,40]./255, 'LineWidth',2); 


%xlim([11500,  12500])
%xticks([11 11.5 11.75 12 12.25 12.5 13]*1000)
%xticklabels({'$11$', '$11.5$', '$11.75$','$12$', '$12.25$', '$12.5$', '$13$'})
ylim([-40, 60])
legend('init', 'optim')
xlabel('Frequency (kHz)')
ylabel('Magnitude (dB)')
ax=gca;
ax.FontSize = 24;
set(ax, 'box', 'on', 'Visible', 'on')

figure(2)
res = db(abs(residues.('init')));
res = res - mean(res);
histogram(res,'FaceAlpha',0.1,'BinWidth',1, 'FaceColor', [0,135,255]./255, 'EdgeColor', [0,135,255]./255); hold on 
res = db(abs(residues.('optim')));
res = res - mean(res);
histogram(res,'FaceAlpha',0.1,'BinWidth',1, 'FaceColor',  [239,95,40]./255, 'EdgeColor', [239,95,40]./255); hold on 

%plot(angle(poles.('init')), db(abs(residues.('init'))),'LineStyle','none','Marker','.')
%plot(angle(poles.('optim')), db(abs(residues.('optim'))),'LineStyle','none','Marker','.')

function Y = skew(X)
    X = triu(X,1);
    Y = X - transpose(X);
end