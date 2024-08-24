clear all; close all; clc

addpath(genpath('../fdnToolbox'))
set(groot, 'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');

addpath(genpath('fdnToolbox'))
addpath(genpath('utilities'))
rng(111)

results_dir = 'training-results';

%% N=4
fs = 48000;
irLen = 2*fs;
delays =  [797.0, 839.0, 3547.0, 3581.0];
Gamma = diag(0.9998.^delays);
% load parameters
load(fullfile(results_dir, '4', 'parameters.mat'))
type = 'optim4';
temp = B; clear B
B.(type) = temp(:);        % input gains as column
temp = C; clear C
C.(type) = temp;
D.(type) = zeros(1:1);     % direct gains
% make matrix A orthogonal 
temp = A; clear A
A.(type) = double(expm(skew(temp))*Gamma);
[tfB1, tfA1] = dss2tf(delays, A.(type), B.(type), C.(type), D.(type));
[residues.(type), poles1.(type), ...
        direct.(type), isConjugatePolePair.(type), metaData.(type)] = ...
        dss2pr(delays,A.(type), B.(type), C.(type), D.(type));

load(fullfile(results_dir, '4','parameters_init.mat'))
type = 'init4';
temp = B; clear B
B.(type) = temp(:);        % input gains as column
temp = C; clear C
C.(type) = temp;
D.(type) = zeros(1:1);     % direct gains
% make matrix A orthogonal 
temp = A; clear A
A.(type) = double(expm(skew(temp))*Gamma);
[tfB2, tfA2] = dss2tf(delays, A.(type), B.(type), C.(type), D.(type));
[residues.(type), poles2.(type), ...
        direct.(type), isConjugatePolePair.(type), metaData.(type)] = ...
        dss2pr(delays,A.(type), B.(type), C.(type), D.(type));

% transfer function
[h1,w] = freqz(squeeze(tfB1),squeeze(tfA1),2^15, fs);
[h2,w] = freqz(squeeze(tfB2),squeeze(tfA2),2^15, fs);

%% N=6
fs = 48000;
irLen = 2*fs;
delays =  [887.0, 911.0, 941.0, 2017.0, 2053.0, 2129.0] ;66
Gamma = diag(0.9998.^delays);
% load parameters
load(fullfile(results_dir, '6','parameters.mat'))
type = 'optim6';
temp = B; clear B
B.(type) = temp(:);        % input gains as column
temp = C; clear C
C.(type) = temp;
D.(type) = zeros(1:1);     % direct gains
% make matrix A orthogonal 
temp = A; clear A
A.(type) = double(expm(skew(temp))*Gamma);
[tfB1, tfA1] = dss2tf(delays, A.(type), B.(type), C.(type), D.(type));
[residues.(type), poles1.(type), ...
        direct.(type), isConjugatePolePair.(type), metaData.(type)] = ...
        dss2pr(delays,A.(type), B.(type), C.(type), D.(type));

load(fullfile(results_dir, '6','parameters_init.mat'))
type = 'init6';
temp = B; clear B
B.(type) = temp(:);        % input gains as column
temp = C; clear C
C.(type) = temp;
D.(type) = zeros(1:1);     % direct gains
% make matrix A orthogonal 
temp = A; clear A
A.(type) = double(expm(skew(temp))*Gamma);
[tfB2, tfA2] = dss2tf(delays, A.(type), B.(type), C.(type), D.(type));
[residues.(type), poles2.(type), ...
        direct.(type), isConjugatePolePair.(type), metaData.(type)] = ...
        dss2pr(delays,A.(type), B.(type), C.(type), D.(type));

% transfer function
[h3,w] = freqz(squeeze(tfB1),squeeze(tfA1),2^15, fs);
[h4,w] = freqz(squeeze(tfB2),squeeze(tfA2),2^15, fs);

%% N=8

fs = 48000;
irLen = 2*fs;
delays = [241.0, 263.0, 281.0, 293.0, 1871.0, 1973.0, 1999.0, 2027.0];
Gamma = diag(0.9998.^delays);
% load parameters
load(fullfile(results_dir, '8', 'parameters.mat'))
type = 'optim8';
temp = B; clear B
B.(type) = temp(:);        % input gains as column
temp = C; clear C
C.(type) = temp;
D.(type) = zeros(1:1);     % direct gains
% make matrix A orthogonal 
temp = A; clear A
A.(type) = double(expm(skew(temp))*Gamma);
[tfB1, tfA1] = dss2tf(delays, A.(type), B.(type), C.(type), D.(type));
[residues.(type), poles2.(type), ...
        direct.(type), isConjugatePolePair.(type), metaData.(type)] = ...
        dss2pr(delays,A.(type), B.(type), C.(type), D.(type));

load(fullfile(results_dir, '8', 'parameters_init.mat'))
type = 'init8';
temp = B; clear B
B.(type) = temp(:);        % input gains as column
temp = C; clear C
C.(type) = temp;
D.(type) = zeros(1:1);     % direct gains
% make matrix A orthogonal 
temp = A; clear A
A.(type) = double(expm(skew(temp))*Gamma);
[tfB2, tfA2] = dss2tf(delays, A.(type), B.(type), C.(type), D.(type));
[residues.(type), poles2.(type), ...
        direct.(type), isConjugatePolePair.(type), metaData.(type)] = ...
        dss2pr(delays,A.(type), B.(type), C.(type), D.(type));

% transfer function
[h5,w] = freqz(squeeze(tfB1),squeeze(tfA1),2^15, fs);
[h6,w] = freqz(squeeze(tfB2),squeeze(tfA2),2^15, fs);

%% 
% plotting
xlimL = 10000;
xlimR = 12500;
figure('Position', [0 0 1000 400]); 
hold on; grid minor;
plot(w, db(abs(h2)),'Color', [120,120,120]./255, 'LineWidth',2); 
plot(w, -50+db(abs(h1)),'Color', [0,135,255]./255, 'LineWidth',2); 

txt = '$N=4$';
text(xlimL+(xlimR-xlimL)/2,20,txt, 'FontSize', 20, 'HorizontalAlignment','center')

plot(w, -120+db(abs(h4)),'Color', [120,120,120]./255, 'LineWidth',2); 
plot(w, -170+db(abs(h3)),'Color', [0,135,255]./255, 'LineWidth',2); 

txt = '$N=6$';
text(xlimL+(xlimR-xlimL)/2,-100,txt, 'FontSize', 20, 'HorizontalAlignment','center')

plot(w, -240+db(abs(h6)),'Color', [120,120,120]./255, 'LineWidth',2); 
plot(w, -290+db(abs(h5)),'Color', [0,135,255]./255, 'LineWidth',2); 

txt = '$N=8$';
text(xlimL+(xlimR-xlimL)/2,-220,txt, 'FontSize', 20, 'HorizontalAlignment','center')

xlim([xlimL,  xlimR])
xticks([10 10.25 10.5 10.75 11 11.25 11.5 11.75 12 12.25 12.5 13]*1000)
xticklabels({'$10$','$10.25$','$10.5$','$10.75$','$11$','$11.25$', '$11.5$', '$11.75$','$12$', '$12.25$', '$12.5$', '$13$'})
ylim([-340, 50])
set(gca,'yTickLabels',[])
%yticks([-250 -200 -150 -100 -50 0])
%yticklabels({'$0$', '$0$', '$0$', '$0$', '$0$', '$0$'})

legend('Optim', 'Init')
xlabel('Frequency (kHz)')
ylabel('Magnitude (dB)')
ax=gca;
ax.FontSize = 24;
set(ax, 'box', 'on', 'Visible', 'on')
%% modal decomposition 
types = {'init4','optim4','init6','optim6','init8','optim8'};
xx = [0 0 40 40 80 80];
% modal excitation histogram 
figure(); hold on; grid on;
i = 1;
colors = {[120,120,120]./255,  [0,135,255]./255, [120,120,120]./255,  [0,135,255]./255,[120,120,120]./255,  [0,135,255]./255}
for typeCell = types
    type = typeCell{1};
    res = db(abs(residues.(type)));
    res = res - mean(res) +xx(i);
    histogram(res,'FaceAlpha',0.2,'BinWidth',2,'FaceColor',colors{i},'Normalization', 'probability','Linewidth',1.5,'EdgeColor',colors{i})
    i = i+1;
end
txt = '$N=4$';
text(0,0.275,txt, 'FontSize', 20, 'HorizontalAlignment','center')

txt = '$N=6$';
text(40,0.275,txt, 'FontSize', 20, 'HorizontalAlignment','center')

txt = '$N=8$';
text(80,0.275,txt, 'FontSize', 20, 'HorizontalAlignment','center')


xticks([-10 0 10 20 30 40 50 60 70 80 90])
xticklabels({'$-10$','$0$','$10$', [], '$-10$','$0$','$10$', [], '$-10$', '$0$','$10$', []})
xlim([-30, 110])
legend('Init','Optim')
xlabel('Residue Magnitude (dB)')
ylabel('Relative probability')
ax=gca;
ax.FontSize = 24;
set(ax, 'box', 'on', 'Visible', 'on')

function Y = skew(X)
    X = triu(X,1);
    Y = X - transpose(X);
end