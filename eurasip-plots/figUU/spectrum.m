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
Gamma = diag(0.9999.^delays);
% load parameters
load(fullfile(results_dir, '4', 'parameters.mat'))
type = 'optim';
temp = B; clear B
B.(type) = temp(:);        % input gains as column
temp = C; clear C
C.(type) = temp;
D.(type) = zeros(1:1);     % direct gains
% make matrix A orthogonal 
temp = A; clear A
A.(type) = double(expm(skew(temp))*Gamma);
[tfB1, tfA1] = dss2tf(delays, A.(type), B.(type), C.(type), D.(type));


load(fullfile(results_dir, '4','parameters_init.mat'))
type = 'init';
temp = B; clear B
B.(type) = temp(:);        % input gains as column
temp = C; clear C
C.(type) = temp;
D.(type) = zeros(1:1);     % direct gains
% make matrix A orthogonal 
temp = A; clear A
A.(type) = double(expm(skew(temp))*Gamma);
[tfB2, tfA2] = dss2tf(delays, A.(type), B.(type), C.(type), D.(type));


% transfer function
[h1,w] = freqz(squeeze(tfB1),squeeze(tfA1),2^15, fs);
[h2,w] = freqz(squeeze(tfB2),squeeze(tfA2),2^15, fs);

%% N=6
fs = 48000;
irLen = 2*fs;
delays =  [887.0, 911.0, 941.0, 2017.0, 2053.0, 2129.0] ;66
Gamma = diag(0.9999.^delays);
% load parameters
load(fullfile(results_dir, '6','parameters.mat'))
type = 'optim';
temp = B; clear B
B.(type) = temp(:);        % input gains as column
temp = C; clear C
C.(type) = temp;
D.(type) = zeros(1:1);     % direct gains
% make matrix A orthogonal 
temp = A; clear A
A.(type) = double(expm(skew(temp))*Gamma);
[tfB1, tfA1] = dss2tf(delays, A.(type), B.(type), C.(type), D.(type));


load(fullfile(results_dir, '6','parameters_init.mat'))
type = 'init';
temp = B; clear B
B.(type) = temp(:);        % input gains as column
temp = C; clear C
C.(type) = temp;
D.(type) = zeros(1:1);     % direct gains
% make matrix A orthogonal 
temp = A; clear A
A.(type) = double(expm(skew(temp))*Gamma);
[tfB2, tfA2] = dss2tf(delays, A.(type), B.(type), C.(type), D.(type));


% transfer function
[h3,w] = freqz(squeeze(tfB1),squeeze(tfA1),2^15, fs);
[h4,w] = freqz(squeeze(tfB2),squeeze(tfA2),2^15, fs);

%% N=8

fs = 48000;
irLen = 2*fs;
delays = [241.0, 263.0, 281.0, 293.0, 1871.0, 1973.0, 1999.0, 2027.0];
Gamma = diag(0.9999.^delays);
% load parameters
load(fullfile(results_dir, '8', 'parameters.mat'))
type = 'optim';
temp = B; clear B
B.(type) = temp(:);        % input gains as column
temp = C; clear C
C.(type) = temp;
D.(type) = zeros(1:1);     % direct gains
% make matrix A orthogonal 
temp = A; clear A
A.(type) = double(expm(skew(temp))*Gamma);
[tfB1, tfA1] = dss2tf(delays, A.(type), B.(type), C.(type), D.(type));


load(fullfile(results_dir, '8', 'parameters_init.mat'))
type = 'init';
temp = B; clear B
B.(type) = temp(:);        % input gains as column
temp = C; clear C
C.(type) = temp;
D.(type) = zeros(1:1);     % direct gains
% make matrix A orthogonal 
temp = A; clear A
A.(type) = double(expm(skew(temp))*Gamma);
[tfB2, tfA2] = dss2tf(delays, A.(type), B.(type), C.(type), D.(type));


% transfer function
[h5,w] = freqz(squeeze(tfB1),squeeze(tfA1),2^15, fs);
[h6,w] = freqz(squeeze(tfB2),squeeze(tfA2),2^15, fs);

%% 
% plotting
xlimL = 10000;
xlimR = 12500;
figure('Position', [0 0 1000 400]); 
hold on; grid on;
plot(w, db(abs(h1)),'Color', [0,135,255]./255, 'LineWidth',2); 
plot(w, db(abs(h2)),'Color', [239,95,40, 200]./255, 'LineWidth',2); 

txt = '$N=4$';
text(xlimL+(xlimR-xlimL)/2,15,txt, 'FontSize', 20, 'HorizontalAlignment','center')

plot(w, -55+db(abs(h3)),'Color', [0,135,255]./255, 'LineWidth',2); 
plot(w, -55+db(abs(h4)),'Color', [239,95,40, 200]./255, 'LineWidth',2); 

txt = '$N=6$';
text(xlimL+(xlimR-xlimL)/2,-40,txt, 'FontSize', 20, 'HorizontalAlignment','center')

plot(w, -110+db(abs(h5)),'Color', [0,135,255]./255, 'LineWidth',2); 
plot(w, -110+db(abs(h6)),'Color', [239,95,40, 200]./255, 'LineWidth',2); 

txt = '$N=8$';
text(xlimL+(xlimR-xlimL)/2,-90,txt, 'FontSize', 20, 'HorizontalAlignment','center')

xlim([xlimL,  xlimR])
xticks([10 10.5 11 11.5 11.75 12 12.25 12.5 13]*1000)
xticklabels({'$10$','$10.5$','$11$', '$11.5$', '$11.75$','$12$', '$12.25$', '$12.5$', '$13$'})
ylim([-150, 30])
yticks([-110 -55 0])
yticklabels({'$0$', '$0$', '$0$'})

legend('Optim', 'Init')
xlabel('Frequency (kHz)')
ylabel('Magnitude (dB)')
ax=gca;
ax.FontSize = 24;
set(ax, 'box', 'on', 'Visible', 'on')

function Y = skew(X)
    X = triu(X,1);
    Y = X - transpose(X);
end