clear all; close all; clc

addpath(genpath('../fdnToolbox'))
set(groot, 'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');

addpath(genpath('fdnToolbox'))
addpath(genpath('utilities'))
rng(111)

%% 

% construct FDN
N = 8; 
B = ones(N,1);
C = ones(1,N);
D = zeros(1,1);
A = diag([1, -1, 1, -1, 1, -1, 1, -1]);
m1 = randi([500,2000],[1,N]);   
m2 = floor(m1/10);
g = 0.9999;
Gamma1 = diag(g.^m1);
Gamma2 = diag(g.^m2);
fs = 48000; 

% transfer function
[tfB1, tfA1] = dss2tf(m1, A*Gamma1, B, C, D);
[tfB2, tfA2] = dss2tf(m2, A*Gamma2, B, C, D);
[h1,w] = freqz(squeeze(tfB1),squeeze(tfA1),2^15, fs);
[h2,w] = freqz(squeeze(tfB2),squeeze(tfA2),2^15, fs);

% plotting
figure('Position', [0 0 1000 400]); 
hold on; grid on;
plot(w, db(abs(h1)),'Color', [0,135,255]./255, 'LineWidth',2); 
plot(w, db(abs(h2)),'Color', [239,95,40]./255, 'LineWidth',2); 

xlim([11500,  12500])
xticks([11 11.5 11.75 12 12.25 12.5 13]*1000)
xticklabels({'$11$', '$11.5$', '$11.75$','$12$', '$12.25$', '$12.5$', '$13$'})
ylim([-40, 60])
legend('Long Delays', 'Short Delays')
xlabel('Frequency (kHz)')
ylabel('Magnitude (dB)')
ax=gca;
ax.FontSize = 24;
set(ax, 'box', 'on', 'Visible', 'on')