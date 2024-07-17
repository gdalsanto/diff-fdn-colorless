clear all; close all; clc

addpath(genpath('../../fdnToolbox'))
set(groot, 'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');
rng(2307)

%% FDN
delays = [809, 877, 937, 1049, 1151, 1249, 1373, 1499];
N = length(delays);
g = [1 0.9999 0.9990];
t60 = -60./db(g)./48000; 
B = ones(N,1);
C = ones(1,N);
D = zeros(1,1);
A = fdnMatrixGallery(N,'orthogonal');

% save FDN parameters
fdn = struct;
fdn.A = A; fdn.B = B; fdn.C = C; fdn.D = D; fdn.delays = delays; fdn.Gamma = diag(g(3).^delays);
save('fdn.mat', 'fdn')

tfB = zeros(length(g), sum(delays)+1);
tfA = zeros(length(g), sum(delays)+1);

figure('Name', 'TF at gamma', 'Position', [0 0 1000 400]);
grid on; hold on;
colors = {[0,199,87]/255, [0,135,255]/255, [239,95,40]/255,};

gOrder = [1, 2, 3]
for i = 1:length(gOrder)
    iG = gOrder(i);
    Gamma = diag(g(iG).^delays);
    Ag = A*Gamma;
    [b,a] = dss2tf(double(delays),Ag,B,C,D);
    tfB(iG,:) = squeeze(b);
    tfA(iG,:) = squeeze(a);
    [h, w] = freqz(tfB(iG,:), tfA(iG,:), 2^17);
    plot(w*1000, db(h),'Color',colors{iG},'LineWidth',2);
end

xlabel('Frequency (Hz)','FontSize',24);
ylabel('Magnitude (dB)','FontSize',24);
plots=get(gca, 'Children');
legend({"$T_{60}=\infty$", "$T_{60}=1.44$",  "$T_{60}=0.14$"}, 'FontSize',24)
ax=gca;
ax.FontSize = 24;
xlim([550 600]);
ylim([-15 45])
set(ax, 'box', 'on', 'Visible', 'on')
