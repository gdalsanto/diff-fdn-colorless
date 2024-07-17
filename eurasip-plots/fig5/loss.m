clear all; close all; clc

addpath(genpath('../../fdnToolbox'))
set(groot, 'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');

results_dir = 'training-results';
%% 

load(fullfile(results_dir, 'losses_partial.mat'))
load(fullfile(results_dir, 'density.mat'))

colors = {[239,95,40]/255, [83,157,255]/255,[0,199,87]/255};

figure('Position', [0 0 1300 400]); 
ax1 = subplot(1, 2, 1);
set(ax1, 'box', 'on', 'Visible', 'on')

SparsityShift = 1+Sparsity;
SpectralShift = Spectral;
hold on
plot(SpectralShift+SparsityShift,'LineWidth',2, "Color",colors{2}); grid on
plot(SpectralShift, '-.','LineWidth',2, "Color",colors{2});grid on
plot(SparsityShift, '--', 'LineWidth',2, "Color",colors{2});grid on
xlabel('Epoch','FontSize',24);
legend('${\mathcal{L}}$', '${\mathcal{L}}_\textrm{spectral}$', '${\mathcal{L}}_\textrm{sparsity}$', 'FontSize',24)

grid on
ylabel('Loss')
ax=gca;
ax.FontSize = 24;
xlim([1, 15])
ylim([0, 2])

ax2 = subplot(1, 2, 2);
set(ax2, 'box', 'on', 'Visible', 'on')

plot(density/density(1),'LineWidth',2, "Color",colors{1});
grid on;
yticks([1 1.05 1.1 1.15 1.2])
yticklabels({'0','5','10','15','20'})

ylabel('Relative Density Increase ($\%$)')
xlabel('Epoch','FontSize',24);

ax=gca;
ax.FontSize = 24;
xlim([1, 15])
ylim([1, 1.20001])
