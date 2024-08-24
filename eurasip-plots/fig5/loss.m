clear all; close all; clc

addpath(genpath('../../fdnToolbox'))
addpath('training-results')

set(groot, 'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');

results_dir = 'training-results';
colors = {[239,95,40]/255, [83,157,255]/255,[0,199,87]/255, [120,120,120]./255};

%% alpha = 1

load(fullfile(results_dir, 'alpha1','train_loss.mat'))
load(fullfile(results_dir, 'alpha1','losses_partial.mat'))
load(fullfile(results_dir, 'alpha1','density.mat'))


% figure(1, 'Position', [0 0 1300 400]); 
figure(1)
ax1 = subplot(1, 2, 1);
set(ax1, 'box', 'on', 'Visible', 'on')

hold on
plot(train_loss,'LineWidth',2, "Color",colors{2}); grid on
plot(spectral, '-.','LineWidth',2, "Color",colors{2});grid on
plot(sparsity, '--', 'LineWidth',2, "Color",colors{2});grid on
xlabel('Epoch','FontSize',24);

grid on
ylabel('Loss')
ax=gca;
ax.FontSize = 24;
xlim([1, 10])
ylim([-1.25, 1.5])


%% alpha = 0

load(fullfile(results_dir, 'alpha0','train_loss.mat'))
load(fullfile(results_dir, 'alpha0','losses_partial.mat'))
load(fullfile(results_dir, 'alpha0','density.mat'))

figure(1);
subplot(1, 2, 1);
plot(train_loss,'LineWidth',2, "Color",colors{3}); grid on
legend('${\mathcal{L}}$', '${\mathcal{L}}_\textrm{spectral}$', '${\mathcal{L}}_\textrm{sparsity}$',  '${\mathcal{L}}_\textrm{spectral}^*$', 'FontSize',24, 'Location','southeast')

%% echo density
fs = 48000;
irLen = 2*fs;
delays = [809.0, 877.0, 937.0, 1049.0, 1151.0, 1249.0, 1373.0, 1499.0];
Gamma = diag(0.99999.^delays);
% load parameters
load(fullfile(results_dir, 'alpha0','parameters.mat'))
type = 'alpha0';
temp = B; clear B
B.(type) = temp(:);        % input gains as column
temp = C; clear C
C.(type) = temp;
D.(type) = zeros(1:1);     % direct gains
% make matrix A orthogonal 
temp = A; clear A
A.(type) = double(expm(skew(temp))*Gamma);
ir.(type) = dss2impz(...
    irLen, delays, A.(type), B.(type), C.(type), D.(type));
[t_abel.(type),echo_dens.(type)] = echoDensity(ir.(type), 1024, fs, 0); 

load(fullfile(results_dir, 'alpha1','parameters.mat'))
type = 'alpha1';
temp = B; clear B
B.(type) = temp(:);        % input gains as column
temp = C; clear C
C.(type) = temp;
D.(type) = zeros(1:1);     % direct gains
% make matrix A orthogonal 
temp = A; clear A
A.(type) = double(expm(skew(temp))*Gamma);
ir.(type) = dss2impz(...
    irLen, delays, A.(type), B.(type), C.(type), D.(type));
[t_abel.(type),echo_dens.(type)] = echoDensity(ir.(type), 1024, fs, 0); 

load(fullfile(results_dir, 'alpha1','parameters_init.mat'))
type = 'init';
temp = B; clear B
B.(type) = temp(:);        % input gains as column
temp = C; clear C
C.(type) = temp;
D.(type) = zeros(1:1);     % direct gains
% make matrix A orthogonal 
temp = A; clear A
A.(type) = double(expm(skew(temp))*Gamma);
ir.(type) = dss2impz(...
    irLen, delays, A.(type), B.(type), C.(type), D.(type));
[t_abel.(type),echo_dens.(type)] = echoDensity(ir.(type), 1024, fs, 0); 

time = linspace(0, irLen/fs, irLen); 
ax2 = subplot(1, 2, 2); hold on; grid on
set(ax2, 'box', 'on', 'Visible', 'on')
plot(time, echo_dens.('init'), "Color",colors{4},'LineWidth',2);
plot(time, echo_dens.('alpha0'), "Color",colors{3},'LineWidth',2);
plot(time, echo_dens.('alpha1'), "Color",colors{2},'LineWidth',2);

ylabel('Echo Density')
xlabel('Time (s)','FontSize',24);

ax=gca;
ax.FontSize = 24;

legend('Init', 'Optim ($\alpha=0$)', 'Optim ($\alpha=1$)', 'Location','southeast')
function Y = skew(X)
    X = triu(X,1);
    Y = X - transpose(X);
end