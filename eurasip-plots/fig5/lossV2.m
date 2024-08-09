clear all; close all; clc

addpath(genpath('../../fdnToolbox'))
addpath('training-results')

set(groot, 'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');

results_dir = 'training-results';
colors = {[239,95,40]/255, [83,157,255]/255,[0,199,87]/255};

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
%xlim([1, 15])
%ylim([0, 2])

ax2 = subplot(1, 2, 2); hold on;
set(ax2, 'box', 'on', 'Visible', 'on')

plot(density/density(1),'LineWidth',2, "Color",colors{1});
grid on;
yticks([1 1.05 1.1 1.15 1.2])
yticklabels({'0','5','10','15','20'})

ylabel('Relative Density Increase ($\%$)')
xlabel('Epoch','FontSize',24);

ax=gca;
ax.FontSize = 24;
%xlim([1, 15])
%ylim([1, 1.20001])

%% alpha = 0

load(fullfile(results_dir, 'alpha0','train_loss.mat'))
load(fullfile(results_dir, 'alpha0','losses_partial.mat'))
load(fullfile(results_dir, 'alpha0','density.mat'))

figure(1);
subplot(1, 2, 1);
plot(train_loss,'LineWidth',2, "Color",colors{3}); grid on
legend('${\mathcal{L}}$', '${\mathcal{L}}_\textrm{spectral}$', '${\mathcal{L}}_\textrm{sparsity}$',  '${\mathcal{L}}^*$', 'FontSize',24)

subplot(1, 2, 2); 
plot(density/density(1),'LineWidth',2, "Color",colors{3});

%% echo density
fs = 48000;
irLen = 2*fs;
delays = [809, 877, 937, 1049, 1151, 1249, 1373, 1499];
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


figure(2); hold on;
plot(echo_dens.('init'));
plot(echo_dens.('alpha0'));
plot(echo_dens.('alpha1'));
function Y = skew(X)
    X = triu(X,1);
    Y = X - transpose(X);
end