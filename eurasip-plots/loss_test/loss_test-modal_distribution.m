
clear all; close all; clc

addpath(genpath('../../fdnToolbox'))
set(groot, 'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');

addpath(genpath('utilities'))
rng(111)
%% 
results_dir = fullfile('output','siso-magnitude-loss');
delays = [1499., 1889., 2381., 2999.];
Gamma =  diag(0.9999.^delays);
res_stats = zeros(2, 100, sum(delays)); 
res_total_init = [];
res_total_optim = [];

for i = 0:99
    load(fullfile(results_dir, "test"+num2str(i), 'parameters_init.mat'))
    temp = B; clear B
    B = temp(:);        % input gains as column
    temp = C; clear C
    C = temp;
    D = zeros(1:1);     % direct gains
    % make matrix A orthogonal 
    temp = A; clear A
    A = double(expm(skew(temp))*Gamma);
    [tfB1, tfA1] = dss2tf(delays, A, B, C, D);
    [residues, ~, ~, ~, ~] = ...
            dss2pr(delays,A, B, C, D);
    res_stats(1, i+1, 1:length(residues)) = residues;
    res_total_init = [res_total_init; residues];
    load(fullfile(results_dir, "test"+num2str(i), 'parameters.mat'))
    temp = B; clear B
    B = temp(:);        % input gains as column
    temp = C; clear C
    C = temp;
    D = zeros(1:1);     % direct gains
    % make matrix A orthogonal 
    temp = A; clear A
    A = double(expm(skew(temp))*Gamma);
    [tfB2, tfA2] = dss2tf(delays, A, B, C, D);
    [residues, ~, ~, ~, ~] = ...
            dss2pr(delays,A, B, C, D);
    res_stats(2, i+1, 1:length(residues)) = residues;
    res_total_optim = [res_total_optim; residues];
end

std_init = std(db(res_total_init));
std_optim = std(db(res_total_optim));

save(fullfile(results_dir,'std_optim.mat'),'std_optim')
save(fullfile(results_dir,'std_init.mat'),'std_init')
save(fullfile(results_dir,'res_stats.mat'), 'res_stats')
save(fullfile(results_dir,'res_total_init.mat'),'res_total_init')
save(fullfile(results_dir,'res_total_optim.mat'),'res_total_optim')
%%  std_init

res_init=[];
res_optim=[];
for i=1:100
    res_init = [res_init, std(db(res_stats(1, i, 1:find(squeeze(res_stats(1, i, :)), 1, 'last'))))];
    res_optim = [res_optim, std(db(res_stats(2, i, 1:find(squeeze(res_stats(2, i, :)), 1, 'last'))))];
end

save(fullfile(results_dir,'res_optim.mat'),'res_optim')
save(fullfile(results_dir,'res_init.mat'),'res_init')
function Y = skew(X)
    X = triu(X,1);
    Y = X - transpose(X);
end