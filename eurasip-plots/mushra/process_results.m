close all; clear all; clc;
resultsFolder = '.';
filename = 'mushra.csv';

% set latex as default interpreter
set(groot, 'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');

tbl = readtable(fullfile(resultsFolder,filename));

%% Preprocess
% remove trail 
toDelete = strcmpi(tbl.trial_id,"trial");
tbl(toDelete,:) = [];
toDelete = strcmpi(tbl.trial_id,"trial2");
tbl(toDelete,:) = [];
toDelete = strcmpi(tbl.trial_id,"trial3");
tbl(toDelete,:) = [];

% find instances where the score given to the hidden reference is below a
% threshold 

hidRef = strcmpi(tbl.rating_stimulus,"C1");
hidRefVal = tbl(hidRef,:).rating_score;
outliers = tbl((tbl.rating_score < 90) & hidRef,:); % OLD : tbl((tbl.rating_score < 90) & hidRef,:);
% an assessor should be excluded from the aggregated responses if he or she 
% rates the hidden reference condition for > 15% of the test items lower 
% than a score of 90
outliers_uuid = unique(outliers.session_uuid);
% remove the results from those outlier pages 
toDelete = zeros(height(tbl), 1);
n_outliers = 0;
for i = 1:height(outliers_uuid)
     if sum(strcmp(outliers.session_uuid, outliers_uuid{i})) > height(tbl)/14/6*0.15
         toDelete = toDelete | (strcmpi(tbl.session_uuid, outliers_uuid{i})); %  & strcmpi(tbl.trial_id, {outliers.trial_id{i}}));
         n_outliers = n_outliers + 1; 
     end
%    OLD : toDelete = toDelete | (strcmpi(tbl.session_uuid, outliers.session_uuid{i}) & strcmpi(tbl.trial_id, {outliers.trial_id{i}}));
end
tbl(toDelete,:) = [];


% substitude train id
toSubstitude = strcmpi(tbl.trial_id,'N4D1A_LL') | strcmpi(tbl.trial_id,'N4D1B_LL');
tbl(toSubstitude,"trial_id") = {'N4D1_LL'};
toSubstitude = strcmpi(tbl.trial_id,'N6D1A_LL') | strcmpi(tbl.trial_id,'N6D1B_LL');
tbl(toSubstitude,"trial_id") = {'N6D1_LL'};
toSubstitude = strcmpi(tbl.trial_id,'N8D1A_LL') | strcmpi(tbl.trial_id,'N8D1B_LL');
tbl(toSubstitude,"trial_id") = {'N8D1_LL'};

toSubstitude = strcmpi(tbl.trial_id,'N4D1A_HL') | strcmpi(tbl.trial_id,'N4D1B_HL');
tbl(toSubstitude,"trial_id") = {'N4D1_HL'};
toSubstitude = strcmpi(tbl.trial_id,'N6D1A_HL') | strcmpi(tbl.trial_id,'N6D1B_HL');
tbl(toSubstitude,"trial_id") = {'N6D1_HL'};
toSubstitude = strcmpi(tbl.trial_id,'N8D1A_HL') | strcmpi(tbl.trial_id,'N8D1B_HL');
tbl(toSubstitude,"trial_id") = {'N8D1_HL'};

toSubstitude = strcmpi(tbl.trial_id,'N4D1A_LY') | strcmpi(tbl.trial_id,'N4D1B_LY');
tbl(toSubstitude,"trial_id") = {'N4D1_LY'};
toSubstitude = strcmpi(tbl.trial_id,'N6D1A_LY') | strcmpi(tbl.trial_id,'N6D1B_LY');
tbl(toSubstitude,"trial_id") = {'N6D1_LY'};
toSubstitude = strcmpi(tbl.trial_id,'N8D1A_LY') | strcmpi(tbl.trial_id,'N8D1B_LY');
tbl(toSubstitude,"trial_id") = {'N8D1_LY'};

% hardcoding
condOrder = {'N4D1_LL','N6D1_LL','N8D1_LL', ...
    'N4D1_HL','N6D1_HL','N8D1_HL', ...
    'N4D1_LY','N6D1_LY','N8D1_LY'};
stimuliList = {'C1', 'C2', 'C3', 'C4', 'C5', 'C6'};
PlotColors = {[239,95,40]./255, [0,135,255]./255, [0,199,87]./255, [236,179,48]./255, [228,57,215]./255, [120,120,120]./255};

%% Plot Results

tbl.rating_stimulus = categorical(tbl.rating_stimulus, stimuliList);
tiledlayout(3,1);
% fig1 = figure( 'Position', [0 0 1200 500]); 
% legend({'REF','DiffFDN', 'SCAT', 'HH', 'RO', 'SH'},'FontSize',26, 'Location', 'northoutside', 'Orientation','horizontal')
nexttile
b1 = boxchart(categorical(tbl.trial_id, {condOrder{1:3}}), tbl.rating_score,'GroupByColor', tbl.rating_stimulus,'Notch','on','LineWidth',2);
ylabel('Score');
% xlabel('Configuration')
xline([1.5 2.5],'--','HandleVisibility','off')
xticklabels({'$N=4$','$N=6$','$N=8$'})
set(gca,'FontSize',24) 
grid minor
ylim([-2 102])
ax = gca;
set(ax, 'box', 'on', 'Visible', 'on')
xlabel('\textbf{(a)}', 'Fontsize', 26)

% fig2 = figure( 'Position', [0 0 1200 480]); title('Homogeneous Loss');
nexttile
b2 = boxchart(categorical(tbl.trial_id, {condOrder{4:6}}), tbl.rating_score,'GroupByColor', tbl.rating_stimulus,'Notch','on','LineWidth',2);
ylabel('Score'); %  xlabel('Configuration')
xline([1.5 2.5],'--','HandleVisibility','off')
% legend({'REF','DiffFDN', 'SCAT', 'HH', 'RO', 'SH'},'FontSize',26, 'Location', 'northeastoutside', 'Orientation','vertical')
xticklabels({'$N=4$','$N=6$','$N=8$'})
set(gca,'FontSize',24) 
grid minor
ylim([-2 102])
ax = gca;
set(ax, 'box', 'on', 'Visible', 'on')
xlabel('\textbf{(b)}', 'Fontsize', 26)


% fig3 = figure( 'Position', [0 0 1200 480]); title('Frequency Dependent Decay');
nexttile
b3 = boxchart(categorical(tbl.trial_id, {condOrder{7:9}}), tbl.rating_score,'GroupByColor', tbl.rating_stimulus,'Notch','on','LineWidth',2);
ylabel('Score'); 
xline([1.5 2.5],'--','HandleVisibility','off')

% legend({'REF','DiffFDN', 'SCAT', 'HH', 'RO', 'SH'},'FontSize',26, 'Location', 'northeastoutside', 'Orientation','vertical')
xticklabels({'$N=4$','$N=6$','$N=8$'})
set(gca,'FontSize',24) 
grid minor
ylim([-2 102])
b1 = applyBoxColor(b1, PlotColors);
b2 = applyBoxColor(b2, PlotColors);
b3 = applyBoxColor(b3, PlotColors);
xlabel('\textbf{(c)}', 'Fontsize', 26)
ax = gca;
set(ax, 'box', 'on', 'Visible', 'on')
lg = legend({'REF','DiffFDN-O', 'DiffFDN-SCAT', 'DiffFDN-HH', 'RO', 'SH'},'FontSize',26, 'Location', 'northoutside', 'Orientation','horizontal');
lg.Layout.Tile = 'North';
% %% difference 2
% 
% diffMatrix = zeros(length(condOrder), length(stimuliList)-1, 22);
% trial_id = [];
% rating_stimulus = [];
% relative_score = [];
% 
% for iCond = 1:length(condOrder)
%     scoresRef = tbl.('rating_score')(tbl.trial_id == categorical({condOrder{iCond}})& (tbl.rating_stimulus == categorical({'C5'})));
%     for iStimuli = 2:length(stimuliList)
%         scoresCond = tbl.('rating_score')(tbl.trial_id == categorical({condOrder{iCond}})& (tbl.rating_stimulus == categorical({stimuliList{iStimuli}})));
%         diffScores = scoresCond - scoresRef;
%         for j = 1:length(diffScores)
%             trial_id = [trial_id; categorical({condOrder{iCond}})];
%             rating_stimulus = [rating_stimulus; string(stimuliList{iStimuli})];
%             relative_score = [relative_score; diffScores(j)];
%         end
%     end
% end
% 
% tbl2 = table(trial_id, rating_stimulus, relative_score);
% tbl2.rating_stimulus = categorical(tbl2.rating_stimulus, {'C2', 'C3', 'C4'});
% 
% %% Plot Results
% 
% fig4 = figure( 'Position', [0 0 1400 800]); title('Lossless'); 
% b4 = boxchart(categorical(tbl2.trial_id, {condOrder{1:3}}), tbl2.relative_score,'GroupByColor', tbl2.rating_stimulus,'Notch','on','LineWidth',2);
% ylabel('Score'); xlabel('Configuration')
% xline([1.5 2.5],'--','HandleVisibility','off')
% legend({'REF','DiffFDN', 'SCAT', 'HH', 'RO', 'SH'},'FontSize',31, 'Location', 'northoutside', 'Orientation','horizontal')
% xticklabels({'$N=4$','$N=6$','$N=8$'})
% set(gca,'FontSize',31) 
% grid minor
% 
% fig5 = figure( 'Position', [0 0 1400 800]); title('Homogeneous Loss');
% b5 = boxchart(categorical(tbl2.trial_id, {condOrder{4:6}}), tbl2.relative_score,'GroupByColor', tbl2.rating_stimulus,'Notch','on','LineWidth',2);
% ylabel('Score'); xlabel('Configuration')
% xline([1.5 2.5],'--','HandleVisibility','off')
% legend({'REF','DiffFDN', 'SCAT', 'HH', 'RO', 'SH'},'FontSize',31, 'Location', 'northoutside', 'Orientation','horizontal')
% xticklabels({'$N=4$','$N=6$','$N=8$'})
% set(gca,'FontSize',31) 
% grid minor
% 
% fig6 = figure( 'Position', [0 0 1400 800]); title('Frequency Dependent Decay');
% b6 = boxchart(categorical(tbl2.trial_id, {condOrder{7:9}}), tbl2.relative_score,'GroupByColor', tbl2.rating_stimulus,'Notch','on','LineWidth',2);
% ylabel('Score'); xlabel('Configuration')
% xline([1.5 2.5],'--','HandleVisibility','off')
% legend({'REF','DiffFDN', 'SCAT', 'HH', 'RO', 'SH'},'FontSize',31, 'Location', 'northoutside', 'Orientation','horizontal')
% xticklabels({'$N=4$','$N=6$','$N=8$'})
% set(gca,'FontSize',31) 
% grid minor



%% Normality check 

% is the data coming form hormal distirbution ? 
shapiroInput = tbl.('rating_score')(tbl.rating_stimulus == categorical({'C2'}) | ...
    tbl.rating_stimulus == categorical({'C3'}) | ...
    tbl.rating_stimulus == categorical({'C4'}) | ...
    tbl.rating_stimulus == categorical({'C5'}));

normTestResults = normalitytest(shapiroInput');
shapiroWilkResults = normTestResults(7,2);  % no p < alpha thus there's SSD


% run multiple comparison tests 
shapiroWilkResults = zeros(length(condOrder),length(stimuliList)-1);
wilcoxonResults = zeros(length(condOrder),length(stimuliList), length(stimuliList));
hypotResults = wilcoxonResults;
for iCond = 1:length(condOrder)
    for iStimuli = 1:length(stimuliList)
       
        shapiroInput = tbl.('rating_score')(tbl.trial_id == categorical({condOrder{iCond}}) ...
            & (tbl.rating_stimulus == categorical({stimuliList{iStimuli}})));
        % normTestResults = normalitytest(shapiroInput');
        % shapiroWilkResults(iCond, iStimuli-1) = normTestResults(7,2);

        for iStimuli2 = 1:length(stimuliList)
            wilcoxonInput = [];
            wilcoxonInput = [shapiroInput, tbl.('rating_score')(tbl.trial_id == categorical({condOrder{iCond}}) ...
                             & (tbl.rating_stimulus == categorical({stimuliList{iStimuli2}})))];
            [wilcoxonResults(iCond, iStimuli,iStimuli2), hypotResults(iCond, iStimuli,iStimuli2)] = signrank(wilcoxonInput(:,1),wilcoxonInput(:,2),'alpha',0.05/15);
            
        end
    end 
end 

%% 

function b = applyBoxColor(b, PlotColors)
    for ii = 1:length(PlotColors)
        b(ii, 1).BoxFaceColor = PlotColors{1, ii};
        b(ii, 1).MarkerColor = PlotColors{1, ii};
    end
end