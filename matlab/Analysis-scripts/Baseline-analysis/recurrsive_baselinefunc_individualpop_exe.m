%% This script apply baseline finding algorithm on directories which contain
%% different results for different Arky to STR strengths.
clear

conds = 0.2:0.05:1.2;
tr_vec = 1:10;


% desired_pops = {'M1','M2','FS'};
desired_pops = {'M1','M2'};
nucleus_pop_name = 'MSN';

switch desired_pops{1}
    case {'M1','M2'}
        fr_vec = 0:0.1:5;
        nucleus_name = 'STR';
    case {'GA','GI'}
        fr_vec = 0:1:60;
        nucleus_name = 'GP';
    case {'FS'}
        fr_vec = 0:1:30;
        nucleus_name = 'STR';
%     case {'GA'}
%         nucleus_pop_name = 'GParky';
%     case {'GI'}
%         nucleus_pop_name = 'GPproto';
end


for dir_id = 1:length(conds)
    clear count_trs
    disp(['Analyzing data for conductance ',num2str(conds(dir_id))])
    for tr_ind = tr_vec
        disp(['trial # ',num2str(tr_ind)])
        data_dir = ['/home/mohaghegh-data/temp-storage/18-02-15-baseline-gGPASTR-increased/',...
                    'GPASTR-Wmod',num2str(conds(dir_id)*100),'-',num2str(dir_id-1),'-tr',num2str(tr_ind),'/'];
        res_dir = '/home/mohaghegh-data/temp-storage/18-02-15-baseline-gGPASTR-increased/Dist-avg-frs/';
        
        if exist(data_dir,'dir') == 7
            [cnt_tmp] = baseline_firingrate(data_dir,res_dir,desired_pops,0,fr_vec);
            counts_trs(tr_ind,:) = mean(cnt_tmp,1);
        end
        
    end
    counts_units(dir_id,:) = mean(counts_trs,1);
end

conds_def = 0.04;
for dir_id = 1:length(conds_def)
    clear count_trs
    disp(['Analyzing data for conductance ',num2str(conds_def(dir_id))])
    for tr_ind = tr_vec
        disp(['trial # ',num2str(tr_ind)])
        data_dir = ['/home/mohaghegh-data/temp-storage/18-02-15-baseline-gGPASTR-increased/',...
                    'GPASTR-Wmod',num2str(conds_def(dir_id)*100),'-',num2str(dir_id-1),'-tr',num2str(tr_ind),'/'];
        res_dir = '/home/mohaghegh-data/temp-storage/18-02-15-baseline-gGPASTR-increased/Dist-avg-frs/';
        
        if exist(data_dir,'dir') == 7
            [cnt_tmp] = baseline_firingrate(data_dir,res_dir,desired_pops,0,fr_vec);
            counts_trs(tr_ind,:) = mean(cnt_tmp,1);
        end
        
    end
    counts_default = mean(counts_trs,1);
end

all_conds = [0.15,conds];
all_counts = [counts_default;counts_units];

figure;
% Simulation baseline firing rate
subplot(211)
imagesc(fr_vec(1:end-1),all_conds,all_counts)
ax = colorbar();
ax.Label.String = 'Probability';
GCA = gca;
GCA.FontSize = 14;
GCA.YTick = [0.15,GCA.YTick];
GCA.YTickLabel{1} = num2str(conds_def);
xlabel('Fring rate (Hz)')
ylabel('G_{GPe_{Arky}\rightarrow STR}')
title(['Baseline Firing rate for ',nucleus_pop_name])

% Model baseline firing rate
subplot(212)
baseline_expdata = load([res_dir,'baseline-exp-',nucleus_name,'-wo-allgo']);
histogram(baseline_expdata.baseline_fr(1,:),fr_vec,'Normalization','probability')
GCA = gca;
GCA.FontSize = 14;
xlabel('Fring rate (Hz)')
ylabel('Probability')
fig_print(gcf,[res_dir,'baseline-fr-',nucleus_pop_name,'-wo-allgo'])