%% This script quantifies the delay comparisons to find which conductance is
%% more effective.
clear
close all

res_dir = '/space2/mohaghegh-data/Working-Directory/PhD/Projects/BGmodel/MATLAB-data-analysis/Analysis/18-02-19-comparison-GPASTN-vs-STN-and-GPA/';
comparison_region = 'GPA';
res_dir = [res_dir,'all_varies/'];

if exist(res_dir,'dir') ~= 7
    mkdir(res_dir)
end

conds = 0.2:0.05:0.65;
gpaf = 500:500:2000;
strf = 500:500:1500;
stnf = 500:500:2000;



for gp_ind = 1:length(gpaf)
    for stn_ind = 1:length(stnf)
        for str_ind = 1:length(strf)
            for c_ind = 1:length(conds)

                data_dir = ['/home/mohaghegh-data/temp-storage/',...
                            '/18-02-14-gostop+GPA-longsensorystim-gGPASTR-increased/',...
                            num2str(conds(c_ind)*100),...
                            '/all-in-one-numtr20/SN/latency-comparison-ISI-20-vs-',comparison_region,'-RELvsREL/SN/'];
                delay_data = load([data_dir,'latency-var-peakth-widthth20']);

                td_avg = delay_data.time_diff_stngpa_stn;
                td_avg = td_avg(:,:,stn_ind,str_ind,gp_ind);

                bins = -50:51;
                cnts(c_ind,:) = histcounts(td_avg(td_avg~=0),bins,'Normalization','probability');
            %     title(['G_{Arky\rightarrow STR} = ',num2str(conds(c_ind)),', mean = ',num2str(mean(td_avg(td_avg~= 0)))])
            end
            
            res_dir2 = [res_dir,'STR',num2str(strf(str_ind)),'-STN',num2str(stnf(stn_ind)),'/'];
            
            if exist(res_dir2,'dir') ~= 7
                mkdir(res_dir2)
            end

            figure;
            imagesc(bins(1:end-1),conds,cnts)
            ax = colorbar();
            ax.Label.String = 'Probability';
            GCA = gca;
            GCA.FontSize = 14;
            xlabel('Delay (ms)')
            ylabel('G_{GPe_{Arky}\rightarrow STR}')
            title(['Comparison vs sensory response in ',comparison_region,...
                   '-STN',num2str(stnf(stn_ind)),'-GPA',num2str(gpaf(gp_ind)),'-STR',num2str(strf(str_ind))])
            fig_print(gcf,[res_dir2,'DistofDelays-',comparison_region,...
                           '-GPA',num2str(gpaf(gp_ind))])
        end
    end
end