%% This script quantifies the delay comparisons to find which conductance is
%% more effective.

clear

res_dir = '/space2/mohaghegh-data/Working-Directory/PhD/Projects/BGmodel/MATLAB-data-analysis/Analysis/18-02-19-comparison-GPASTN-vs-STN-and-GPA/';
comparison_region = 'STN';
res_dir = [res_dir,'all-together/'];

if exist(res_dir,'dir') ~= 7
    mkdir(res_dir)
end
conds = 0.2:0.05:0.65;
gpaf = 500:500:2000;
strf = 500:500:1500;
stnf = 500:500:2000;

strf_ind = 2;
% stngpa_f = combvec(stnf,gpaf);

for c_ind = 1:length(conds)
    for f_ind = 1:length(gpaf)

        data_dir = ['/home/mohaghegh-data/temp-storage/',...
                    '/18-02-14-gostop+GPA-longsensorystim-gGPASTR-increased/',...
                    num2str(conds(c_ind)*100),...
                    '/all-in-one-numtr20/SN/latency-comparison-ISI-20-vs-',comparison_region,'-RELvsREL-mod/SN/'];
        delay_data = load([data_dir,'latency-var-peakth-widthth20']);

        td_avg = delay_data.time_diff_stngpa_stn;
        td_avg = td_avg(:,:,:,strf_ind,f_ind);

        bins = -50:51;
        cnts(c_ind,f_ind) = mean(td_avg(td_avg~=0));
    end
            %     title(['G_{Arky\rightarrow STR} = ',num2str(conds(c_ind)),', mean = ',num2str(mean(td_avg(td_avg~= 0)))])
end

figure;
imagesc(stnf,conds,cnts)
ax = colorbar();
ax.Label.String = 'Delay (ms)';
GCA = gca;
GCA.FontSize = 14;
xlabel('GPA stim (Hz)')
ylabel('G_{GPe_{Arky}\rightarrow STR}')
title(['Comparison vs sensory response in ',comparison_region])
fig_print(gcf,[res_dir,'DistofDelays-avg-GPAaxis-',comparison_region,'-STR',num2str(strf(strf_ind))])