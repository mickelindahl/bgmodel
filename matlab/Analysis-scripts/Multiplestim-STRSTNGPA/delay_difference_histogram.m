%% This script quantifies the delay comparisons to find which conductance is
%% more effective.

clear

res_dir = '/space2/mohaghegh-data/Working-Directory/PhD/Projects/BGmodel/MATLAB-data-analysis/Analysis/18-03-06-comparison-GPASTN-vs-STN-and-GPA/';
comparison_region = 'STN';
res_dir = [res_dir,'all-together/'];
det_hist = 1;

if exist(res_dir,'dir') ~= 7
    mkdir(res_dir)
end
conds = 0.2:0.05:0.65;

c_ind = 1;
init_data = ['/home/mohaghegh-data/temp-storage/',...
            '/18-02-14-gostop+GPA-longsensorystim-gGPASTR-increased/',...
            num2str(conds(c_ind)*100),...
            '/all-in-one-numtr20/SN/latency-comparison-ISI-20-vs-',comparison_region,'-RELvsREL/SN/'];
tmp_data = load([init_data,'latency-var-peakth-widthth20']);

gpaf = unique(tmp_data.params_stngpa(:,1));
stnf = unique(tmp_data.params_stngpa(:,2));
relgpa = unique(tmp_data.params_stngpa(:,3));
relstn = unique(tmp_data.params_stngpa(:,4));
strf = unique(tmp_data.params_stngpa(:,5));

% gpaf = 500:500:2000;
% strf = 500:500:1500;
% stnf = 500:500:2000;

strf_ind = 2;

RRSSG = combvec(relgpa',relstn',stnf',strf(2)',gpaf');

for c_ind = 1:length(conds)

    data_dir = ['/home/mohaghegh-data/temp-storage/',...
                '/18-02-14-gostop+GPA-longsensorystim-gGPASTR-increased/',...
                num2str(conds(c_ind)*100),...
                '/all-in-one-numtr20/SN/latency-comparison-ISI-20-vs-',comparison_region,'-RELvsREL/SN/'];
    delay_data = load([data_dir,'latency-var-peakth-widthth20']);

    td_avg = delay_data.time_diff_stngpa_stn;
    td_avg = td_avg(:,:,:,strf_ind,:);

    bins = -50:51;
    cnts(c_ind,:) = histcounts(td_avg(td_avg~=0),bins,'Normalization','probability');
            %     title(['G_{Arky\rightarrow STR} = ',num2str(conds(c_ind)),', mean = ',num2str(mean(td_avg(td_avg~= 0)))])
    if det_hist == 1
        neg_inds = find(td_avg<0);
        neg_for_pars = RRSSG(:,neg_inds);
        figure;
        subplot(2,3,1)
        histogram(td_avg,bins)
        subplot(2,3,2)
        histogram(neg_for_pars(1,:),relgpa)
        title('GPA rel time')
        subplot(2,3,3)
        histogram(neg_for_pars(2,:),relstn)
        title('STN rel time')
        subplot(2,3,4)
        histogram(neg_for_pars(3,:),stnf)
        title('STN freq')
        subplot(2,3,5)
        histogram(neg_for_pars(4,:),strf)
        title('STR freq')
        subplot(2,3,6)
        histogram(neg_for_pars(5,:),gpaf)
        title('GPA freq')
        fig_print(gcf,[res_dir,num2str(conds(c_ind)*100)])
        close(gcf)
    end
end

figure;
imagesc(bins(1:end-1),conds,cnts)
ax = colorbar();
ax.Label.String = 'Probability';
GCA = gca;
GCA.FontSize = 14;
xlabel('Delay (ms)')
ylabel('G_{GPe_{Arky}\rightarrow STR}')
title(['Comparison vs sensory response in ',comparison_region])
fig_print(gcf,[res_dir,'DistofDelays-',comparison_region,'-STR',num2str(strf(strf_ind))])