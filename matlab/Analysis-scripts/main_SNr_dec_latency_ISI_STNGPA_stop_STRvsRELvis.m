%% Date created 28.12.17 by Mohagheghi 

%% Analysis and visualization of spiking data

% This scripts load the data which contains the average firing rate of all
% stimulation paramtere combination across all trials and is going to
% visualize the change in the latency of the decrease in avergae firing
% rate of SNr. The data is stored in "avg_fr_data_eachtr".

%% Modified 10.01.18

% The purpose of modification is to change the visualization such that it
% proveds more information. In the first place, the same colorplots will be
% maintained with the difference that instead of STN on y-axis and separate
% figures for each STR, the will be separate figures for STN and STR on
% y-axis. All four plots will be combined in on plots. In addition the
% colorplot for the positive values will be removed and the one with the
% number of samples suppressed will be represented as percetange to total
% number of samples.

function [] = main_SNr_dec_latency_ISI_STNGPA_stop_STRvsRELvis(data_dir,fig_dir)
%     warning('off')
    nc_id = 1;
    nuclei = {'SN','FS','GA','GI','M1','M2','ST'};
    nc_names = {'SNr','FSI','GPe Arky','GPe Proto',...
                'MSN D1','MSN D2','STN'};
    pause_vals = [];
    pause_vals_str = [];
            
    fig_dir = [fig_dir,'latency-comparison-ISI-20-STRvsREL/',nuclei{nc_id},'/'];
    
    if exist(fig_dir,'dir') ~= 7
        mkdir(fig_dir)
    end

%     start_search_time = -150;                   % Time to look for the decrease

    avgfr_data = load([data_dir,'avg_fr_data_eachtr_ISI_50']);
    off_time = avgfr_data.off_time;
    off_time_str = avgfr_data.off_time_str;
    params = avgfr_data.params;
%     t_vec = avgfr_data.t_samples;
    
%     cnt = cnt(:,t_vec >= start_search_time);
%     cnt_str = cnt_str(:,t_vec >= start_search_time);
%     t_vec = t_vec(t_vec >= start_search_time);
    gpa_f = unique(params(:,1));
    stn_f = unique(params(:,2));
    rel_time_gpa = unique(params(:,3));
    rel_time_stn = unique(params(:,4));
    str_f = unique(params(:,5));
    tr_vec = unique(params(:,6));
    
%     SNr_fr_th = 2;                  % Threshold for the SNr movement trigger
    width_th = 20;
    for 
    for stnf_ind = 1:length(stn_f)
        for strf_ind = 1:length(str_f)
            for rel1_ind = 1:length(rel_time_gpa)
            for rel2_ind = 1:length(rel_time_stn)
                sel_inds = find(params(:,1) == stn_f(stnf_ind) & ...
                                params(:,2) == rel_time(rel2_ind) & ...
                                params(:,3) == str_f(strf_ind));
%                 for inds = 1:length(sel_inds)
                cnt_time = off_time(sel_inds);
                cnt_str_time = off_time_str(sel_inds);
%                 end
                num_suppressed_mov(strf_ind,rel2_ind,stnf_ind) = (sum(~isnan(cnt_str_time)) - ...
                                     sum(~isnan(cnt_time)))/sum(~isnan(cnt_str_time))*100;
                time_diff = cnt_time - cnt_str_time;
                time_diff_mean = mean(time_diff(~isnan(time_diff)));
                if isnan(time_diff_mean)
                    time_diff_avg(strf_ind,rel2_ind,stnf_ind) = 0;
                else
                    time_diff_avg(strf_ind,rel2_ind,stnf_ind) = time_diff_mean;
                end
                time_pos_diff_avg(strf_ind,rel2_ind,stnf_ind) = mean(time_diff(~isnan(time_diff) & ...
                                                                              (time_diff>=0)));
                num_non_nan(strf_ind,rel2_ind,stnf_ind) = sum(~isnan(time_diff));
                                
            end
        end
        figure;
        subplot(211)
        imagesc(rel_time,str_f,time_diff_avg(:,:,stnf_ind))
        GCA = gca;
        GCA.FontSize = 14;
        GCA.Box = 'off';
        GCA.TickDir = 'out';
        xlabel('Time to ramp offset (ms)')
        ylabel('Max input firing rate to STR (Hz)')
        title([nc_names{nc_id},...
            '-STN = ',num2str(stn_f(stnf_ind)),' (Hz)'])
        ax = colorbar();
        ax.Label.String = 'Time difference in Stop - Go experiment (ms)';
        ax.Label.FontSize = 14;
%         fig_print(gcf,[fig_dir,'STR-',num2str(stn_f(stnf_ind))])
%         close(gcf)
        
%         figure;
        subplot(212)
        imagesc(rel_time,str_f,num_suppressed_mov(:,:,stnf_ind),[0,100])
        GCA = gca;
        GCA.FontSize = 14;
        GCA.Box = 'off';
        GCA.TickDir = 'out';
        xlabel('Time to ramp offset (ms)')
        ylabel('Max input firing rate to STR (Hz)')
        title([nc_names{nc_id},...
            '-STN = ',num2str(stn_f(stnf_ind)),' (Hz)'])
        ax = colorbar();
        ax.Label.String = 'Number of Samples';
        ax.Label.FontSize = 14;
        fig = gcf;
        fig.Position = fig.Position.*[1,1,1,2];
        fig_print(gcf,[fig_dir,'numsamples-',num2str(stn_f(stnf_ind))])
        close(gcf)
        
%         figure;
%         imagesc(rel_time,stn_f,num_suppressed_mov(:,:,stnf_ind))
%         GCA = gca;
%         GCA.FontSize = 14;
%         GCA.Box = 'off';
%         GCA.TickDir = 'out';
%         xlabel('Time to ramp offset (ms)')
%         ylabel('Max input firing rate to STN (Hz)')
%         title([nc_names{nc_id},...
%             '-STR = ',num2str(str_f(strf_ind)),' (Hz)'])
%         ax = colorbar();
%         ax.Label.String = 'Number of suppressed movements';
%         ax.Label.FontSize = 14;
%         fig_print(gcf,[fig_dir,'numsuppsamples-',num2str(stn_f(stnf_ind))])
%         close(gcf)
%         
%         figure;
%         imagesc(rel_time,stn_f,time_pos_diff_avg(:,:,stnf_ind))
%         GCA = gca;
%         GCA.FontSize = 14;
%         GCA.Box = 'off';
%         GCA.TickDir = 'out';
%         xlabel('Time to ramp offset (ms)')
%         ylabel('Max input firing rate to STN (Hz)')
%         title([nc_names{nc_id},...
%             '-STR = ',num2str(stn_f(stnf_ind)),' (Hz)'])
%         ax = colorbar();
%         ax.Label.String = 'Time difference in Stop - Go experiment (ms)';
%         ax.Label.FontSize = 14;
%         fig_print(gcf,[fig_dir,'STR-',num2str(stn_f(stnf_ind)),'-posdiff'])
%         close(gcf)
    end
    save([fig_dir,'latency-var-peakth',num2str(SNr_fr_th),'-widthth',num2str(width_th)],...
         'time_diff_avg','num_non_nan','num_suppressed_mov')
end

function [disinh_ind,width] = disinhibition_find(signal,th,w_th)
%     signal = -signal;
    [~,ind,width,~] = findpeaks(-signal,'MinPeakHeight',-th,'MinPeakWidth',w_th);
%     [max_val,max_w_ind] = max(width);
    if ~isempty(ind)
        disinh_ind = ind(1);
    else
        disinh_ind = ind;
    end
end
