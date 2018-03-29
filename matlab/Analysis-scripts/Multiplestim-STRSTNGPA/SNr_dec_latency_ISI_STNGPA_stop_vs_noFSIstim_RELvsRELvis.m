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

function [] = SNr_dec_latency_ISI_STNGPA_stop_vs_noFSIstim_RELvsRELvis(data_dir,noFSI_dir,fig_dir)
%     warning('off')
    
%     STNGPA_stop_dir = '/space2/mohaghegh-data/temp-storage/18-01-31-gostop+GPA-longsensorystim-gGPASTR-increased-5weights/2/all-in-one-numtr20/figs/SN/';
%     data_dir = STNGPA_stop_dir;
%     GPA_stop_dir = '/space2/mohaghegh-data/Working-Directory/PhD/Projects/BGmodel/MATLAB-data-analysis/Analysis/Gostop-avgfrs-STN-1000-2000-STR-400-2000-reltime--130-20/';
    nc_id = 1;
    nuclei = {'SN','FS','GA','GI','M1','M2','ST'};
    nc_names = {'SNr','FSI','GPe Arky','GPe Proto',...
                'MSN D1','MSN D2','STN'};
    pause_vals = [];
    pause_vals_str = [];
    
    exclusive_fig_dirs = 1;
    
    fig_dir = data_dir;
            
    fig_dir = [fig_dir,'latency-comparison-ISI-20-vs-noFSI-wo-STR-RELvsREL/',nuclei{nc_id},'/'];
    
    if exist(fig_dir,'dir') ~= 7
        mkdir(fig_dir)
    end

%     start_search_time = -150;                   % Time to look for the decrease

    data_stngpa = load([data_dir,'avg_fr_data_eachtr_ISI']);
    offtime_stngpa = data_stngpa.off_time;
    offtime_sg_str = data_stngpa.off_time_str;
    params_stngpa = data_stngpa.params;
%     t_vec = avgfr_data.t_samples;
    
    data_nofsi = load([noFSI_dir,'avg_fr_data_eachtr_ISI']);
    offtime_nofsi = data_nofsi.off_time;
    offtime_nofsi = data_nofsi.off_time_str;
    params_nofsi = data_nofsi.params;
%     cnt = cnt(:,t_vec >= start_search_time);
%     cnt_str = cnt_str(:,t_vec >= start_search_time);
%     t_vec = t_vec(t_vec >= start_search_time);
    gpa_f = unique(params_stngpa(:,1));
    stn_f = unique(params_stngpa(:,2));
    rel_time_gpa = unique(params_stngpa(:,3));
    rel_time_stn = unique(params_stngpa(:,4));
    str_f = unique(params_stngpa(:,5));
    tr_vec = unique(params_stngpa(:,6));
    
%     SNr_fr_th = 2;                  % Threshold for the SNr movement trigger
    width_th = 20;
    for gpaf_ind = 1:length(gpa_f)
        for stnf_ind = 1:length(stn_f)
            for strf_ind = 1:length(str_f)
                for rel1_ind = 1:length(rel_time_gpa)
                    for rel2_ind = 1:length(rel_time_stn)
                        % Selected IDs for stop signal in GPA and STN
                        sel_inds = find(params_stngpa(:,1) == gpa_f(gpaf_ind) & ...
                                        params_stngpa(:,2) == stn_f(stnf_ind) & ...
                                        params_stngpa(:,3) == rel_time_gpa(rel1_ind) & ...
                                        params_stngpa(:,4) == rel_time_stn(rel2_ind) & ...
                                        params_stngpa(:,5) == str_f(strf_ind));
                                    
        %                 for inds = 1:length(sel_inds)
                        cnt_time = offtime_stngpa(sel_inds);
                        cnt_str_time = offtime_sg_str(sel_inds);
                        
                        num_mov_str(rel1_ind,rel2_ind,stnf_ind,strf_ind,gpaf_ind) = sum(~isnan(cnt_str_time));
                        num_mov(rel1_ind,rel2_ind,stnf_ind,strf_ind,gpaf_ind) = sum(~isnan(cnt_time));
                        
%                         num_suppressed_mov(rel1_ind,rel2_ind,stnf_ind,strf_ind,gpaf_ind) = (sum(~isnan(cnt_str_time)) - ...
%                                              sum(~isnan(cnt_time)))/sum(~isnan(cnt_str_time))*100;  %percent

                        num_suppressed_mov(rel1_ind,rel2_ind,stnf_ind,strf_ind,gpaf_ind) = sum(~isnan(cnt_str_time)) - ...
                                             sum(~isnan(cnt_time));
                        time_diff = cnt_time - cnt_str_time;
                        time_diff_mean = mean(time_diff(~isnan(time_diff)));
                        
                        % Selected IDs for stop signal in STN
                        sel_ind_nofsi = find(params_nofsi(:,1) == gpa_f(gpaf_ind) & ...
                                            params_nofsi(:,2) == stn_f(stnf_ind) & ...
                                            params_nofsi(:,3) == rel_time_gpa(rel1_ind) & ...
                                            params_nofsi(:,4) == rel_time_stn(rel2_ind) & ...
                                            params_nofsi(:,5) == str_f(strf_ind));
                                       
                        cnt_time_nofsi = offtime_nofsi(sel_ind_nofsi);
                        cnt_str_time_nofsi = offtime_nofsi(sel_ind_nofsi);
                        
                        num_mov_str_gpa(rel1_ind,rel2_ind,stnf_ind,strf_ind,gpaf_ind) = sum(~isnan(cnt_str_time_nofsi));
                        num_mov_gpa(rel1_ind,rel2_ind,stnf_ind,strf_ind,gpaf_ind) = sum(~isnan(cnt_time_nofsi));
                        
        %                 end
%                         num_suppressed_mov_stn(rel1_ind,rel2_ind,stnf_ind,strf_ind,gpaf_ind) = (sum(~isnan(cnt_str_time_gpa)) - ...
%                                              sum(~isnan(cnt_time_gpa)))/sum(~isnan(cnt_str_time_gpa))*100;
%                                              %Percent
                        num_suppressed_mov_stn(rel1_ind,rel2_ind,stnf_ind,strf_ind,gpaf_ind) = sum(~isnan(cnt_str_time_nofsi)) - ...
                                             sum(~isnan(cnt_time_nofsi));
                                         
                        time_diff_nofsi = cnt_time_nofsi - cnt_str_time_nofsi;
                        time_diff_mean_nofsi = mean(time_diff_nofsi(~isnan(time_diff_nofsi)));
                        
                        time_diff = cnt_time;
                        time_diff_nofsi = cnt_time_nofsi;
                        
                        if length(time_diff) > length(time_diff_nofsi)
                            DIFF = time_diff(1:length(time_diff_nofsi)) - time_diff_nofsi;
                        else
                            DIFF = time_diff - time_diff_nofsi(1:length(time_diff));
                        end
                        
                        mean_DIFF = mean(DIFF(~isnan(DIFF)));
                        
                        if isnan(mean_DIFF)
                            time_diff_avg(rel1_ind,rel2_ind,stnf_ind,strf_ind,gpaf_ind) = 0;
                        else
                            time_diff_avg(rel1_ind,rel2_ind,stnf_ind,strf_ind,gpaf_ind) = mean_DIFF;
                        end
                        
%                         if isnan(time_diff_mean_nofsi)
%                             time_diff_avg_stn(rel1_ind,rel2_ind,stnf_ind,strf_ind,gpaf_ind) = 0;
%                         else
%                             time_diff_avg_stn(rel1_ind,rel2_ind,stnf_ind,strf_ind,gpaf_ind) = time_diff_mean_nofsi;
%                         end
%                         time_pos_diff_avg(rel1_ind,rel2_ind,stnf_ind,strf_ind,gpaf_ind) = mean(time_diff(~isnan(time_diff) & ...
%                                                                                       (time_diff>=0)));
%                         num_non_nan(rel1_ind,rel2_ind,stnf_ind,strf_ind,gpaf_ind) = sum(~isnan(time_diff));
                    end
                end
                
                figure;
                subplot(211)
%                 imagesc(rel_time_stn,rel_time_gpa,time_diff_avg(:,:,stnf_ind,strf_ind,gpaf_ind) ...
%                                                 - time_diff_avg_stn(:,:,stnf_ind,strf_ind,gpaf_ind))
                imagesc(rel_time_stn,rel_time_gpa,time_diff_avg(:,:,stnf_ind,strf_ind,gpaf_ind))
                GCA = gca;
                GCA.FontSize = 14;
                GCA.Box = 'off';
                GCA.TickDir = 'out';
                xlabel('Time to ramp offset (ms),STN')
                ylabel('Time to ramp offset (ms),GPA')
                title([nc_names{nc_id},...
                    '-GPA = ',num2str(gpa_f(gpaf_ind),'%i'),' (Hz)',...
                    '-STN = ',num2str(stn_f(stnf_ind),'%i'),' (Hz)',...
                    '-STR = ',num2str(str_f(strf_ind),'%i'),' (Hz)'])
                ax = colorbar();
                ax.Label.String = 'Time difference in Stop - Go experiment (ms)';
                ax.Label.FontSize = 14;
        %         fig_print(gcf,[fig_dir,'STR-',num2str(stn_f(stnf_ind))])
        %         close(gcf)

        %         figure;
                subplot(212)
                imagesc(rel_time_stn,rel_time_gpa,num_suppressed_mov(:,:,stnf_ind,strf_ind,gpaf_ind)...
                                                - num_suppressed_mov_stn(:,:,stnf_ind,strf_ind,gpaf_ind),...
                                                [-max(tr_vec),max(tr_vec)])
                GCA = gca;
                GCA.FontSize = 14;
                GCA.Box = 'off';
                GCA.TickDir = 'out';
                xlabel('Time to ramp offset (ms),STN')
                ylabel('Time to ramp offset (ms),GPA')
                ax = colorbar();
                ax.Label.String = 'Number of Samples';
                ax.Label.FontSize = 14;
                fig = gcf;
                fig.Position = fig.Position.*[1,1,1,2];
                fig_print(gcf,[fig_dir,'diff-stn-numsamples-',num2str(gpa_f(gpaf_ind),'%i')...
                                                    ,num2str(stn_f(stnf_ind),'%i')...
                                                    ,num2str(str_f(strf_ind),'%i')])
                                                
                if exclusive_fig_dirs == 1                                
                    STR_fig_dir = [fig_dir,'STR',num2str(str_f(strf_ind),'%i'),'/'];
                    if exist(STR_fig_dir,'dir') ~= 7
                        mkdir(STR_fig_dir)
                    end
                    
                    STN_fig_dir = [fig_dir,'STN',num2str(stn_f(stnf_ind),'%i'),'/'];
                    if exist(STN_fig_dir,'dir') ~= 7
                        mkdir(STN_fig_dir)
                    end
                    
                    GPA_fig_dir = [fig_dir,'GPA',num2str(gpa_f(gpaf_ind),'%i'),'/'...
                                    ];
                    if exist(GPA_fig_dir,'dir') ~= 7
                        mkdir(GPA_fig_dir)
                    end
                    
                    fig_print(gcf,[STR_fig_dir,'diff-stn-numsamples-',num2str(gpa_f(gpaf_ind),'%i')...
                                              ,num2str(stn_f(stnf_ind),'%i')])

                    fig_print(gcf,[STN_fig_dir,'diff-stn-numsamples-',num2str(stn_f(stnf_ind),'%i')...
                                              ,num2str(str_f(strf_ind),'%i')])
                                          
                    fig_print(gcf,[GPA_fig_dir,'diff-stn-numsamples-',num2str(stn_f(stnf_ind),'%i')...
                                              ,num2str(str_f(strf_ind),'%i')])
                                          
                end
                
                close(gcf)
            end
        end
    end
%         time_diff_stngpa_stn = time_diff_avg - time_diff_avg_stn;
        save([fig_dir,'latency-var-peakth','-widthth',num2str(width_th)],...
             'time_diff_avg','time_diff_avg',...
             'num_suppressed_mov','num_suppressed_mov_stn',...
             'params_stngpa','params_nofsi','num_mov_str','num_mov','num_mov_str_gpa','num_mov_gpa')
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
