%% Date Created 19.03.18 by M. Mohagheghi

% The purpose of this script is to take different stimulus parameters and
% extract the corresponding delay from the data. This script is specially
% designed for sensory stimulation in both GPA and STN with respect to STN
% stimulation alone.

function [] = hists_of_delays(data_dir,STN_stop_dir)

    show_diff = false;

    gpa_f_par = input('GPA stim param: ');
    stn_f_par = input('STN stim param: ');
    str_f_par = input('STR stim param: ');
    reltime_ss_par = input('REL STR-STN stim param: ');
    reltime_sg_par = input('REL STR-GPA stim param: ');
    
    nc_id = 1;
    nuclei = {'SN','FS','GA','GI','M1','M2','ST'};
    nc_names = {'SNr','FSI','GPe Arky','GPe Proto',...
                'MSN D1','MSN D2','STN'};
    pause_vals = [];
    pause_vals_str = [];
    
    exclusive_fig_dirs = 1;
    
    fig_dir = [data_dir,'samplehists/'];
        
    if exist(fig_dir,'dir') ~= 7
        mkdir(fig_dir)
    end


%     start_search_time = -150;                   % Time to look for the decrease

    data_stngpa = load([data_dir,'avg_fr_data_eachtr_ISI']);
    offtime_stngpa = data_stngpa.off_time;
    offtime_sg_str = data_stngpa.off_time_str;
    params_stngpa = data_stngpa.params;
%     t_vec = avgfr_data.t_samples;
    
    data_stn = load([STN_stop_dir,'avg_fr_data_eachtr_ISI']);
    offtime_stn = data_stn.off_time;
    offtime_s_str = data_stn.off_time_str;
    params_stn = data_stn.params;
%     cnt = cnt(:,t_vec >= start_search_time);
%     cnt_str = cnt_str(:,t_vec >= start_search_time);
%     t_vec = t_vec(t_vec >= start_search_time);
    gpa_f = unique(params_stngpa(:,1));
    stn_f = unique(params_stngpa(:,2));
    rel_time_gpa = unique(params_stngpa(:,3));
    rel_time_stn = unique(params_stngpa(:,4));
    str_f = unique(params_stngpa(:,5));
    tr_vec = unique(params_stngpa(:,6));
    
    
    gpaf_ind = find(gpa_f == gpa_f_par);
    stnf_ind = find(stn_f == stn_f_par);
    strf_ind = find(str_f == str_f_par);
    rel1_ind = find(rel_time_gpa == reltime_sg_par);
    rel2_ind = find(rel_time_stn == reltime_ss_par);
    
    
    
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
%                                              sum(~isnan(cnt_time)))/sum(~isnan(cnt_str_time))*100;
%                                              %Percent

    num_suppressed_mov(rel1_ind,rel2_ind,stnf_ind,strf_ind,gpaf_ind) = sum(~isnan(cnt_str_time)) - ...
                         sum(~isnan(cnt_time));
    time_diff = cnt_time - cnt_str_time;
    time_diff_mean = mean(time_diff(~isnan(time_diff)));

    % Selected IDs for stop signal in STN
    sel_ind_stn = find(params_stn(:,3) == str_f(strf_ind) & ...
                       params_stn(:,2) == rel_time_stn(rel2_ind) & ...
                       params_stn(:,1) == stn_f(stnf_ind) & ...
                       params_stn(:,4) >= tr_vec(1) & ...
                       params_stn(:,4) <= tr_vec(end));

    cnt_time_stn = offtime_stn(sel_ind_stn);
    cnt_str_time_stn = offtime_s_str(sel_ind_stn);

    num_mov_str_stn(rel1_ind,rel2_ind,stnf_ind,strf_ind,gpaf_ind) = sum(~isnan(cnt_str_time_stn));
    num_mov_stn(rel1_ind,rel2_ind,stnf_ind,strf_ind,gpaf_ind) = sum(~isnan(cnt_time_stn));
    time_diff_stn = cnt_time_stn - cnt_str_time_stn;
    
%                 end
%                         num_suppressed_mov_stn(rel1_ind,rel2_ind,stnf_ind,strf_ind,gpaf_ind) = (sum(~isnan(cnt_str_time_stn)) - ...
%                                              sum(~isnan(cnt_time_stn)))/sum(~isnan(cnt_str_time_stn))*100;%
%                                              Percent
    num_suppressed_mov_stn(rel1_ind,rel2_ind,stnf_ind,strf_ind,gpaf_ind) = sum(~isnan(cnt_str_time_stn)) - ...
                         sum(~isnan(cnt_time_stn));
    if length(time_diff) > length(time_diff_stn)
        DIFF = time_diff(1:length(time_diff_stn)) - time_diff_stn;
    else
        DIFF = time_diff - time_diff_stn(1:length(time_diff));
    end
    figure;
    
    if show_diff
        hist_edges = -20:20;
        histogram(DIFF(~isnan(DIFF)),hist_edges)
        title(['Diff-GPA=',num2str(gpa_f(gpaf_ind),'%i'),'-',...
               'STN=',num2str(stn_f(stnf_ind),'%i'),'-',...
               'STR=',num2str(str_f(strf_ind),'%i'),'-',...
               'RELSG=',num2str(rel_time_gpa(rel1_ind),'%i'),'-',...
               'RELSS=',num2str(rel_time_stn(rel2_ind),'%i')])
        xlabel('Delay (ms)')
        fig_print(gcf,[fig_dir,'Diff-GPA',num2str(gpa_f(gpaf_ind),'%i'),'-',...
                               'STN',num2str(stn_f(stnf_ind),'%i'),'-',...
                               'STR',num2str(str_f(strf_ind),'%i'),'-',...
                               'RELSG',num2str(rel_time_gpa(rel1_ind),'%i'),'-',...
                               'RELSS',num2str(rel_time_stn(rel2_ind),'%i')])
    else
        hist_edges = -100:0;
        histogram(cnt_time(~isnan(cnt_time)),hist_edges,'facecolor','green')
        hold on
        histogram(cnt_time_stn(~isnan(cnt_time_stn)),hist_edges,'facecolor','red')
        histogram(cnt_str_time(~isnan(cnt_str_time)),hist_edges,'facecolor','blue')
        legend({'GPA+STN','STN','w/o SS'})
        title(['All-GPA=',num2str(gpa_f(gpaf_ind),'%i'),'-',...
               'STN=',num2str(stn_f(stnf_ind),'%i'),'-',...
               'STR=',num2str(str_f(strf_ind),'%i'),'-',...
               'RELSG=',num2str(rel_time_gpa(rel1_ind),'%i'),'-',...
               'RELSS=',num2str(rel_time_stn(rel2_ind),'%i')])
        xlabel('Movement decrease onset (ms)')
        fig_print(gcf,[fig_dir,'All-GPA',num2str(gpa_f(gpaf_ind),'%i'),'-',...
                               'STN',num2str(stn_f(stnf_ind),'%i'),'-',...
                               'STR',num2str(str_f(strf_ind),'%i'),'-',...
                               'RELSG',num2str(rel_time_gpa(rel1_ind),'%i'),'-',...
                               'RELSS',num2str(rel_time_stn(rel2_ind),'%i')])
    end