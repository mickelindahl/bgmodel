%% Date Created 19.03.18 by M. Mohagheghi

% The purpose of this script is to take different stimulus parameters and
% extract the corresponding delay from the data.

function [] = hist_of_delays(data_dir)

    show_diff = false;

    stn_f_par = input('STN stim param: ');
    str_f_par = input('STR stim param: ');
    reltime_par = input('REL stim param: ');

    hist_edges = -20:20;

    nc_id = 1;
    nuclei = {'SN','FS','GA','GI','M1','M2','ST'};
    nc_names = {'SNr','FSI','GPe Arky','GPe Proto',...
                'MSN D1','MSN D2','STN'};
    pause_vals = [];
    pause_vals_str = [];

    fig_dir = [data_dir,'samplehists/'];
        
    if exist(fig_dir,'dir') ~= 7
        mkdir(fig_dir)
    end

    %     start_search_time = -150;                   % Time to look for the decrease

    avgfr_data = load([data_dir,'avg_fr_data_eachtr_ISI']);
    off_time = avgfr_data.off_time;
    off_time_str = avgfr_data.off_time_str;
    params = avgfr_data.params;
    %     t_vec = avgfr_data.t_samples;

    %     cnt = cnt(:,t_vec >= start_search_time);
    %     cnt_str = cnt_str(:,t_vec >= start_search_time);
    %     t_vec = t_vec(t_vec >= start_search_time);

    stn_f = unique(params(:,1));
    rel_time = unique(params(:,2));
    str_f = unique(params(:,3));
    tr_vec = unique(params(:,4));

    %     SNr_fr_th = 2;                  % Threshold for the SNr movement trigger
    width_th = 20;
    
    stnf_ind = find(stn_f == stn_f_par);
    strf_ind = find(str_f == str_f_par);
    relt_ind = find(rel_time == reltime_par);


    sel_inds = find(params(:,1) == stn_f(stnf_ind) & ...
                    params(:,2) == rel_time(relt_ind) & ...
                    params(:,3) == str_f(strf_ind));
%                 for inds = 1:length(sel_inds)
    cnt_time = off_time(sel_inds);
    cnt_str_time = off_time_str(sel_inds);
%                 end
    num_suppressed_mov(strf_ind,relt_ind,stnf_ind) = (sum(~isnan(cnt_str_time)) - ...
                         sum(~isnan(cnt_time)))/sum(~isnan(cnt_str_time))*100;
    time_diff = cnt_time - cnt_str_time;
    time_diff_mean = mean(time_diff(~isnan(time_diff)));
    
    figure;
    
    if ~show_diff
        hist_edges = -100:0;
        histogram(cnt_time(~isnan(cnt_time)),hist_edges,'facecolor','red')
        hold on
        histogram(cnt_str_time(~isnan(cnt_str_time)),hist_edges,'facecolor','blue')
        legend({'STN','w/o STN'})
        title(['STN=',num2str(stn_f(stnf_ind),'%i'),'-',...
               'STR=',num2str(str_f(strf_ind),'%i'),'-',...
               'REL=',num2str(rel_time(relt_ind),'%i')])
        xlabel('Movement decrease onset (ms)')
        fig_print(gcf,[fig_dir,'Both-STN',num2str(stn_f(stnf_ind),'%i'),'-',...
                               'STR',num2str(str_f(strf_ind),'%i'),'-',...
                               'REL',num2str(rel_time(relt_ind),'%i')])
        
    else
        hist_edges = -20:20;
        histogram(time_diff(~isnan(time_diff)),hist_edges)
        title(['STN=',num2str(stn_f(stnf_ind),'%i'),'-',...
               'STR=',num2str(str_f(strf_ind),'%i'),'-',...
               'REL=',num2str(rel_time(relt_ind),'%i')])
        xlabel('Delay (ms)')
        fig_print(gcf,[fig_dir,'Diff-STN',num2str(stn_f(stnf_ind),'%i'),'-',...
                               'STR',num2str(str_f(strf_ind),'%i'),'-',...
                               'REL',num2str(rel_time(relt_ind),'%i')])
    end