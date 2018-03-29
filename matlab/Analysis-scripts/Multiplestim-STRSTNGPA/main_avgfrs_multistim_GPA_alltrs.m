%% Analysis of spiking data

% The purpose of this file is to compute average firing rates of SNr neurons 
% and put it in a matrix which further can be used to compute the latency
% difference after STN stimulation (Stop) with respect to Go experiment.

function [] = main_avgfrs_multistim_GPA_alltrs(data_dir,res_dir)
%     data_dir = '/space2/mohaghegh-data/temp-storage/18-01-30-gostop+GPA-longsensorystim-gGPASTR-increased/all-in-one-numtr20/';
%     res_dir = '/space2/mohaghegh-data/temp-storage/18-01-30-gostop+GPA-longsensorystim-gGPASTR-increased/all-in-one-numtr20/';
    if isempty(data_dir)
        data_dir = [pwd,'/STN-dur10.0-1000.0-2000.0-50.0/'];%mat_data/']
    end
%     fig_dir = [res_dir,'/figs/'];
%     if exist(fig_dir,'dir') ~= 7
%         mkdir(fig_dir)
%     end
    nuclei = {'SN','FS','GA','GI','M1','M2','ST'};
    nc_names = {'SNr','FSI','GPe Arky','GPe Proto',...
                'MSN D1','MSN D2','STN'};

    stimdata = load([data_dir,'all_stimspec.mat']);
    numtrs = length(stimdata.stimvars_alltrs);
    stimtimes_c = stimdata.stimvars_alltrs(1).go.STRramp.start_times + 150;
%     stimvars_c = load([data_dir_compare,'stimspec.mat']);
%     stimtimes_c = stimvars_c.STRramp.start_times + 150;    % The difference between
                                                    % start and stop times.
    stimrate_c = stimdata.stimvars_alltrs(1).go.STRramp.rates;% Rate in ramp sim.
    
    stimtimes = stimdata.stimvars_alltrs(1).gostop.STRramp.stop_times;
    stimrate = stimdata.stimvars_alltrs(1).gostop.STNstop.rates;
    
    gpa_f = unique(stimrate(1,:));      % Max cortical rate to GPA
    str_f = unique(stimrate(2,:));      % Max cortical rate to STR
    stn_f = unique(stimrate(3,:));      % Max cortical rate to STN
%    stn_f = stn_f([5,10,end]);
    str_stn_lat = unique(stimrate(5,:));% Time interval between STR and STN
    str_gpa_lat = unique(stimrate(4,:));% Time interval between STR and GPA
    
    params_str = {'GPA','STN','STR','STRGPA_reltime','STRSTN_reltime'};

    % Averaging window 

    win_width = 10;
    overlap = 1;
    
    % Averaging start and end times
    
    avg_st = -500;
    avg_end = 500;

    % nuclei_fr_hist(nuclei)
    
    all_diff_struc = struct([]);
    
%     cnt = [];
%     cnt_str = [];
%     params = [];

    for nc_id = 1:length(nuclei)
        inner_cnt = 0;
        fig_dir = [res_dir,'/figs/',nuclei{nc_id},'/'];
        if exist(fig_dir,'dir') ~= 7
            mkdir(fig_dir)
        end
        
        diff_struc = struct([]);
        
        %% Data of ramping and stop-signal
%         if exist('data','var') ~= 1
        data = load([data_dir,nuclei{nc_id}]);
%         end
%         IDsall = data.gostop.N_ids;
%         numunits = length(unique(data.gostop.N_ids)); % Too slow with high
%                                                       % memory demand.
        numunits = double(max(data.gostop.N_ids)) - double(min(data.gostop.N_ids)) + 1;
        
%         data.gostop = rmfield(data.gostop,'N_ids');
        
        trials_gostop = data.gostop.trial_ids;
        
%         IDs = IDsall - min(IDsall) + 1;
        spk_times = double(data.gostop.spk_times)/10;
    %     spk_times_d = double(spk_times);
%         numunits = max(IDs) - min(IDs) + 1;
        
        %% Data of stop signal
        
        spk_times_c = double(data.go.spk_times)/10;
        trials_go = data.go.trial_ids;
        
        tr_start = min(trials_go);
        tr_end = max(trials_go);
        tr_vec = tr_start:tr_end;
        
        clear data
        
        % Matrices initialization
        t_samples = (avg_st + win_width/2):overlap:(avg_end - win_width/2);
        cnt = zeros(size(stimrate,2),...
                    length(t_samples));
        cnt_str = zeros(size(cnt));
        params = zeros(size(cnt,1),length(params_str));
        
        
        for gpaf_ind = 1:length(gpa_f)
            disp(['Stim GPA = ',num2str(gpa_f(gpaf_ind))])
            h_sim_time = max(stimtimes(stimrate(1,:) == gpa_f(gpaf_ind)));
            l_sim_time = min(stimtimes(stimrate(1,:) == gpa_f(gpaf_ind)));
            spk_times_sel_init = spk_times(spk_times >= l_sim_time - 1000 &...
                                            spk_times <= h_sim_time + 1000);
            trials_gostop_sel_init = trials_gostop(spk_times >= l_sim_time - 1000 &...
                                            spk_times <= h_sim_time + 1000);
            stimrate_sel = stimrate(2:end,stimrate(1,:) == gpa_f(gpaf_ind));
            stimtimes_sel = stimtimes(stimrate(1,:) == gpa_f(gpaf_ind));
            for stnf_ind = 1:length(stn_f)
                disp(['Stim STN = ',num2str(stn_f(stnf_ind))])
                h_sim_time = max(stimtimes_sel(stimrate_sel(2,:) == stn_f(stnf_ind)));
                l_sim_time = min(stimtimes_sel(stimrate_sel(2,:) == stn_f(stnf_ind)));
                spk_times_sel_sec = spk_times_sel_init(spk_times_sel_init >= l_sim_time - 1000 &...
                                                spk_times_sel_init <= h_sim_time + 1000);
                trials_gostop_sel = trials_gostop_sel_init(spk_times_sel_init >= l_sim_time - 1000 &...
                                                spk_times_sel_init <= h_sim_time + 1000);
                stimtimes_sel_sec = stimtimes_sel(stimrate_sel(2,:) == stn_f(stnf_ind));
                stimrate_sel_sec = stimrate_sel(:,stimrate_sel(2,:) == stn_f(stnf_ind));
                for rel_time_ind = 1:length(stimtimes_sel_sec)
                    disp(['Rel time = ',num2str(stimrate_sel_sec(4,rel_time_ind)),...
                          num2str(stimrate_sel_sec(3,rel_time_ind))])
                    
                    str_stn_reltime = stimrate_sel_sec(4,rel_time_ind);
                    str_gpa_reltime = stimrate_sel_sec(3,rel_time_ind);

%                     for st_id = 1:length(sel_stimtimes)
                    str_freq = stimrate_sel_sec(1,stimtimes_sel_sec == stimtimes_sel_sec(rel_time_ind));
                    disp([nuclei{nc_id},'-',num2str(rel_time_ind)])
                    st_time = stimtimes_sel_sec(rel_time_ind) - 500;
                    end_time = stimtimes_sel_sec(rel_time_ind) + 500;
%                         for tr_ind = 1:length(tr_vec)
                    inner_cnt = inner_cnt + 1;
                    spk_times_sel = spk_times_sel_init(spk_times_sel_init >= st_time & ...
                                                spk_times_sel_init <= end_time);
                    [cnt_temp,~] = PSTH_mov_win(spk_times_sel,...
                        win_width,overlap,st_time,end_time,numtrs*numunits,1);

                    cnt(inner_cnt,:) = cnt_temp;

                    sel_stimtimes_c = stimtimes_c(stimrate_c == str_freq);
                    st_time = sel_stimtimes_c - 500;
                    end_time = sel_stimtimes_c + 500;
                    spk_times_sel = spk_times_c(spk_times_c >= st_time & ...
                                                spk_times_c <= end_time);
                    [cnt_str_temp,~] = PSTH_mov_win(spk_times_sel,...
                        win_width,overlap,st_time,end_time,numtrs*numunits,1);
                    cnt_str(inner_cnt,:) = cnt_str_temp;

                    params(inner_cnt,:) = [gpa_f(gpaf_ind),...
                                          stn_f(stnf_ind),...
                                          str_freq,...
                                          str_gpa_reltime,...
                                          str_stn_reltime];

                end
            end
        end
        
        save([fig_dir,'avg_fr_data_eachtr'],...
             'params','params_str','cnt','cnt_str',...
             'numunits','t_samples')
    end
