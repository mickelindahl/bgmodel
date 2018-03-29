%% Analysis of spiking data

% The purpose of this file is to compute average firing rates of SNr neurons 
% and put it in a matrix which further can be used to compute the latency
% difference after STN stimulation (Stop) with respect to Go experiment.

function [] = main_avgfrs_multistim_eachtr_ISI_parfor(data_dir,res_dir,disinh_width)
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
    
    str_f = unique(stimrate(1,:));      % Max cortical rate to STR
    stn_f = unique(stimrate(2,:));      % Max cortical rate to STN
%    stn_f = stn_f([5,10,end]);
    str_stn_lat = unique(stimrate(3,:));% Time interval between STR and STN
    
    params_str = {'STN','Reltime','STR','Trial'};

    % Averaging window 
%     disinh_width = 20;
%     win_width = 10;
%     overlap = 1;
%     
%     width_th = 
    
    % Averaging start and end times
    
    avg_st = -500;
    avg_end = 500;

    % nuclei_fr_hist(nuclei)
    
    all_diff_struc = struct([]);
    
%     cnt = [];
%     cnt_str = [];
%     params = [];

    for nc_id = 1:1%length(nuclei)
        inner_cnt = 0;
        fig_dir = [res_dir,'/figs/',nuclei{nc_id},'/'];
        if exist(fig_dir,'dir') ~= 7
            mkdir(fig_dir)
        end
        
        diff_struc = struct([]);
        
        %% Data of ramping and stop-signal
        
        data = load([data_dir,nuclei{nc_id}]);
%         IDsall = data.gostop.N_ids;
%         numunits = length(unique(data.gostop.N_ids)); % Too slow with high
%                                                       % memory demand.
        numunits = double(max(data.gostop.N_ids)) - double(min(data.gostop.N_ids)) + 1;
        
        data.gostop = rmfield(data.gostop,'N_ids');
        
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
%         t_samples = (avg_st + win_width/2):overlap:(avg_end - win_width/2);
%         cnt = zeros(length(stn_f)*length(str_stn_lat)*length(str_f)*length(tr_vec),...
%                     length(t_samples));
%         cnt_str = zeros(size(cnt));
%         params = zeros(size(cnt,1),length(params_str));

%         cnt = [];
        off_time = [];
%         cnt_str = [];
        off_time_str = [];
        params = [];
        
%         num_samples = length(t_samples);
        

        for stnf_ind = 1:length(stn_f)
%             disp(['Stim STN = ',num2str(stn_f(stnf_ind))])
            h_sim_time = max(stimtimes(stimrate(2,:) == stn_f(stnf_ind)));
            l_sim_time = min(stimtimes(stimrate(2,:) == stn_f(stnf_ind)));
            spk_times_sel_init = spk_times(spk_times >= l_sim_time - 1000 &...
                                            spk_times <= h_sim_time + 1000);
            trials_gostop_sel = trials_gostop(spk_times >= l_sim_time - 1000 &...
                                            spk_times <= h_sim_time + 1000);
            tic
            for strf_ind = 1:length(str_f)
                disp(['Stim STN = ',num2str(stn_f(stnf_ind)),'; Stim STR = ',num2str(str_f(strf_ind))])
%                 disp(['Stim STR = ',num2str(str_f(strf_ind))])
                h_sim_time = max(stimtimes(stimrate(2,:) == stn_f(stnf_ind) & ...
                                 stimrate(1,:) == str_f(strf_ind)));
                l_sim_time = min(stimtimes(stimrate(2,:) == stn_f(stnf_ind) & ...
                                 stimrate(1,:) == str_f(strf_ind)));
                spk_times_sel_sec = spk_times_sel_init(spk_times_sel_init >= l_sim_time - 1000 &...
                                            spk_times_sel_init <= h_sim_time + 1000);
                trials_gostop_sec = trials_gostop_sel(spk_times_sel_init >= l_sim_time - 1000 &...
                                                spk_times_sel_init <= h_sim_time + 1000);
                
%                 for rel_time_ind = 1:length(str_stn_lat)
%                     disp(['Rel time = ',num2str(str_stn_lat(rel_time_ind))])
%                     sel_stimtimes = stimtimes(...
%                                     stimrate(3,:) == str_stn_lat(rel_time_ind) & ...
%                                     stimrate(2,:) == stn_f(stnf_ind));
                sel_stimtimes = stimtimes(...
                                    stimrate(1,:) == str_f(strf_ind) & ...
                                    stimrate(2,:) == stn_f(stnf_ind));
                                
                % Average firing rate for trials without STN brief
                % stimulation
                
                str_freq = str_f(strf_ind);

                for st_id = 1:length(sel_stimtimes)
%                     str_freq = stimrate(3,stimtimes == sel_stimtimes(st_id));
                    rel_time = stimrate(3,stimtimes == sel_stimtimes(st_id));
%                     disp([nuclei{nc_id},'-',num2str(st_id)])
                    st_time = sel_stimtimes(st_id) - 500;
                    end_time = sel_stimtimes(st_id) + 500;
                    
                    sel_stimtimes_c = stimtimes_c(stimrate_c == str_freq);
                    st_time_c = sel_stimtimes_c - 500;
                    end_time_c = sel_stimtimes_c + 500;
                    for tr_ind = 1:length(tr_vec)
%                         inner_cnt = inner_cnt + 1;
                        spk_times_sel = spk_times_sel_sec(spk_times_sel_sec >= st_time & ...
                                                    spk_times_sel_sec <= end_time & ...
                                                    trials_gostop_sec == tr_vec(tr_ind));
%                         [cnt_temp,~] = PSTH_mov_win_fast(spk_times_sel,...
%                             win_width,overlap,st_time,end_time,numtrs,num_samples,1);
                        
                        spk_times_sel_sort = sort(spk_times_sel);
                        ISI = diff(spk_times_sel_sort);
                        no_fr = find(ISI>=disinh_width,1);
                        if isempty(no_fr)
                            no_fr_time = NaN;
                        else
                            no_fr_time = spk_times_sel_sort(no_fr) - sel_stimtimes(st_id);
                        end
%                         cnt(inner_cnt,:) = cnt_temp;
%                         cnt = [cnt;cnt_temp];
                        off_time = [off_time;no_fr_time];
                        
                        spk_times_sel = spk_times_c(spk_times_c >= st_time_c & ...
                                                    spk_times_c <= end_time_c & ...
                                                    trials_go == tr_vec(tr_ind));
%                         [cnt_str_temp,~] = PSTH_mov_win_fast(spk_times_sel,...
%                             win_width,overlap,st_time_c,end_time_c,numtrs,num_samples,1);
% %                         cnt_str(inner_cnt,:) = cnt_str_temp;
%                         cnt_str = [cnt_str;cnt_str_temp];

                        spk_times_sel_sort = sort(spk_times_sel);
                        ISI = diff(spk_times_sel_sort);
                        no_fr = find(ISI>=disinh_width,1);
                        if isempty(no_fr)
                            no_fr_time = NaN;
                        else
                            no_fr_time = spk_times_sel_sort(no_fr) - sel_stimtimes_c;
                        end
                        off_time_str = [off_time_str;no_fr_time];
                        
%                         params(inner_cnt,:) = [stn_f(stnf_ind),...
%                                           rel_time,...
%                                           str_freq,...
%                                           double(tr_vec(tr_ind))];
                        params = [params;[stn_f(stnf_ind),...
                                          rel_time,...
                                          str_freq,...
                                          double(tr_vec(tr_ind))]];
                    end
                end
                
            end
            toc
        end
        sel_stimtimes = stimtimes(...
                                stimrate(1,:) == str_f(end) & ...
                                stimrate(2,:) == stn_f(end));
%         cnttimes = cnttimes(1,:) - sel_stimtimes(end);
        
        save([fig_dir,'avg_fr_data_eachtr_ISI'],...
             'params','off_time','off_time_str',...
             'numunits','-v7.3')
    end
%     save([fig_dir,'diffmat-tr',num2str(trial)],'all_diff_struc')
end

function [ids,spktimes] = spk_id_time_ex(dir)
    data = load(dir);
    IDs = double(data.N_ids);
    ids = IDs - min(IDs);
    spktimes = double(data.spk_times)/10;
%     spk_times_d = double(spk_times);
%    numunits = max(ids) - min(ids) + 1;
end
function [stim_ids,stim_spktimes] = spk_id_time_subpop_ex(subpop_ids,ids,spk_times)
    stim_ids = [];
    stim_spktimes = [];
    for sp_ind = 1:length(subpop_ids)
        stim_ids = [stim_ids;ids(ids == subpop_ids(sp_ind))];
        stim_spktimes = [stim_spktimes;spk_times(ids == subpop_ids(sp_ind))];
    end
end
function renumbered = ids_renum_for_raster(IDs)
    IDs_u = unique(IDs);
    renumbered = IDs;
    for ind = 1:length(IDs_u)
        renumbered(renumbered == IDs_u(ind)) = ind;
    end
end
function [] = silent_snr_id(ids,spk_times,fig_dir)
    ids_u = unique(ids);
    for id_ind = 1:length(ids_u)
        spks_in_id = sort(spk_times(ids==ids_u(id_ind)));
        figure;
        histogram(diff(spks_in_id),[0:10:100])
        GCA = gca;
        GCA.FontSize = 14;
        xlabel('ISI (ms)')
        ylabel('Counts')
        title(['SNr unit # ',num2str(ids_u(id_ind))])
        histdir = [fig_dir,'ISIhist/'];
        if exist(histdir,'dir') ~= 7
            mkdir(histdir)
        end
        fig_print(gcf,[histdir,'ISIhist-SNr-',num2str(ids_u(id_ind))])
        close(gcf)
    end
end
function [] = nuclei_fr_hist(nuclei,fig_dir)
    for nc_id = 1:length(nuclei)
        [IDs,spk_times] = spk_id_time_ex([data_dir,'mat_data/',nuclei{nc_id},'-spikedata']);
        IDs_u = unique(IDs);
        firingrates = zeros(2,length(IDs_u));

        for id_ind = 1:length(IDs_u)
            firingrates(1,id_ind) = IDs(id_ind);
            firingrates(2,id_ind) = sum(IDs == IDs_u(id_ind))/max(spk_times)*1000;
        end

        SUFR(nc_id).nc_name = nuclei(nc_id);
        SUFR(nc_id).fr_ids = firingrates;
        figure;
        histogram(firingrates(2,:),20)
        GCA = gca;
        GCA.FontSize = 14;
        GCA.Box = 'off';
        GCA.TickDir = 'out';
        histdir = [fig_dir,'ISIhist/'];
        fig_print(gcf,[histdir,'hist-',nuclei{nc_id}])
        close(gcf)  
    end
end
