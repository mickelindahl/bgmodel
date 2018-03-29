%% Analysis of spiking data

% The purpose of this file is to compute average firing rates of SNr neurons 
% and put it in a matrix which further can be used to compute the latency
% difference after STN stimulation (Stop) with respect to Go experiment.
% Modified 04.01.18: Because the structure of the data file this file uses
% is changed I changed the file accordingly.

function [] = main_avgfrs_multistim_eachtr_newtrid_ISI(data_dir,res_dir)
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
    
    params_string = {'STN','Reltime','STR','Trial'};

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
        data.gostop = rmfield(data.gostop,'spk_times');
    %     spk_times_d = double(spk_times);
%         numunits = max(IDs) - min(IDs) + 1;
        
        %% Data of stop signal
        
        spk_times_c = double(data.go.spk_times)/10;
        trials_go = data.go.trial_ids;
        
%         tr_start = min(trials_go);
%         tr_end = max(trials_go);
        tr_vec = 1:size(trials_go,1);
        
        clear data
        
        % Matrices initialization
        t_samples = (avg_st + win_width/2):overlap:(avg_end - win_width/2);
        off_time = zeros(length(stn_f)*length(str_stn_lat)*length(str_f)*length(tr_vec),1);
        off_time_str = zeros(size(off_time));
        params = zeros(size(off_time,1),length(params_string));
        
        num_samples = length(t_samples);
        

        for stnf_ind = 1:length(stn_f)
            disp(['Stim STN = ',num2str(stn_f(stnf_ind))])
            h_sim_time = max(stimtimes(stimrate(2,:) == stn_f(stnf_ind)));
            l_sim_time = min(stimtimes(stimrate(2,:) == stn_f(stnf_ind)));
            spk_times_sel_init = spk_times(spk_times >= l_sim_time - 1000 &...
                                            spk_times <= h_sim_time + 1000);
            trials_gostop_sel = trials_gostop(spk_times >= l_sim_time - 1000 &...
                                            spk_times <= h_sim_time + 1000);
            for rel_time_ind = 1:length(str_stn_lat)
                disp(['Rel time = ',num2str(str_stn_lat(rel_time_ind))])
                sel_stimtimes = stimtimes(...
                                stimrate(3,:) == str_stn_lat(rel_time_ind) & ...
                                stimrate(2,:) == stn_f(stnf_ind));

                for st_id = 1:length(sel_stimtimes)
                    str_freq = stimrate(1,stimtimes == sel_stimtimes(st_id));
                    disp([nuclei{nc_id},'-',num2str(st_id)])
                    st_time = sel_stimtimes(st_id) - 500;
                    end_time = sel_stimtimes(st_id) + 500;
                    for tr_ind = 1:length(tr_vec)
                        inner_cnt = inner_cnt + 1;
                        spk_times_sel = spk_times_sel_init(spk_times_sel_init >= st_time & ...
                                                    spk_times_sel_init <= end_time & ...
                                                    trials_gostop_sel == tr_vec(tr_ind));
%                         [cnt_temp,~] = PSTH_mov_win_fast(spk_times_sel,...
%                             win_width,overlap,st_time,end_time,numtrs,num_samples,1);
                        spk_times_sel_sort = sort(spk_times_sel);
                        ISI = diff(spk_times_sel_sort);
                        no_fr = find(ISI>=win_width,1);
                        if isempty(no_fr)
                            no_fr_time = NaN;
                        else
                            no_fr_time = spk_times_sel_sort(no_fr);
                        end
                        
                        off_time(inner_cnt,:) = no_fr_time;

                        sel_stimtimes_c = stimtimes_c(stimrate_c == str_freq);
                        st_time = sel_stimtimes_c - 500;
                        end_time = sel_stimtimes_c + 500;
                        spk_times_sel = spk_times_c(spk_times_c >= st_time & ...
                                                    spk_times_c <= end_time & ...
                                                    trials_go == tr_vec(tr_ind));
%                         [cnt_str_temp,~] = PSTH_mov_win_fast(spk_times_sel,...
%                             win_width,overlap,st_time,end_time,numtrs,num_samples,1);

                        spk_times_sel_sort = sort(spk_times_sel);
                        ISI = diff(spk_times_sel_sort);
                        no_fr = find(ISI>=win_width,1);
                        if isempty(no_fr)
                            no_fr_time = NaN;
                        else
                            no_fr_time = spk_times_sel_sort(no_fr);
                        end
                        off_time_str(inner_cnt) = no_fr_time;
                        
                        params(inner_cnt,:) = [stn_f(stnf_ind),...
                                          str_stn_lat(rel_time_ind),...
                                          str_freq,...
                                          double(tr_vec(tr_ind))];
                    end
                end
                
            end
        end
        cnttimes = cnttimes(1,:) - sel_stimtimes(st_id);
        
        save([fig_dir,'SNr_silencetime_data_eachtr'],...
             'params','off_time_str','off_time',...
             'numunits','params_string')
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
