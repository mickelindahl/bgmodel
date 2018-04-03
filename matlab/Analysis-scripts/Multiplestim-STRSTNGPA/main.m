%% Date created 01.04.18 by M. Mohagheghi

% This script analyzes spike times from different nuclei simulated
% separately. This separate simulation re-establishes all simulation
% condition where we can make sure that the network state for each
% stimulation is similar.

function [] = main(data_path,data_dir,weights,numtrs)
    compress_flag = true;
    for dir_ind = 1:length(data_dir)
        
        [st_par,nctr,avgfr,...
         avgfrnoov,offt,stimISI,...
         nctrISI,nc] = main_postproc([data_path,'/',data_dir{dir_ind}],weights,numtrs,compress_flag);
     
        procdata{dir_ind} = struct('stim_param',st_par,...
                                   'nuclei_trials',nctr,...
                                   'average_fr',avgfr,...
                                   'average_fr_no_overlap',avgfrnoov,...
                                   'offtime',offt,...
                                   'stim_param_ISI',stimISI,...
                                   'nuclei_trials_ISI',nctrISI,...
                                   'nuclei',struct('nc_names',nc),...
                                   'data_dir',[data_path,'/',data_dir{dir_ind}]);
                               
    end
    save([data_path,'all_proc_data'],'procdata')
end

function [stim_pars,nc_trs,avg_frs,...
          avg_frs_no_ov,off_time,...
          stim_pars_ISI,nc_trs_ISI,...
          nuclei] = main_postproc(data_dir, weights, numtrs, comp_flag)
    
    stim_pars = [];
    nc_trs = [];
    avg_frs = [];
    avg_frs_no_ov = [];
    off_time = [];
    stim_pars_ISI = [];
    nc_trs_ISI = [];
    

    for w_ind = 1:length(weights)
        
        for tr_ind = 1:numtrs
            
            data = load([data_dir,'W',num2str(weights(w_ind)*100,2),...
                                 '-tr',num2str(tr_ind,2)]);
                             
            nuclei = data.nuclei;
            
            data = data.data;
            
            for nc_ind = 1:length(nuclei)
                
                for st_ind = 1:size(data,2)
                    
                    disp(['w',num2str(w_ind),'-st',num2str(st_ind),'-nc',num2str(nc_ind),'-tr',num2str(tr_ind)])
                    
                    spk_data = getfield(data{nc_ind,st_ind},nuclei{nc_ind});
                    spktimes = double(spk_data.spktimes)/10;
                    
                    if ~isempty(spktimes)
                        N_ids = double(spk_data.N_ids);

                        stim_pars = [stim_pars;data{nc_ind,st_ind}.rates];
                        nc_trs = [nc_trs;[nc_ind,tr_ind]];

                        reftime = data{nc_ind,st_ind}.timerefs.STRstim;

                        % Averaging window 

                        win_width = 10; %ms
                        overlap = 1;    %ms

                        win_width_no_ov = 1;    %ms

                        % Averaging start and end times

                        avg_st = -500;
                        avg_end = 500;

                        % Average firing rate of single trials
                        
                        if comp_flag
                            
                            t_samples_no_ov = (avg_st + win_width_no_ov/2):(avg_end - win_width_no_ov/2);
                            cnttmp = average_firingrate_hist(spktimes,reftime,N_ids,win_width_no_ov,...
                                                             avg_st,avg_end);

                            avg_frs_no_ov = [avg_frs_no_ov;uint8(cnttmp)];
                            
                        else

                            t_samples = (avg_st + win_width/2):overlap:(avg_end - win_width/2);
                            cnttmp = average_firingrate(spktimes,reftime,N_ids,win_width,...
                                                       overlap,avg_st,avg_end);
                            avg_frs = [avg_frs;cnttmp];

                            t_samples_no_ov = (avg_st + win_width_no_ov/2):(avg_end - win_width_no_ov/2);
                            cnttmp = average_firingrate_hist(spktimes,reftime,N_ids,win_width_no_ov,...
                                                             avg_st,avg_end);

                            avg_frs_no_ov = [avg_frs_no_ov;cnttmp];
                            
                        end

                        if strcmpi(nuclei{nc_ind},'SN')

                            % Window width for ISI

                            disinh_w = 20;  % ms

                            off_time_tmp = disinh_time_ISI(spktimes,disinh_w,reftime);

                            off_time = [off_time;off_time_tmp];

                            stim_pars_ISI = [stim_pars_ISI;data{nc_ind,st_ind}.rates];
                            nc_trs_ISI = [nc_trs_ISI;[nc_ind,tr_ind]];

                        end
                    end
                    
                end
%                 data_nc = data
            end
            
        end
        
    end
    
    disp('finished')
    save([data_dir,'procdata_avg_ISI'],'off_time','stim_pars_ISI','nc_trs_ISI',...
                                       'avg_frs','stim_pars','nc_trs','t_samples',...
                                       'avg_frs_no_ov','t_samples_no_ov',...
                                       'nuclei')

end

function cnt = average_firingrate(spk_times,reftime,N_ids,win_width,...
                                    overlap,avg_st,avg_end)
    
    numunits = max(N_ids) - min(N_ids) + 1;
    
    [cnt,~] = PSTH_mov_win(spk_times-reftime,...
                           win_width,overlap,avg_st,avg_end,numunits,1);
end

function cnt = average_firingrate_hist(spk_times,reftime,N_ids,win_width,...
                                       avg_st,avg_end)
    
    numunits = max(N_ids) - min(N_ids) + 1;
    spk_times = spk_times - reftime;
    spk_times_sel = spk_times(spk_times >= avg_st & spk_times <= avg_end);
    hist_edges = avg_st:win_width:avg_end;
    cnt = histcounts(spk_times_sel,hist_edges);
    cnt = cnt/win_width*1000/numunits;
end

function no_fr_time = disinh_time_ISI(spk_times,disinh_width,reftime)
    spk_times_sort = sort(spk_times);
    ISI = diff(spk_times_sort);
    no_fr = find(ISI>=disinh_width,1);
    if isempty(no_fr)
        no_fr_time = NaN;
    else
        no_fr_time = spk_times_sort(no_fr) - reftime;
    end
end