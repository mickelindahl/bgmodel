%% Date created 01.04.18 by M. Mohagheghi

% This script analyzes spike times from different nuclei simulated
% separately. This separate simulation re-establishes all simulation
% condition where we can make sure that the network state for each
% stimulation is similar.

function [] = main(data_path,weights,numtrs)
    compress_flag = true;
    data_dir = directory_extract(data_path);
    if exist(fullfile(data_path,'all_proc_data.mat'),'file') ~= 2
        for dir_ind = 1:length(data_dir)

            [st_par,nctr,avgfr,...
             avgfrnoov,offt,stimISI,...
             nctrISI,nc,t_sample_noov] = main_postproc(fullfile(data_path,data_dir{dir_ind}),weights,numtrs,compress_flag);

            procdata{dir_ind} = struct('stim_param',st_par,...
                                       'nuclei_trials',nctr,...
                                       'average_fr',avgfr,...
                                       'average_fr_no_overlap',avgfrnoov,...
                                       'offtime',offt,...
                                       'stim_param_ISI',stimISI,...
                                       'nuclei_trials_ISI',nctrISI,...
                                       'time_vec',t_sample_noov,...
                                       'nuclei',struct('nc_names',nc),...
                                       'data_dir',fullfile(data_path,'/',data_dir{dir_ind}));

<<<<<<< HEAD
    for dir_ind = 1:length(data_dir)
        
        [st_par,nctr,avgfr,...
         avgfrnoov,offt,stimISI,...
         nctrISI,nc] = main_postproc([data_path,'/',data_dir{dir_ind}],weights,numtrs);
     
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
    save([data_path,'/all_proc_data'],'procdata','-v7.3')
=======
        end
        save(fullfile(data_path,'all_proc_data'),'procdata','nc')
    else
%         indir = dir(data_path);
        for dir_ind = 1:length(data_dir)
%             mat_flname = '/procdata_avg_ISI.mat';
            [st_par,nctr,avgfr,...
             avgfrnoov,offt,stimISI,...
             nctrISI,nc] = main_postproc(fullfile(data_path,data_dir{dir_ind}),weights,numtrs,compress_flag);

            procdata{dir_ind} = struct('stim_param',st_par,...
                                       'nuclei_trials',nctr,...
                                       'average_fr',avgfr,...
                                       'average_fr_no_overlap',avgfrnoov,...
                                       'offtime',offt,...
                                       'stim_param_ISI',stimISI,...
                                       'nuclei_trials_ISI',nctrISI,...
                                       'nuclei',struct('nc_names',nc),...
                                       'data_dir',fullfile(data_path,data_dir{dir_ind}));
        end
        save(fullfile(data_path,'all_proc_data'),'procdata')
    end
>>>>>>> cb501857ce49af1432da7228dcc0701477a10c06
end

function [stim_pars,nc_trs,avg_frs,...
          avg_frs_no_ov,off_time,...
          stim_pars_ISI,nc_trs_ISI,...
<<<<<<< HEAD
          nuclei] = main_postproc(data_dir, weights, numtrs)
    
    stim_pars = [];
    nc_trs = [];
    avg_frs = [];
    avg_frs_no_ov = [];
    off_time = [];
    stim_pars_ISI = [];
    nc_trs_ISI = [];
    
    % Averaging window 

    win_width = 10; %ms
    overlap = 1;    %ms

    win_width_no_ov = 1;    %ms

    % Averaging start and end times

    avg_st = -500;
    avg_end = 500;
    
    t_samples = (avg_st + win_width/2):overlap:(avg_end - win_width/2);
    t_samples_no_ov = (avg_st + win_width_no_ov/2):(avg_end - win_width_no_ov/2);
    

    for w_ind = 1:length(weights)
        disp(['W = ',num2str(weights(w_ind))])
        tic
        for tr_ind = 1:numtrs
            
            data = load([data_dir,'W',num2str(weights(w_ind)*100,2),...
                                 '-tr',num2str(tr_ind,2)]);
                             
            nuclei = data.nuclei;
            
            data = data.data;
            
            parfor nc_ind = 1:length(nuclei)
                
                for st_ind = 1:size(data,2)
                    
                    %disp(['w',num2str(w_ind),'-st',num2str(st_ind),'-nc',num2str(nc_ind),'-tr',num2str(tr_ind)])
                    
                    spk_data = getfield(data{nc_ind,st_ind},nuclei{nc_ind});
                    spktimes = double(spk_data.spktimes)/10;
                    
                    if ~isempty(spktimes)
                        N_ids = double(spk_data.N_ids);
=======
          nuclei,t_samples_no_ov] = main_postproc(data_dir, weights, numtrs, comp_flag)
    file_path = fullfile(data_dir,'procdata_avg_ISI.mat');
    if exist(file_path,'file') ~= 2
      
        stim_pars = [];
        nc_trs = [];
        avg_frs = [];
        avg_frs_no_ov = [];
        off_time = [];
        stim_pars_ISI = [];
        nc_trs_ISI = [];
        
        % Averaging window 

        win_width = 10; %ms
        overlap = 1;    %ms

        win_width_no_ov = 1;    %ms

        % Averaging start and end times

        avg_st = -500;
        avg_end = 500;

        
        t_samples_no_ov = (avg_st + win_width_no_ov/2):(avg_end - win_width_no_ov/2);


        for w_ind = 1:length(weights)

            for tr_ind = 1:numtrs

                data = load(fullfile(data_dir,['W',num2str(weights(w_ind)*100,2),...
                                     '-tr',num2str(tr_ind,2)]));

                nuclei = data.nuclei;
>>>>>>> cb501857ce49af1432da7228dcc0701477a10c06

                data = data.data;

                parfor nc_ind = 1:length(nuclei)

<<<<<<< HEAD
                        % Average firing rate of single trials

                        
                        cnttmp = average_firingrate(spktimes,reftime,N_ids,win_width,...
                                                   overlap,avg_st,avg_end);
                        avg_frs = [avg_frs;cnttmp];

                        
                        cnttmp = average_firingrate_hist(spktimes,reftime,N_ids,win_width_no_ov,...
                                                         avg_st,avg_end);
=======
                    for st_ind = 1:size(data,2)

%                         disp(['w',num2str(w_ind),'-st',num2str(st_ind),'-nc',num2str(nc_ind),'-tr',num2str(tr_ind)])

                        spk_data = getfield(data{nc_ind,st_ind},nuclei{nc_ind});
                        spktimes = double(spk_data.spktimes)/10;

                        if ~isempty(spktimes)
                            N_ids = double(spk_data.N_ids);

                            stim_pars = [stim_pars;data{nc_ind,st_ind}.rates];
                            nc_trs = [nc_trs;[nc_ind,tr_ind,weights(w_ind)]];

                            reftime = data{nc_ind,st_ind}.timerefs.STRstim;

                            % Average firing rate of single trials

                            if comp_flag
>>>>>>> cb501857ce49af1432da7228dcc0701477a10c06

                                t_samples_no_ov = (avg_st + win_width_no_ov/2):(avg_end - win_width_no_ov/2);
                                cnttmp = average_firingrate_hist(spktimes,reftime,N_ids,win_width_no_ov,...
                                                                 avg_st,avg_end);

                                avg_frs_no_ov = [avg_frs_no_ov;uint8(cnttmp)];

                            else

                                t_samples = (avg_st + win_width/2):overlap:(avg_end - win_width/2);
                                cnttmp = average_firingrate(spktimes,reftime,N_ids,win_width,...
                                                           overlap,avg_st,avg_end);
                                avg_frs = [avg_frs;cnttmp];

                                
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
                                nc_trs_ISI = [nc_trs_ISI;[nc_ind,tr_ind,weights(w_ind)]];

                            end
                        end

                    end
    %                 data_nc = data
                end

            end
<<<<<<< HEAD
            
            
        end
        toc
    end
    
    disp('finished')
    save([data_dir,'procdata_avg_ISI'],'off_time','stim_pars_ISI','nc_trs_ISI',...
                                       'avg_frs','stim_pars','nc_trs','t_samples',...
                                       'avg_frs_no_ov','t_samples_no_ov',...
                                       'nuclei','-v7.3')
=======

        end

        disp('finished')
        save(file_path,'off_time','stim_pars_ISI','nc_trs_ISI',...
                                           'avg_frs','stim_pars','nc_trs',...
                                           'avg_frs_no_ov','t_samples_no_ov',...
                                           'nuclei')
                                       
    else
        load(file_path)
    end
>>>>>>> cb501857ce49af1432da7228dcc0701477a10c06

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
<<<<<<< HEAD
=======

function dirs = directory_extract(data_path)
    indir = dir(data_path);
    cnt_tmp = 0;
    for ind = 1:length(indir)
        name_str = indir(ind).name;
        if indir(ind).isdir && ~strcmpi(name_str,'.') && ~strcmpi(name_str,'..')
            cnt_tmp = cnt_tmp + 1;
            dirs{cnt_tmp} = name_str;
        end
    end
end
>>>>>>> cb501857ce49af1432da7228dcc0701477a10c06
