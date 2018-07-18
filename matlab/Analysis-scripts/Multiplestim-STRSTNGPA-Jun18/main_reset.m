%% Date created 01.04.18 by M. Mohagheghi

% This script analyzes spike times from different nuclei simulated
% separately. This separate simulation re-establishes all simulation
% condition where we can make sure that the network state for each
% stimulation is similar.

function [] = main_reset(data_path)

    % Parallelization
    
%     parpool('local',str2double(getenv('MOAB_PROCCOUNT')))

    % Averaging window

    win_width = 10; %ms
    overlap = 1;    %ms

    win_width_no_ov = 1;    %ms

    avg_st = -1000;
    avg_end = 1000;

    t_samples_no_ov = (avg_st + win_width_no_ov/2):(avg_end - win_width_no_ov/2);


    numtrs = 100;
    num_ncs = 8;
    nrows = numtrs*num_ncs;
    
    compress_flag = true;
    matfiles = matfiles_extract(data_path);
    if exist(fullfile(data_path,'AllProcData'),'dir') ~= 7
        disp('Concatenated data does not exist!')
        
%         [st_par_ssg, nctr_ssg, avgfr_ssg, avgfrnoov_ssg, offt_ssg, stimISI_ssg, ...
%          nctrISI_ssg, nu_ssg] = ...
%         variables_init(length(matfiles), numtrs, num_ncs, length(t_samples_no_ov));
        
%         [st_par_ss, nctr_ss, avgfr_ss, avgfrnoov_ss, offt_ss, stimISI_ss, ...
%          nctrISI_ss, nu_ss] = ...
%         variables_init(length(matfiles), numtrs, num_ncs, length(t_samples_no_ov));
%     
%         [st_par_s, nctr_s, avgfr_s, avgfrnoov_s, offt_s, stimISI_s, ...
%          nctrISI_s, nu_s] = ...
%         variables_init(length(matfiles), numtrs, num_ncs, length(t_samples_no_ov));

        [st_par, nctr, avgfr, avgfrnoov, offt, stimISI, ...
         nctrISI, nu] = ...
        variables_init(length(matfiles), numtrs, num_ncs, length(t_samples_no_ov));        
            
        st_par_ssg  = []; st_par_ss  = []; st_par_s     = [];
        nctr_ssg    = []; nctr_ss    = []; nctr_s      = [];
        avgfr_ssg   = []; avgfr_ss   = []; avgfr_s     = [];
        nu_ssg      = []; nu_ss      = []; nu_s        = [];
        offt_ssg    = []; offt_ss    = []; offt_s      = [];
        stimISI_ssg = []; stimISI_ss = []; stimISI_s   = [];
        nctrISI_ssg = []; nctrISI_ss = []; nctrISI_s   = [];
        nc_ssg      = []; nc_ss      = []; nc_s        = [];
        avgfrnoov_ssg = []; avgfrnoov_ss = []; avgfrnoov_s = [];
        t_sample_noov_ssg = []; t_sample_noov_ss = []; t_sample_noov_s = [];
        
        loc_ind_ssg = 0; loc_ind_ss = 0; loc_ind_s = 0;
        for fl_ind = 1:length(matfiles)
            
            disp(['Progress percent: ',num2str(fl_ind/length(matfiles)*100,'%.2f'), ' %'])
            
            [st_par_tmp,nctr_tmp,avgfr,...
             avgfrnoov_tmp,offt_tmp,stimISI_tmp,...
             nctrISI_tmp,nc_tmp,t_sample_noov,...
             nu_tmp,sim_type] = main_postproc(fullfile(data_path,matfiles{fl_ind}),compress_flag);
         
            NC = nc_tmp;
            
            IND1 = loc_ind_ssg*nrows + 1;
            IND1_I = loc_ind_ssg*numtrs + 1;
            loc_ind_ssg = loc_ind_ssg + 1;
            IND2 = loc_ind_ssg*nrows;
            IND2_I = loc_ind_ssg*numtrs;

            st_par(IND1:IND2,:) = st_par_tmp;
            nctr(IND1:IND2,:) = nctr_tmp;
%                 avgfr_ssg(IND1:IND2,:) = avgfr_tmp;
            nu(IND1:IND2,:) = nu_tmp;
            offt(IND1_I:IND2_I) = offt_tmp;
            stimISI(IND1_I:IND2_I,:) = stimISI_tmp;
            nctrISI(IND1_I:IND2_I,:) = nctrISI_tmp;
            avgfrnoov(IND1:IND2,:) = avgfrnoov_tmp;

        end
        
        
        
        disp('Scenario STRSTNGPA')
        
        sim_type = 'STRSTNGPA';
        
        [st_par_ssg, nctr_ssg, nu_ssg, offt_ssg, stimISI_ssg, nctrISI_ssg, avgfrnoov_ssg] = ...
            var_discrimination(st_par, nctr, nu, offt, stimISI, nctrISI, avgfrnoov, sim_type);
        
        savematfile(st_par_ssg, nctr_ssg, avgfr_ssg, avgfrnoov_ssg, ...
                    offt_ssg, stimISI_ssg, nctrISI_ssg, t_sample_noov, ...
                    nu_ssg, NC, data_path, matfiles{fl_ind}, sim_type)
%       

        disp('Scenario STRSTN')
        
        sim_type = 'STRSTN';
        
        [st_par_ss, nctr_ss, nu_ss, offt_ss, stimISI_ss, nctrISI_ss, avgfrnoov_ss] = ...
            var_discrimination(st_par, nctr, nu, offt, stimISI, nctrISI, avgfrnoov, sim_type);
        
        disp('Scenario STRSTN')
        savematfile(st_par_ss(:,[2,3,5]), nctr_ss, avgfr_ss, avgfrnoov_ss, ...
                    offt_ss, stimISI_ss(:,[2,3,5]), nctrISI_ss, t_sample_noov, ...
                    nu_ss, NC, data_path, matfiles{fl_ind}, 'STRSTN')
                
        disp('Scenario STR')
        
        sim_type = 'STR';
        
        [st_par_s, nctr_s, nu_s, offt_s, stimISI_s, nctrISI_s, avgfrnoov_s] = ...
            var_discrimination(st_par, nctr, nu, offt, stimISI, nctrISI, avgfrnoov, sim_type);
        
        disp('Scenario STR')
        savematfile(st_par_s(:,2), nctr_s, avgfr_s, avgfrnoov_s, ...
                    offt_s, stimISI_s(:,2), nctrISI_s, t_sample_noov, ...
                    nu_s, NC, data_path, matfiles{fl_ind}, 'STR')
        
    else
        disp('Concatenated data exists!')
%         indir = dir(data_path);
%         for fl_ind = 1:length(matfiles)
% %             mat_flname = '/procdata_avg_ISI.mat';
%             
%             [st_par,nctr,avgfr,...
%              avgfrnoov,offt,stimISI,...
%              nctrISI,nc,t_sample_noov,nu] = main_postproc(fullfile(data_path,matfiles{fl_ind}),weights,numtrs,compress_flag);       
% 
%             procdata{fl_ind} = struct('stim_param',st_par,...
%                                        'nuclei_trials',nctr,...
%                                        'average_fr',avgfr,...
%                                        'average_fr_no_overlap',avgfrnoov,...
%                                        'offtime',offt,...
%                                        'stim_param_ISI',stimISI,...
%                                        'nuclei_trials_ISI',nctrISI,...
%                                        'time_vec',t_sample_noov,...
%                                        'num_units',nu,...
%                                        'nuclei',struct('nc_names',nc),...
%                                        'data_dir',fullfile(data_path,'/',matfiles{fl_ind}));
%         end
%         save(fullfile(data_path,'all_proc_data'),'procdata','nc','-v7.3')
    end
end

function [stim_pars,nc_trs,avg_frs,...
          avg_frs_no_ov,off_time,...
          stim_pars_ISI,nc_trs_ISI,...
          nuclei,t_samples_no_ov,num_units,...
          sim_paradigm] = main_postproc(data_dir, comp_flag)  
    
    disp(['Loading file ',data_dir])
    file_path = fullfile(data_dir,'procdata_avg_ISI.mat');
    if exist(file_path,'file') ~= 2
%         disp('Data for this directory does not exist!')

        % Averaging window

        win_width = 10; %ms
        overlap = 1;    %ms

        win_width_no_ov = 1;    %ms

        % Averaging start and end times

        avg_st = -1000;
        avg_end = 1000;
        
        t_samples_no_ov = (avg_st + win_width_no_ov/2):(avg_end - win_width_no_ov/2);


        numtrs = 100;
        num_ncs = 8;
%         nrows = numtrs*num_ncs;
        
        [stim_pars, nc_trs, avg_frs, avg_frs_no_ov, off_time, stim_pars_ISI, ...
         nc_trs_ISI, num_units, ] = ...
        variables_init([], numtrs, num_ncs, length(t_samples_no_ov));

%         stim_pars = [];
%         nc_trs = [];
        avg_frs = [];
%         avg_frs_no_ov = [];
%         off_time = [];
%         stim_pars_ISI = [];
%         nc_trs_ISI = [];
%         num_units = [];

        raw_data = load(data_dir);
        
        nuclei  = raw_data.nuclei;
        weight  = raw_data.data{1}.weight;
        rates   = raw_data.data{1}.rates;
        data    = raw_data.data;
        
%         maxuint08 = intmax('uint8');
%         maxuint16 = intmax('uint16');
%         maxuint32 = intmax('uint32');
        
        
        if length(rates) == 1
            sim_paradigm = 'STR';
        elseif length(rates) == 5
            if sum(rates==0) == 4
                sim_paradigm = 'STR';
            elseif sum(rates==0) == 0
                sim_paradigm = 'STRSTNGPA';
            else
%                 if rates(1) == 0 && rates(4) == 0
                if rates(1) == 0
                    sim_paradigm = 'STRSTN';
%                 elseif rates(3) == 0 && rates(5) == 0
                elseif rates(3) == 0
                    sim_paradigm = 'STRGPA';
                elseif rates(1) == 0 && rates(3) == 0
                    sim_paradigm = 'STR';
                else
                    disp(['Weird!, data path: ', data_dir,...
                          '; rates vector: ', rates])
                end
            end
        end
        
%         for w_ind = 1:length(weights)

%             disp(['Processing data for weight ',num2str(weights(w_ind))])

        for tr_ind = 1:size(data,2)

            for nc_ind = 1:length(nuclei)
%             for nc_ind = 1:length(nuclei)

%                 for st_ind = 1:size(data,2)

%                         disp(['w',num2str(w_ind),'-st',num2str(st_ind),'-nc',num2str(nc_ind),'-tr',num2str(tr_ind)])

                spk_data = getfield(data{nc_ind,tr_ind},nuclei{nc_ind});
                spktimes = double(spk_data.spktimes)/10;

                if ~isempty(spktimes)
                    N_ids = double(spk_data.N_ids);

%                     stim_pars = [stim_pars;rates];
                    stim_pars((tr_ind-1)*num_ncs + nc_ind, :) = rates;
%                     nc_trs = [nc_trs;[nc_ind,tr_ind,weight]];
                    nc_trs((tr_ind-1)*num_ncs + nc_ind, :) = [nc_ind,tr_ind,weight];

                    reftime = data{nc_ind,tr_ind}.timerefs.STRstim(tr_ind);

                    % Average firing rate of single trials

                    if comp_flag

%                                 t_samples_no_ov = (avg_st + win_width_no_ov/2):(avg_end - win_width_no_ov/2);
                        [cnttmp,numu] = average_firingrate_hist(spktimes,reftime,N_ids,win_width_no_ov,...
                                                         avg_st,avg_end);
                                                     
%                          if max(cnttmp(:)) > maxuint16
%                              i_cnttmp = uint32(cnttmp);
%                              disp('uint16 format cannot store the data.')
%                              disp('uint32 is used instead.')
%                              avg_frs_no_ov = uint32(avg_frs_no_ov);
%                          else
                             i_cnttmp = uint32(cnttmp);
%                          end

%                         avg_frs_no_ov = [avg_frs_no_ov;i_cnttmp];
                        avg_frs_no_ov((tr_ind-1)*num_ncs + nc_ind,:) = i_cnttmp;
%                         num_units = [num_units;numu];
                        num_units((tr_ind-1)*num_ncs + nc_ind) = numu;

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

%                         off_time = [off_time;off_time_tmp];
                        off_time(tr_ind) = off_time_tmp;

%                         stim_pars_ISI = [stim_pars_ISI;rates];
                        stim_pars_ISI(tr_ind, :) = rates;
%                         nc_trs_ISI = [nc_trs_ISI;[nc_ind,tr_ind,weight]];
                        nc_trs_ISI(tr_ind, :) = [nc_ind,tr_ind,weight];

                    end
                else
%                             disp('Weird!')
                    disp(['Path: ',data_dir])
                    disp(['Nuclei: ',nuclei{nc_ind},' - Trial: ',num2str(tr_ind)])
                end

%                 end
%                 data_nc = data
            end

        end

%         end

        disp('finished')
%         save(file_path,'off_time','stim_pars_ISI','nc_trs_ISI',...
%                                            'avg_frs','stim_pars','nc_trs',...
%                                            'avg_frs_no_ov','t_samples_no_ov',...
%                                            'nuclei','num_units','-v7.3')

    else
        disp('Data for this directory exist!')
%         load(file_path)
    end

end

function cnt = average_firingrate(spk_times,reftime,N_ids,win_width,...
                                    overlap,avg_st,avg_end)

    numunits = max(N_ids) - min(N_ids) + 1;

    [cnt,~] = PSTH_mov_win(spk_times-reftime,...
                           win_width,overlap,avg_st,avg_end,numunits,1);
end

function [cnt,numunits] = average_firingrate_hist(spk_times,reftime,N_ids,win_width,...
                                       avg_st,avg_end)

    numunits = max(N_ids) - min(N_ids) + 1;
    spk_times = spk_times - reftime;
    spk_times_sel = spk_times(spk_times >= avg_st & spk_times <= avg_end);
    hist_edges = avg_st:win_width:avg_end;
    cnt = histcounts(spk_times_sel,hist_edges);
    cnt = cnt/win_width*1000;
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

function dirs = directory_extract(data_path)
    indir = dir(data_path);
    cnt_tmp = 0;
    for ind = 1:length(indir)
        name_str = indir(ind).name;
        if indir(ind).isdir && ~strcmpi(name_str,'.') && ~strcmpi(name_str,'..') && ...
           ~strcmpi(name_str,'Figs') && ~strcmpi(name_str,'Figs2') && ~strcmpi(name_str,'Figsold')
            cnt_tmp = cnt_tmp + 1;
            dirs{cnt_tmp} = name_str;
        end
    end
end

function mat_files = matfiles_extract(data_path)
    filesindir = what(data_path);
    mat_files = filesindir.mat;
end

function [st_par_, nctr_, nu_, offt_, stimISI_, nctrISI_, avgfrnoov_] = ...
         var_discrimination(st_par, nctr, nu, offt, stimISI, nctrISI, avgfrnoov, sim_paradigm)

    switch sim_paradigm
        case 'STRSTN'
            IND = (st_par(:,1) == 0) & (st_par(:,3) ~= 0);
            INDISI = (stimISI(:,1) == 0) & (stimISI(:,3) ~= 0);
        case 'STR'
            IND = (st_par(:,1) == 0) & (st_par(:,3) == 0);
            INDISI = (stimISI(:,1) == 0) & (stimISI(:,3) == 0);
        case 'STRSTNGPA'
            IND = (st_par(:,1) ~= 0) & (st_par(:,3) ~= 0);
            INDISI = (stimISI(:,1) ~= 0) & (stimISI(:,3) ~= 0);
    end
    
    st_par_ = st_par(IND, :);
    nctr_ = nctr(IND, :);
    nu_ = nu(IND, :);
    offt_ = offt(INDISI, :);
    avgfrnoov_ = avgfrnoov(IND, :);
    stimISI_ = stimISI(INDISI,:);
    nctrISI_ = nctrISI(INDISI,:);
 
end

function [] = savematfile(st_par, nctr, avgfr, avgfrnoov, offt, stimISI, ...
                          nctrISI, t_sample_noov, nu, nc, ...
                          data_path, matfiles, sim_type)
    conc_data_dir = 'AllProcData';
    procdata = struct('stim_param',st_par,...
                       'nuclei_trials',nctr,...
                       'average_fr',avgfr,...
                       'average_fr_no_overlap',avgfrnoov,...
                       'offtime',offt,...
                       'stim_param_ISI',stimISI,...
                       'nuclei_trials_ISI',nctrISI,...
                       'time_vec',t_sample_noov,...
                       'num_units',nu,...
                       'nuclei',struct('nc_names',nc),...
                       'data_dir',fullfile(data_path,'/',matfiles));
                   
    data_path = fullfile(data_path,conc_data_dir);
    if exist(data_path, 'dir') ~= 7
        mkdir(data_path)
    end
    save(fullfile(data_path,['all_proc_data-',sim_type]),'procdata','nc','-v7.3')
end

function [stim_par, nc_tr, avg_fr, avg_fr_no_ov, off_tim, stim_par_ISI, nc_tr_ISI, num_unit] = ...
        variables_init(n_files, numtrs, num_ncs, L_time)
    
        if isempty(n_files)
            nrows = numtrs*num_ncs;
        else
            nrows = n_files*numtrs*num_ncs;
        end
        
        stim_par = zeros(nrows, 5);
        nc_tr = zeros(nrows, 3);
        avg_fr = [];
        avg_fr_no_ov = uint32(zeros(nrows, L_time));
        off_tim = zeros(numtrs, 1);
        stim_par_ISI = zeros(numtrs, 5);
        nc_tr_ISI = zeros(numtrs, 3);
        num_unit = zeros(nrows, 1);
end
