%% Created by Mohagheghi on 09.11.17

% This file reads nest data stored in .gdf files in order to apply furhter
% analysis in MATLAB like average firing rate and ...

% Modified on 14.06.18

% This script now concatenates all trials ran for a single parameter
% combination in the simulation where ResetNetwork reinitializes the state
% variables.

function data_concat_save_as_mat_sep_stim_reset(varargin)
    disp(['Data path is: ',varargin])

    nuclei = {'FS','GA','GF','GI','M1','M2','SN','ST'};
    % data_dir = [pwd,'/'];

    if length(varargin) >= 2
        data_dir = varargin{1};
        if isstr(varargin{2})
            res = str2double(varargin{2});
        else
            res = varargin{2};
        end
        submit_dir_fullpath = varargin{3};
    else
        data_dir = ['/Users/Mohammad/Documents/PhD/Projects/BGmodel/bgmodel',...
            '/results/example/eneuro/3000/activation-control/',...
            'GPA-500.0-2000.0--100.0STR-600.0-500.0STN-500.0-2000.0--100.0tr1/'];
        res = 0.1;           %ms
    end


    % Reading data one by one

    % Initialization of empty vars
    % res = 0.1;           %ms
    submitdir = strsplit(submit_dir_fullpath,'/');
    submitdir = submitdir{end};

    mat_datadir = data_dir;
    % nestdir = [data_dir,'nest/'];
    % 
    % stimdata = load([data_dir,'stimspec']);
    % 
    % ref_time = stimdata.STRramp.stop_times;
    % 
    all_dirs_data = load(fullfile(data_dir,'dir-data.mat'));
    all_dirs = all_dirs_data.dirs;
    weight_stim_dir = all_dirs_data.wsdir;
    % trnum = all_dirs_data.tr;
    Gs = [];

    stimdata = load(fullfile(weight_stim_dir,'stimspec'));
    cond_GA_MSN_data = load(fullfile(weight_stim_dir,'modifiedweights'));
    ref_time = stimdata.STRramp.stop_times;

    for nc_ind = 1:length(nuclei)

        for dir_ind = 1:length(ref_time)
    %         nestdir = all_dirs(dir_ind,:);
            nestdir = fullfile(weight_stim_dir,num2str(dir_ind,'%.0f'));
            trnum = dir_ind;
    %         trnum = strsplit(nestdir,'/');
    %         trnum = str2double(trnum{end});

    %         data_dir = strsplit(nestdir,'nest');
    %         data_dir = data_dir{1};        
            GA_M1 = (cond_GA_MSN_data.GA_M1.max + cond_GA_MSN_data.GA_M1.min)/2;
            Gs = [Gs,GA_M1];

            N_ids = [];
            spk_times = [];
            trs = [];
            gdfdir = dir(fullfile(nestdir,[nuclei{nc_ind},'*.gdf']));
            disp(['Reading ',nuclei(nc_ind)])

            for fl_ind = 1:length(gdfdir)

                fl_name = gdfdir(fl_ind).name;
                [ids,spks] = readgdf_sep_stim(fullfile(nestdir,fl_name),res,ref_time(dir_ind));

                N_ids = [N_ids;ids];
                spk_times = [spk_times;spks];
                trs = [trs;uint8(trnum*ones(size(N_ids)))];
            end

            if sum(strcmpi(fieldnames(stimdata),'GPAstop')) == 1
                data{nc_ind,dir_ind} = struct(nuclei{nc_ind},struct('spktimes',spk_times,'N_ids',N_ids),...
                                        'rates',stimdata.STRramp.rates,...
                                        'timerefs',struct('GPAstim',stimdata.GPAstop.start_times,...
                                                          'STRstim',stimdata.STRramp.stop_times,...
                                                          'STNstim',stimdata.STNstop.start_times),...
                                        'weight',GA_M1);
            elseif sum(strcmpi(fieldnames(stimdata),'STNstop')) == 1
                data{nc_ind,dir_ind} = struct(nuclei{nc_ind},struct('spktimes',spk_times,'N_ids',N_ids),...
                                        'rates',stimdata.STRramp.rates,...
                                        'timerefs',struct('STRstim',stimdata.STRramp.stop_times,...
                                                          'STNstim',stimdata.STNstop.start_times),...
                                        'weight',GA_M1);
            else
                data{nc_ind,dir_ind} = struct(nuclei{nc_ind},struct('spktimes',spk_times,'N_ids',N_ids),...
                                        'rates',stimdata.STRramp.rates,...
                                        'timerefs',struct('STRstim',stimdata.STRramp.stop_times),...
                                        'weight',GA_M1);
            end
        end
    %     save([mat_data_dir,nuclei{nc_ind},'-spikedata'],'N_ids','spk_times','res')

    end

    % mat_data_dir = [data_dir,'/','mat_data/'];
    str_tmp = strsplit(data_dir,'tr');
    str_tmp = str_tmp{end};
    % str_tmp = strsplit(str_tmp{1},'tr');


    stim_props = data{1}.rates;

    % mat_datadir = [mat_datadir,'B-'];
    tmp_str = [];
    for ind = 1:length(stim_props)
        tmp_str = [tmp_str,num2str(stim_props(ind))];
    end

%     mat_datadir = fullfile(mat_datadir,tmp_str);
    % stim_props = data{end}.rates;
    % 
    % mat_datadir = [mat_datadir,'-E-'];
    % for ind = 1:length(stim_props)
    %     mat_datadir = [mat_datadir,num2str(stim_props(ind))];
    % end

    if length(unique(Gs)) == 1
        mat_fl_name = [tmp_str,'W',num2str(unique(Gs*100),'%.0f')];
        disp(['All weights are equal. They are store in: ',mat_fl_name])
    else
        mat_fl_name = [tmp_str,'/tr',str_tmp(1:end-1)];
        disp(['All weights are NOT equal. They are store in: ',mat_fl_name])
    end
    % 
    % if exist(mat_datadir,'dir') ~= 7
    %     mkdir(mat_datadir)
    % end

    if ~isempty(getenv('WORK'))
        work_res_dir = fullfile(getenv('WORK'),submitdir);
    else
        work_res_dir = submitdir;
    end

    create_directory(work_res_dir)
    
    save(fullfile(work_res_dir,mat_fl_name),'data','nuclei')

%     if length(varargin) >= 3
%         if strcmpi(varargin{3},'tmpdir')
%             system(['cp ',mat_fl_name,' ',work_res_dir])
%         end
%     end

    exit
    % pause()
end

function [] = create_directory(directory)
    if exist(directory,'dir') ~= 7
        mkdir(directory)
    end
end