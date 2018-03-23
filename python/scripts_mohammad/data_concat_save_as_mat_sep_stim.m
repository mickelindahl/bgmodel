%% Created by Mohagheghi on 09.11.17

% This file reads nest data stored in .gdf files in order to apply furhter
% analysis in MATLAB like average firing rate and ...

function data_concat_save_as_mat_sep_stim(varargin)
disp(['Data path is: ',varargin])

nuclei = {'FS','GA','GF','GI','M1','M2','SN','ST'};
% data_dir = [pwd,'/'];

if length(varargin) >= 2
    data_dir = varargin{1};
    res = str2double(varargin{2});
else
    data_dir = ['/Users/Mohammad/Documents/PhD/Projects/BGmodel/bgmodel',...
        '/results/example/eneuro/3000/activation-control/',...
        'GPA-500.0-2000.0--100.0STR-600.0-500.0STN-500.0-2000.0--100.0tr1/'];
    res = 0.1;           %ms
end


% Reading data one by one

% Initialization of empty vars
% res = 0.1;           %ms
mat_data_dir = [data_dir,'/','mat_data/'];

if exist(mat_data_dir,'dir') ~= 7
    mkdir(mat_data_dir)
end

% nestdir = [data_dir,'nest/'];
% 
% stimdata = load([data_dir,'stimspec']);
% 
% ref_time = stimdata.STRramp.stop_times;
% 
all_dirs_data = load('dirdata.mat');
all_dirs = all_dirs_data.dirs;


for nc_ind = 1:length(nuclei)
    
    for dir_ind = 2:size(all_dirs,1)
        nestdir = all_dirs(dir_ind,:);
        
        data_dir = strsplit(nestdir,'nest');
        data_dir = data_dir{1};
        stimdata = load([data_dir,'stimspec']);

        ref_time = stimdata.STRramp.stop_times;

        N_ids = [];
        spk_times = [];
        gdfdir = dir([nestdir,nuclei{nc_ind},'*.gdf']);
        disp(['Reading ',nuclei(nc_ind)])

        for fl_ind = 1:length(gdfdir)

            fl_name = gdfdir(fl_ind).name;
            [ids,spks] = readgdf_sep_stim([nestdir,fl_name],res,ref_time);

            N_ids = [N_ids;ids];
            spk_times = [spk_times;spks];
        end

        data{nc_ind} = struct(nuclei{nc_ind},struct('spktimes',spk_times,'N_ids',N_ids));
    end
%     save([mat_data_dir,nuclei{nc_ind},'-spikedata'],'N_ids','spk_times','res')
    
end

exit
