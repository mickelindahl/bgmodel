%% Created by Mohagheghi on 09.11.17

% This file reads nest data stored in .gdf files in order to apply furhter
% analysis in MATLAB like average firing rate and ...

function data_concat_save_as_mat(varargin)
disp(['Data path is: ',varargin])

nuclei = {'FS','GA','GF','GI','M1','M2','SN','ST'};
% data_dir = [pwd,'/'];

if length(varargin) >= 2
    data_dir = varargin{1};
    res = str2double(varargin{2});
else
    data_dir = ['/Users/Mohammad/Documents/PhD/Projects/BGmodel/bgmodel',...
        '/results/example/eneuro/10000/activation-control/',...
        'STN-dur10.0-1000.0-2000.0-500.0/nest/'];
    res = 0.1;           %ms
end


% Reading data one by one

% Initialization of empty vars
% res = 0.1;           %ms
mat_data_dir = [data_dir,'/','mat_data/'];

if exist(mat_data_dir,'dir') ~= 7
    mkdir(mat_data_dir)
end

for nc_ind = 1:length(nuclei)
    N_ids = [];
    spk_times = [];
    fls = dir([data_dir,nuclei{nc_ind},'*.gdf']);
    disp(['Reading ',nuclei(nc_ind)])
    
    for fl_ind = 1:length(fls)
        
        fl_name = fls(fl_ind).name;
        [ids,spks] = readgdf([data_dir,fl_name],res);
        
        N_ids = [N_ids;ids];
        spk_times = [spk_times;spks];
    end
    
    save([mat_data_dir,nuclei{nc_ind},'-spikedata'],'N_ids','spk_times','res')
    
end

exit
