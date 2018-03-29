%% Created by Mohagheghi 20.12.17 to reduce the size of data

% Reducing the size of data is done by throwing the waittime block away.

data_dir = '/space2/mohaghegh-data/temp-storage/17-12-15/all-in-one/';
nuclei = {'FS','GA','GI','M1','M2','SN','ST'};
nc_id = 6;
data = load([data_dir,nuclei{nc_id}]);

stimdata = load([data_dir,'all_stimspec.mat']);
stimtimes = stimdata.stimvars_alltrs(1).gostop.STNstop.stop_times;
SPKS = double(data.gostop.spk_times);
clear data


for II = 1:length(stimtimes)-1
    disp(num2str(stimtimes(II)))
    SPKS((SPKS >= stimtimes(II)+500) & (SPKS <= stimtimes(II+1) - 500)) = [];
end