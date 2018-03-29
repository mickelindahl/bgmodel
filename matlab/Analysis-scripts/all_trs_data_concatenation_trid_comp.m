%% Analysis of spiking data

% The purpose of this file is to plot all average firing rates in a
% colorplot for different amplitude of stimulation for each nucleus.

% modified 04.01.18
% The modification is to reduce the size of tr_id matrix.
clear
% function [] = main_all_avgfrs_multistim_avgdiff_fast(data_dir,data_dir_compare,...
%     res_dir,trial)
numtrs = 100;
global_counter = 0;                                        
spktimes = [];
spktimes_c = [];
spkids = [];
params = [];
N_ids = [];
trial_ids = [];
spk_times = [];
N_ids_c = [];
trial_ids_c = [];
spk_times_c = [];

DATA_DIR = '/space2/mohaghegh-data/temp-storage/18-01-03-gostop-reltime--130-0-STN-2000-3000/mutlistim-STN-dur20.0-2050.0-3000.0-50.0';
DATA_DIR_C = '/space2/mohaghegh-data/temp-storage/17-12-15-gostop-reltime--50-20/STR-dur140.0-400.0-2000.0-100.0';
res_dir = '/space2/mohaghegh-data/temp-storage/18-01-03-gostop-reltime--130-0-STN-2000-3000/all-in-one/';

fig_dir = [res_dir,'/all-in-one/'];
if exist(fig_dir,'dir') ~= 7
    mkdir(fig_dir)
end

nuclei = {'FS','GA','GI','M1','M2','SN','ST'};
nc_names = {'FSI','GPe Arky','GPe Proto',...
            'MSN D1','MSN D2','SNr','STN'};
        
for tr_ind = 1:numtrs
    data_dir = [DATA_DIR,'-tr',num2str(tr_ind),'/'];
    data_dir_compare = [DATA_DIR_C,'-tr',num2str(tr_ind),'/'];

    stimvars = load([data_dir,'stimspec.mat']);
    stimvars_c = load([data_dir_compare,'stimspec.mat']);
    
    stimvars_alltrs(tr_ind).gostop = stimvars;
    stimvars_alltrs(tr_ind).go = stimvars_c;
end

save([res_dir,'all_stimspec'],'stimvars_alltrs')
    
for nc_id = 6:length(nuclei)
    % initialization of vars
    num_elements = 0;
    for tr_ind = 1:numtrs
        data_dir = [DATA_DIR,'-tr',num2str(tr_ind),'/'];
        data = matfile([data_dir,'mat_data/',nuclei{nc_id},'-spikedata']);
        num_elements = num_elements + length(data.N_ids);
        clear data
    end
    N_ids = uint16(zeros(num_elements,1));
    trial_ids = [];
    spk_times = uint32(zeros(num_elements,1));
    N_ids_c = [];
    trial_ids_c = [];
    spk_times_c = [];
    tr_start = 1;
    tr_end = 0;
    tr_start_c = 1;
    tr_end_c = 0;
    for tr_ind = 1:numtrs
        
        disp(['Nuclei: ',nuclei{nc_id},' - tr ',num2str(tr_ind)])

        data_dir = [DATA_DIR,'-tr',num2str(tr_ind),'/'];
        data_dir_compare = [DATA_DIR_C,'-tr',num2str(tr_ind),'/'];

        %% Data of ramping and stop-signal

        data = matfile([data_dir,'mat_data/',nuclei{nc_id},'-spikedata']);
        
%         temp_ids = data.N_ids;
        
        tr_end = tr_end + size(data.N_ids,1);
        trial_ids = [trial_ids;[tr_start,tr_end]];
        
        N_ids(tr_start:tr_end) = data.N_ids;
        spk_times(tr_start:tr_end) = data.spk_times;
%         temp_spktimes = data.spk_times;
%         spk_times = [spk_times;temp_spktimes];
%         temp_trid = uint8(tr_ind*ones(size(temp_spktimes)));
%         tr_end = size(N_ids,1);
%         trial_ids = [trial_ids;[tr_start,tr_end]];
        tr_start = tr_end + 1;

        %% Data of stop signal
        data_c = matfile([data_dir_compare,'mat_data/',nuclei{nc_id},'-spikedata']);
        temp_ids = data_c.N_ids;
        N_ids_c = [N_ids_c;temp_ids];
        temp_spktimes = data_c.spk_times;
        spk_times_c = [spk_times_c;temp_spktimes];
%         temp_trid = uint8(tr_ind*ones(size(temp_spktimes)));
        tr_end_c = tr_end_c + size(N_ids_c,1);
        trial_ids_c = [trial_ids_c;[tr_end_c,tr_start_c]];
        tr_start_c = tr_end_c + 1;

    end
    gostop.spk_times = spk_times;
    gostop.N_ids = N_ids;
    gostop.trial_ids = trial_ids;
    gostop.res = data.res;
    
    go.spk_times = spk_times_c;
    go.N_ids = N_ids_c;
    go.trial_ids = trial_ids_c;
    go.res = data_c.res;
    
    save([res_dir,nuclei{nc_id}],'gostop','go','stimvars_alltrs',...
                    'nuclei','nc_names','-v7.3')
    clear gostop go N_ids spk_times trial_ids N_ids_c spk_times_c trial_ids_c
end