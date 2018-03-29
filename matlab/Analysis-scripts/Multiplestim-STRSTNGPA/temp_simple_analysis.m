%% This simple script finds some basic statistics of Stop simulation data
data_dir = [];
if isempty(data_dir)
    data_dir = ['/space2/mohaghegh-data/temp-storage/18-01-03-gostop-reltime--130-0-STN-2000-3000/all-in-one/'];%mat_data/'];
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
stimtimes_c = double(stimdata.stimvars_alltrs(1).go.STRramp.start_times + 150);
%     stimvars_c = load([data_dir_compare,'stimspec.mat']);
%     stimtimes_c = stimvars_c.STRramp.start_times + 150;    % The difference between
                                                % start and stop times.
stimrate_c = double(stimdata.stimvars_alltrs(1).go.STRramp.rates);% Rate in ramp sim.

stimtimes = stimdata.stimvars_alltrs(1).gostop.STRramp.stop_times;
stimrate = stimdata.stimvars_alltrs(1).gostop.STNstop.rates;

% gpa_f = unique(stimrate(1,:));
% str_f = unique(stimrate(2,:));      % Max cortical rate to STR
% stn_f = unique(stimrate(3,:));      % Max cortical rate to STN
% %    stn_f = stn_f([5,10,end]);
% str_gpa_lat = unique(stimrate(4,:));
% str_stn_lat = unique(stimrate(5,:));% Time interval between STR and STN

params_str = {'GPA','STN','ReltimeGPA','ReltimeSTN','STR','Trial'};

% Averaging window 
disinh_width = 20;
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

    diff_struc = struct([]);

    %% Data of ramping and stop-signal

    data = matfile([data_dir,nuclei{nc_id}]);

    %% Data of stop signal
    Go = data.go;
    spk_times_c = double(Go.spk_times)/10;
    trials_go = double(Go.trial_ids);

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
    for sel_time = stimtimes_c
        for tr_ind = tr_vec
    %        disp(['Stim GPA = ',num2str(gpa_f(gpaf_ind))])
           l_time = sel_time - 500;
           h_time = sel_time + 500;

           spk_times_sel = spk_times_c(spk_times_c >= l_time & ...
                                       spk_times_c <= h_time & ...
                                       trials_go == tr_ind);

           spk_times_sel_sort = sort(spk_times_sel);
           ISI = diff(spk_times_sel_sort);
           no_fr = find(ISI>=disinh_width,1);
           if isempty(no_fr)
               no_fr_time = NaN;
           else
               no_fr_time = spk_times_sel_sort(no_fr) - sel_time;
           end

           off_time_str = [off_time_str;no_fr_time];
           params = [params;[tr_ind,stimrate_c(stimtimes_c==sel_time)]];

        end
        num_non_nan = off_time_str(params(:,2)==stimrate_c(stimtimes_c==sel_time));
        disp(sum(~isnan(num_non_nan)))
        
        figure;
        histogram(num_non_nan(~isnan(num_non_nan)),[-100:10])
        pause()
    end
%         cnttimes = cnttimes(1,:) - sel_stimtimes(end);
    
%     save([fig_dir,'avg_fr_data_eachtr_ISI'],...
%          'params','off_time','off_time_str',...
%          'numunits','-v7.3')
end
