%% Created by Mohagheghi on 17.11.17
% This script sums up the two population of MSN D1 and MSN D2 to computed
% the average firing rate and compare it to the data. In the data there is
% no extinction between MSN D1 and MSN D2

function M1M2_avg_fr()

data_dir = [pwd,'/Ramp-dur140-amp200-1000-10-80000/'];%mat_data/'];
fig_dir = [data_dir,'/figs/'];
if exist(fig_dir,'dir') ~= 7
    mkdir(fig_dir)
end

nuclei = {'M1','M2'};

stimvars = load([data_dir,'stimspec.mat']);
stimtimes = stimvars.C1.start_times;
stimrate = stimvars.C1.rates;
M1ids = stimvars.C1.stim_subpop;
M1all = stimvars.C1.allpop;
M2ids = stimvars.C2.stim_subpop;
M2all = stimvars.C2.allpop;

% Averaging window 

win_width = 50;
overlap = 10;

% nuclei_fr_hist(nuclei)

    
    M1data = load([data_dir,'mat_data/M1-spikedata']);
    M1IDsall = double(M1data.N_ids);
    M1IDs = M1IDsall - min(M1IDsall) + 1;
    M1spk_times = double(M1data.spk_times)/10;
    
    M2data = load([data_dir,'mat_data/M2-spikedata']);
    M2IDsall = double(M2data.N_ids);
    M2IDs = M2IDsall - min(M2IDsall) + 1;
    M2spk_times = double(M2data.spk_times)/10;
%     spk_times_d = double(spk_times);
    numunits = max(M1IDs) - min(M1IDs) + 1;
    M1ids = M1ids - min(M1all) + 1;
    [M1subpop_ids,M1subpop_spktimes] = ...
        spk_id_time_subpop_ex(M1ids,M1IDs,M1spk_times);
    
    M2ids = M2ids - min(M2all) + 1;
    [M2subpop_ids,M2subpop_spktimes] = ...
        spk_id_time_subpop_ex(M2ids,M2IDs,M2spk_times);

    
%     if strcmpi(nuclei{nc_id},'SN')
%         silent_snr_id(IDs,spk_times,fig_dir)
%     end

    
    
    for st_id = 1:1:length(stimtimes)
        st_time = stimtimes(st_id) - 1000;
        end_time = stimtimes(st_id) + 1000;
        [M1cnt,M1cnttimes] = PSTH_mov_win(M1spk_times,win_width,overlap,st_time,end_time,numunits,1);
        [M2cnt,~] = PSTH_mov_win(M2spk_times,win_width,overlap,st_time,end_time,numunits,1);
        figure;
        
        subplot(211)
        cnt = (M1cnt + M2cnt)/2;
        plot(M1cnttimes - stimtimes(st_id),cnt,...
            'LineWidth',2)
        GCA = gca;
        GCA.FontSize = 14;
        GCA.Box = 'off';
        GCA.TickDir = 'out';        
        title(['all-Stimulus max frequency = ',num2str(stimrate(st_id)),' Hz'])        
        
        % subpopulation receiving stimuli plot
        
        M1num_units_insubpop = length(unique(M1subpop_ids));
        [M1cnt,M1cnttimes] = PSTH_mov_win(M1subpop_spktimes,win_width,overlap,...
            st_time,end_time,M1num_units_insubpop,1);
        
        M2num_units_insubpop = length(unique(M2subpop_ids));
        [M2cnt,~] = PSTH_mov_win(M2subpop_spktimes,win_width,overlap,...
            st_time,end_time,M2num_units_insubpop,1);
        
        cnt = (M1cnt + M2cnt)/2;
        
        max_fr(st_id,1) = max(cnt);
        max_fr(st_id,2) = stimrate(st_id);
        
        subplot(212)
        plot(M1cnttimes - stimtimes(st_id),cnt,...
            'LineWidth',2)
        GCA = gca;
        GCA.FontSize = 14;
        GCA.Box = 'off';
        GCA.TickDir = 'out';        
        title(['subpop, Stimulus max frequency = ',num2str(stimrate(st_id)),' Hz'])
%         fig_print(gcf,[fig_dir,'subpop-M1M2-',...
%                         num2str(stimrate(st_id)),'-',num2str(win_width)])
        close(gcf)
%         pause()
    end
    figure;
    plot(max_fr(:,2),max_fr(:,1),'LineWidth',2)
    GCA = gca;
    GCA.FontSize = 14;
    GCA.Box = 'off';
    GCA.TickDir = 'out'; 
    xlabel('Max ramping input rate')
    ylabel('Max population rate')
    fig_print(gcf,'subpop output rate vs. input rate ')
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