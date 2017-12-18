%% Analysis of spiking data

function [] = main_multistim()
data_dir = [pwd,'/STN-dur10.0-1000.0-2000.0-500.0/'];%mat_data/'];
fig_dir = [data_dir,'/figs/'];
if exist(fig_dir,'dir') ~= 7
    mkdir(fig_dir)
end
nuclei = {'FS','GA','GI','M1','M2','SN','ST'};
nc_names = {'FSI','GPe Arky','GPe Proto',...
            'MSN D1','MSN D2','SNr','STN'};

stimvars = load([data_dir,'stimspec.mat']);
stim_targets = fieldnames(stimvars);
stimvars_cell = struct2cell(stimvars);
for nc_ind = 1:length(stimvars_cell)
    if length(stim_targets{nc_ind}) <= 2
        stim_pop(nc_ind).allpop = stimvars_cell{nc_ind}.allpop;
        stim_pop(nc_ind).subpop = stimvars_cell{nc_ind}.stim_subpop;
    %     stim_vecs(nc_ind).STN   = stimvars_cell{nc_ind}.STNstop;
    %     stim_vecs(nc_ind).STR   = stimvars_cell{nc_ind}.STRramp;
        switch stim_targets{nc_ind}
            case 'CS'
                stim_targets{nc_ind} = 'ST';
            case 'C1'
                stim_targets{nc_ind} = 'M1';
            case 'C2'
                stim_targets{nc_ind} = 'M2';
            case 'CF'
                stim_targets{nc_ind} = 'FS';
        end
    else
        stimtimes = stimvars_cell{nc_ind}.stop_times;
        stimrate = stimvars_cell{nc_ind}.rates;
    end
end
% stimtimes = stim_vecs(1).STN.stop_times;
% stimrate = stim_vecs(1).STN.rates;
str_f = unique(stimrate(1,:));      % Max cortical rate to STR
stn_f = unique(stimrate(2,:));      % Max cortical rate to STN
str_stn_lat = unique(stimrate(3,:));% Time interval between STR and STN

% Averaging window 

win_width = 50;
overlap = 10;

% nuclei_fr_hist(nuclei)

for nc_id = 1:length(nuclei)
    
    data = load([data_dir,'mat_data/',nuclei{nc_id},'-spikedata']);
    IDsall = double(data.N_ids);
    IDs = IDsall - min(IDsall) + 1;
    spk_times = double(data.spk_times)/10;
%     spk_times_d = double(spk_times);
    numunits = max(IDs) - min(IDs) + 1;
    
    if sum(strcmpi(nuclei{nc_id},stim_targets)) > 0
        sel_tar = strcmpi(nuclei{nc_id},stim_targets);
        tar_ids = stim_pop(sel_tar).allpop;
        tar_stim_ids = stim_pop(sel_tar).subpop;
        tar_stim_ids = tar_stim_ids - min(tar_ids) + 1;
        [subpop_ids,subpop_spktimes] = ...
        spk_id_time_subpop_ex(tar_stim_ids,IDs,spk_times);
    end
%     switch nuclei{nc_id}
%         case 'M1'
%             M1ids = M1ids - min(M1all) + 1;
%             [subpop_ids,subpop_spktimes] = ...
%                 spk_id_time_subpop_ex(M1ids,IDs,spk_times);
%         case 'M2'
%             M2ids = M2ids - min(M2all) + 1;
%             [subpop_ids,subpop_spktimes] = ...
%                 spk_id_time_subpop_ex(M2ids,IDs,spk_times);
%         case 'FS'
%             FSids = FSids - min(FSall) + 1;
%             [subpop_ids,subpop_spktimes] = ...
%                 spk_id_time_subpop_ex(FSids,IDs,spk_times);
%     end
    
%     if strcmpi(nuclei{nc_id},'SN')
%         silent_snr_id(IDs,spk_times,fig_dir)
%     end
    for rel_time_ind = 1:length(str_stn_lat)
        for stnf_ind = 1:length(stn_f)
            sel_stimtimes = stimtimes(...
                            stimrate(3,:) == str_stn_lat(rel_time_ind) & ...
                            stimrate(2,:) == stn_f(stnf_ind));
    
            for st_id = 1:1:length(sel_stimtimes)
                
                str_freq = stimrate(1,stimtimes == sel_stimtimes(st_id));
                disp([nuclei{nc_id},'-',num2str(st_id)])
                st_time = sel_stimtimes(st_id) - 1000;
                end_time = sel_stimtimes(st_id) + 1000;
                [cnt,cnttimes] = PSTH_mov_win(spk_times,win_width,overlap,st_time,end_time,numunits,1);
                figure;

                subplot(211)
                plot(cnttimes - sel_stimtimes(st_id),cnt,...
                    'LineWidth',2)
                GCA = gca;
                GCA.FontSize = 14;
                GCA.Box = 'off';
                GCA.TickDir = 'out';        
                title(['STR = ',num2str(str_freq),...
                       'STN = ',num2str(stn_f(stnf_ind)),...
                       'Rel time = ',num2str(str_stn_lat(rel_time_ind)),' Hz'])

                subplot(212)
                spktimes_raster = ...
                    spk_times(spk_times>=st_time & spk_times<=end_time) - sel_stimtimes(st_id);
                ids_raster = IDs(spk_times>=st_time & spk_times<=end_time);

        %         if strcmpi(nuclei{nc_id},'SN')
        %             silent_snr_id(ids_raster,spktimes_raster,fig_dir)
        %         end
                if numunits > 10000
                    random_sel = round(numunits/100);
                elseif numunits > 1000
                    random_sel = round(numunits/10);
                elseif numunits > 100
                    random_sel = round(numunits/5);
                else
                    random_sel = numunits;
                end
                raster_time_id_nest_rand(spktimes_raster,ids_raster,random_sel,'blue')
                GCA = gca;
                GCA.FontSize = 14;
                GCA.Box = 'off';
                GCA.TickDir = 'out';
                title(nuclei{nc_id})
                fig_print(gcf,[fig_dir,nuclei{nc_id},...
                                '-STR',num2str(str_freq),...
                                '-STN',num2str(stn_f(stnf_ind)),...
                                '-REL',num2str(str_stn_lat(rel_time_ind)),...
                                '-',num2str(win_width)])
                close(gcf)

                % subpopulation receiving stimuli plot

                if sum(strcmpi(nuclei{nc_id},stim_targets)) > 0
                    num_units_insubpop = length(unique(subpop_ids));
                    [cnt,cnttimes] = PSTH_mov_win(subpop_spktimes,win_width,overlap,...
                        st_time,end_time,num_units_insubpop,1);
                    figure;

                    subplot(211)
                    plot(cnttimes - sel_stimtimes(st_id),cnt,...
                        'LineWidth',2)
                    GCA = gca;
                    GCA.FontSize = 14;
                    GCA.Box = 'off';
                    GCA.TickDir = 'out';        
                    title(['STR = ',num2str(str_freq),...
                           'STN = ',num2str(stn_f(stnf_ind)),...
                           'Rel time = ',num2str(str_stn_lat(rel_time_ind)),' Hz'])

                    subplot(212)
                    spktimes_raster = ...
                        subpop_spktimes(subpop_spktimes>=st_time & subpop_spktimes<=end_time) - sel_stimtimes(st_id);
                    ids_raster = ids_renum_for_raster(subpop_ids(subpop_spktimes>=st_time & subpop_spktimes<=end_time));        
                    raster_time_id_nest(spktimes_raster,ids_raster,'blue')
                    GCA = gca;
                    GCA.FontSize = 14;
                    GCA.Box = 'off';
                    GCA.TickDir = 'out';
                    title(nuclei{nc_id})
                    fig_print(gcf,[fig_dir,'subpop-',nuclei{nc_id},...
                                    '-STR',num2str(str_freq),...
                                    '-STN',num2str(stn_f(stnf_ind)),...
                                    '-REL',num2str(str_stn_lat(rel_time_ind)),...
                                    '-',num2str(win_width)])
                    close(gcf)
                end
        %         pause()
            end
        end
    end
end
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