%% Analysis of spiking data

% The purpose of this file is to plot all average firing rates in a
% colorplot for different amplitude of stimulation for each nucleus.

function [counts,fr_vec] = baseline_firingrate(data_dir,fig_dir,nuclei,plot_id,fr_vec)
% data_dir = [pwd,'/Ramp-dur140-amp200-1000-10/'];%mat_data/'];
fig_dir = fullfile(fig_dir,'Dist-avg-frs');
if exist(fig_dir,'dir') ~= 7
    mkdir(fig_dir)
end
% nuclei = {'FS','GA','GI','M1','M2','SN','ST'};
nc_names = {'FSI','GPe Arky','GPe Proto',...
            'MSN D1','MSN D2','SNr','STN'};
        
        
time_win_start = 5000;
time_win_stop = 10000;

% fr_vec = 0:100;

if exist(fullfile(data_dir,'modifiedweights.mat'),'file') == 2
    W = load(fullfile(data_dir,'modifiedweights.mat'));
    new_w = (W.GA_M1.max + W.GA_M1.min)/2;
else
    new_w = [];
end
% nuclei_fr_hist(nuclei)

if ~iscell(nuclei)
    nuclei = {nuclei};
end

for nc_id = 1:length(nuclei)
    
    data = load(fullfile(data_dir,'mat_data/',[nuclei{nc_id},'-spikedata']));
    IDsall = double(data.N_ids);
    IDs = IDsall - min(IDsall) + 1;
    IDs_u = unique(IDs);
    spk_times = double(data.spk_times)/10;
%     spk_times_d = double(spk_times);
    numunits = max(IDs) - min(IDs) + 1;
    
    spk_times_sel = spk_times(spk_times >= time_win_start & ...
                              spk_times <= time_win_stop);
    IDs_sel = IDs(spk_times >= time_win_start & ...
                  spk_times <= time_win_stop);
    FRs = [];          
    for id_ind = 1:length(IDs_u)
        FRs(id_ind) = sum(IDs_sel == IDs_u(id_ind))/(time_win_stop - time_win_start)*1000;
    end
    if plot_id == 1
        figure;
        histogram(FRs,50)
        GCA = gca;
        GCA.FontSize = 14;
        GCA.Box = 'off';
        GCA.TickDir = 'out';
        xlabel('Average firing rates')
        ylabel('Counts')
        title([nc_names{nc_id},'-GPA-M1 = ',num2str(new_w)])
        fig_print(gcf,[fig_dir,'baseline-firingrate-',nuclei{nc_id}])
        close(gcf)
        counts = [];
    else
        counts(nc_id,:) = histcounts(FRs,fr_vec,'Normalization','probability');
%         counts(nc_id,:) = histcounts(FRs,fr_vec);
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