%% Analysis of spiking data

% The purpose of this file is to plot all average firing rates in a
% colorplot for different amplitude of stimulation for each nucleus.

function [] = membrane_pot_plot()
data_dir = ['/home/mohaghegh/PhD/Projects/BGmodel/MATLAB-data-analysis/Analysis/Ramp-dur140ms-STR-500-600-100/'];%mat_data/'];
fig_dir = [data_dir,'/figs-mempot/'];
if exist(fig_dir,'dir') ~= 7
    mkdir(fig_dir)
end
nuclei = {'FS','GA','GI','M1','M2','SN','ST'};
nc_names = {'FSI','GPe Arky','GPe Proto',...
            'MSN D1','MSN D2','SNr','STN'};

stimvars = load([data_dir,'stimspec.mat']);
stimtimes = stimvars.STRramp.start_times;


time_uplim = 500;
time_downlim = -350;


for nc_id = 1:1%length(nuclei)
    
    data = load([data_dir,'mat_data/',nuclei{nc_id},'-mempot']);
    time_vec = double(data.time_vec);
    IDsall = double(data.N_ids);
    IDs = IDsall - min(IDsall) + 1;
    mem_pot = double(data.mem_potential)/100;
%     spk_times_d = double(spk_times);
    numunits = max(IDs) - min(IDs) + 1;
    
    for st_id = 1:1:length(stimtimes)
        disp([nuclei{nc_id},'-',num2str(st_id)])
        ref_time = stimtimes(st_id);
        for id_ind = 1:numunits
            sel_ids = IDs == id_ind;
            sel_time = time_vec >= time_downlim + ref_time & ...
                       time_vec <= time_uplim+ ref_time;
            mempot_tmp = mem_pot(sel_ids);
            sel_mempot(id_ind,:) = mempot_tmp(sel_time);
            
        end
    figure;
    imagesc(time_vec(sel_time) - stimtimes(st_id)-140,IDs,sel_mempot)
    GCA = gca;
    GCA.FontSize = 14;
    GCA.Box = 'off';
    GCA.TickDir = 'out';
    xlabel('Time to ramp offset (ms)')
    ylabel('Neuron ID')
    title(nc_names{nc_id})
    ax = colorbar();
    ax.Label.String = 'Membrane potential (mV)';
    ax.Label.FontSize = 14;
    fig_print(gcf,[fig_dir,'colorplot-',nuclei{nc_id}])
    close(gcf)
    
    figure;
    plot(time_vec(sel_time) - stimtimes(st_id)-140,sel_mempot(10,:))
    GCA = gca;
    GCA.FontSize = 14;
    GCA.Box = 'off';
    GCA.TickDir = 'out';
    xlabel('Time to ramp offset (ms)')
    ylabel('Membrane potential (mV)')
    title(nc_names{nc_id})
%     ax = colorbar();
%     ax.Label.String = 'Average firing rate (Hz)';
%     ax.Label.FontSize = 14;
    fig_print(gcf,[fig_dir,'trace10-',nuclei{nc_id}])
    close(gcf)
    end


end
end

% function [ids,spktimes] = spk_id_time_ex(dir)
%     data = load(dir);
%     IDs = double(data.N_ids);
%     ids = IDs - min(IDs);
%     spktimes = double(data.spk_times)/10;
% %     spk_times_d = double(spk_times);
% %    numunits = max(ids) - min(ids) + 1;
% end
% function [stim_ids,stim_spktimes] = spk_id_time_subpop_ex(subpop_ids,ids,spk_times)
%     stim_ids = [];
%     stim_spktimes = [];
%     for sp_ind = 1:length(subpop_ids)
%         stim_ids = [stim_ids;ids(ids == subpop_ids(sp_ind))];
%         stim_spktimes = [stim_spktimes;spk_times(ids == subpop_ids(sp_ind))];
%     end
% end
% function renumbered = ids_renum_for_raster(IDs)
%     IDs_u = unique(IDs);
%     renumbered = IDs;
%     for ind = 1:length(IDs_u)
%         renumbered(renumbered == IDs_u(ind)) = ind;
%     end
% end
% function [] = silent_snr_id(ids,spk_times,fig_dir)
%     ids_u = unique(ids);
%     for id_ind = 1:length(ids_u)
%         spks_in_id = sort(spk_times(ids==ids_u(id_ind)));
%         figure;
%         histogram(diff(spks_in_id),[0:10:100])
%         GCA = gca;
%         GCA.FontSize = 14;
%         xlabel('ISI (ms)')
%         ylabel('Counts')
%         title(['SNr unit # ',num2str(ids_u(id_ind))])
%         histdir = [fig_dir,'ISIhist/'];
%         if exist(histdir,'dir') ~= 7
%             mkdir(histdir)
%         end
%         fig_print(gcf,[histdir,'ISIhist-SNr-',num2str(ids_u(id_ind))])
%         close(gcf)
%     end
% end
% function [] = nuclei_fr_hist(nuclei,fig_dir)
%     for nc_id = 1:length(nuclei)
%         [IDs,spk_times] = spk_id_time_ex([data_dir,'mat_data/',nuclei{nc_id},'-spikedata']);
%         IDs_u = unique(IDs);
%         firingrates = zeros(2,length(IDs_u));
% 
%         for id_ind = 1:length(IDs_u)
%             firingrates(1,id_ind) = IDs(id_ind);
%             firingrates(2,id_ind) = sum(IDs == IDs_u(id_ind))/max(spk_times)*1000;
%         end
% 
%         SUFR(nc_id).nc_name = nuclei(nc_id);
%         SUFR(nc_id).fr_ids = firingrates;
%         figure;
%         histogram(firingrates(2,:),20)
%         GCA = gca;
%         GCA.FontSize = 14;
%         GCA.Box = 'off';
%         GCA.TickDir = 'out';
%         histdir = [fig_dir,'ISIhist/'];
%         fig_print(gcf,[histdir,'hist-',nuclei{nc_id}])
%         close(gcf)  
%     end
% end