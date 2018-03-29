%% Createrd by Mohagheghi on 20.12.17

%% Analysis of spiking data

% The purpose of this file is to plot all average firing rates in a
% colorplot for different amplitude of stimulation for each nucleus. This
% script is modified for multistim scenario which basically is used to
% investigate the interaction of Stop and Go signals in the BG. This file
% replot what "main_all_avgfrs_multistim_avgdiff_alltrs.m" has already
% ploted for specific window sizes and shift steps. This file uses "avg_fr_data"
% files created by "main_all_avgfrs_multistim_avgdiff_alltrs.m"

function [] = main_all_avgfrs_multistim_alltrs_from_avgdata(data_dir,res_dir)
    if isempty(data_dir)
        data_dir = [pwd,'/STN-dur10.0-1000.0-2000.0-50.0/'];%mat_data/']
    end
    fig_dir = [res_dir,'/figs/'];
    if exist(fig_dir,'dir') ~= 7
        mkdir(fig_dir)
    end
    nuclei = {'SN','FS','GA','GI','M1','M2','ST'};
    nc_names = {'SNr','FSI','GPe Arky','GPe Proto',...
                'MSN D1','MSN D2','STN'};
            
    for nc_id = 1:length(nuclei)
        data = load([data_dir,'/',nuclei{nc_id},'/avg_fr_data.mat']);
        cnt_times = data.rel_time_to_rampoffset;
        stn_f = data.stn_f;
        rel_time = data.str_stn_lat;
        str_freq = data.str_freq;
        stn_rel = combvec(stn_f,rel_time);
        cnt = data.cnt_rel_stn;
        cnt_str = data.cnt_str_rel_stn;
        
%         cnt_times = cnt_times(cnt_times >=- 150 & ...
%                               cnt_times <= 250);
%                           
%         cnt = cnt(:,cnt_times >=- 150 & ...
%                   cnt_times <= 250,:,:);
%               
%         cnt_str = cnt_str(:,cnt_times >=- 150 & ...
%                           cnt_times <= 250,:,:);
        
        
        for s_r_ind = 1:length(stn_rel)
            stn_f_ind = find(stn_f == stn_rel(1,s_r_ind));
            rel_t_ind = find(rel_time == stn_rel(2,s_r_ind));
        
            figure;
            subplot(211)
            imagesc(cnt_times,str_freq,cnt(:,:,rel_t_ind,stn_f_ind),...
                    [min(min(cnt(:,:,rel_t_ind,stn_f_ind))),...
                     max(max(cnt(:,:,rel_t_ind,stn_f_ind)))])
            xlim([-150,250])
            GCA = gca;
            GCA.FontSize = 14;
            GCA.Box = 'off';
            GCA.TickDir = 'out';
%                 xlabel('Time to ramp offset (ms)')
            ylabel([{'Max input firing'},{'rate to STR (Hz)'}])
            title([nc_names{nc_id},...
                '-STN = ',num2str(stn_f(stn_f_ind)),' (Hz)',...
                '-REL = ',num2str(rel_time(rel_t_ind)),' (ms)'])
            ax = colorbar();
            ax.Label.String = 'Average firing rate (Hz)';
            ax.Label.FontSize = 14;

            subplot(212)
            imagesc(cnt_times ,str_freq,...
                    cnt_str(:,:,rel_t_ind,stn_f_ind),...
                    [min(min(cnt(:,:,rel_t_ind,stn_f_ind))),...
                     max(max(cnt(:,:,rel_t_ind,stn_f_ind)))])
            xlim([-150,250])
            GCA = gca;
            GCA.FontSize = 14;
            GCA.Box = 'off';
            GCA.TickDir = 'out';
            xlabel('Time to ramp offset (ms)')
            ylabel([{'Max input firing'},{'rate to STR (Hz)'}])
%                 title([nc_names{nc_id},...
%                     '; STN = ',num2str(stn_f(stnf_ind)),' (Hz)',...
%                     '; REL = ',num2str(str_stn_lat(rel_time_ind)),' (ms)'])
            title('Go experiments without Stop signal')
            ax = colorbar();
            ax.Label.String = 'Average firing rate (Hz)';
            ax.Label.FontSize = 14;

            fig_print(gcf,[fig_dir,'colorplot-',nuclei{nc_id},...
                        '-STN',num2str(stn_f(stn_f_ind)),...
                        '-REL',num2str(mod(rel_t_ind,length(rel_time) + 1)),...
                        '-',num2str(10)])
            close(gcf)
                
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
