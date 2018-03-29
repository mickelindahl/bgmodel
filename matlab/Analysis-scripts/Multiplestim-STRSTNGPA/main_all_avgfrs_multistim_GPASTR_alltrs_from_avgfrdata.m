%% Created by Mohagheghi on 20.12.17

%% Analysis of spiking data

% The purpose of this file is to plot all average firing rates in a
% colorplot for different amplitude of stimulation for each nucleus. This
% script is modified for multistim scenario which basically is used to
% investigate the interaction of Stop and Go signals in the BG. This file
% replot what "main_all_avgfrs_multistim_avgdiff_alltrs.m" has already
% ploted for specific window sizes and shift steps. This file uses "avg_fr_data"
% files created by "main_all_avgfrs_multistim_avgdiff_alltrs.m"

function [] = main_all_avgfrs_multistim_GPASTR_alltrs_from_avgfrdata(data_dir,data_dir_stn,res_dir)

    res_sep_dir = true;

    if isempty(data_dir)
        data_dir = [pwd,'/STN-dur10.0-1000.0-2000.0-50.0/'];%mat_data/']
    end
    fig_dir = [res_dir,'/all-avgfs-all-scenarios/'];
    if exist(fig_dir,'dir') ~= 7
        mkdir(fig_dir)
    end
    nuclei = {'SN','FS','GA','GI','M1','M2','ST'};
    nc_names = {'SNr','FSI','GPe Arky','GPe Proto',...
                'MSN D1','MSN D2','STN'};
            
    for nc_id = 1:length(nuclei)
        data_gpa = load([data_dir,'/',nuclei{nc_id},'/avg_fr_data_eachtr.mat']);
        cnt_times = data_gpa.t_samples;
        stn_f = unique(data_gpa.params(:,2));
        gpa_f = unique(data_gpa.params(:,1));
        rel_str_gpe = unique(data_gpa.params(:,4));
        rel_str_stn = unique(data_gpa.params(:,5));
        str_freq = unique(data_gpa.params(:,3));
%         GSRR = combvec(str_freq',rel_str_gpe',rel_str_stn');
        GSSR = combvec(gpa_f',stn_f',str_freq',rel_str_gpe');
        cnt = data_gpa.cnt;
        cnt_str = data_gpa.cnt_str;
        params = data_gpa.params;
        
        data_stn = load([data_dir_stn,'/',nuclei{nc_id},'/avg_fr_data.mat']);
        st_cnt_times = data_stn.rel_time_to_rampoffset;
        st_stn_f = unique(data_stn.stn_f);
        st_rel_str_stn = unique(data_stn.str_stn_lat);
        st_str_freq = unique(data_stn.str_freq);
        st_cnt_times = data_stn.cnt_rel_stn;
        st_cnt_str = data_stn.cnt_str_rel_stn;
        
%         cnt_times = cnt_times(cnt_times >=- 150 & ...
%                               cnt_times <= 250);
%                           
%         cnt = cnt(:,cnt_times >=- 150 & ...
%                   cnt_times <= 250,:,:);
%               
%         cnt_str = cnt_str(:,cnt_times >=- 150 & ...
%                           cnt_times <= 250,:,:);
        
        
        for s_r_ind = 1:size(GSSR,2)
            sel_ind = find(params(:,1) == GSSR(1,s_r_ind) & ...
                           params(:,2) == GSSR(2,s_r_ind) & ...
                           params(:,3) == GSSR(3,s_r_ind) & ...
                           params(:,4) == GSSR(4,s_r_ind));
        
            figure;
            subplot(311)
            imagesc(cnt_times,rel_str_stn,cnt(sel_ind,:),...
                    [min(cnt(:)),...
                     max(cnt(:))])
            xlim([-150,250])
            GCA = gca;
            GCA.FontSize = 14;
            GCA.Box = 'off';
            GCA.TickDir = 'out';
%                 xlabel('Time to ramp offset (ms)')
            ylabel([{'Max input firing'},{'rate to GPA (Hz)'}])
            xlabel('Time to ramp offset (ms)')
            title([nc_names{nc_id},...
                '-STR = ',num2str(GSSR(3,s_r_ind),'%i'),' (Hz)',...
                '-STN = ',num2str(GSSR(2,s_r_ind),'%i'),' (Hz)',...
                '-GPA = ',num2str(GSSR(1,s_r_ind),'%i'),' (Hz)',...
                '-RELSG = ',num2str(GSSR(4,s_r_ind),'%i'),' (ms)'])
            ax = colorbar();
            ax.Label.String = 'Average firing rate (Hz)';
            ax.Label.FontSize = 14;
            
            st_sel_inds(1) = find(st_str_freq == GSSR(3,s_r_ind));
%             st_sel_inds(2) = find(st_rel_str_stn == GSRR(4,s_r_ind));
            st_sel_inds(2) = find(st_stn_f == GSSR(2,s_r_ind));
            st_cnt_tmp = reshape(st_cnt_times(st_sel_inds(1),:,:,st_sel_inds(2)),[size(st_cnt_times,2),size(st_cnt_times,3)]);
            st_cnt_tmp = st_cnt_tmp';
            
            
            subplot(312)
            imagesc(cnt_times,rel_str_stn,st_cnt_tmp,...
                    [min(cnt(:)),...
                     max(cnt(:))])
            xlim([-150,250])
            GCA = gca;
            GCA.FontSize = 14;
            GCA.Box = 'off';
            GCA.TickDir = 'out';
%                 xlabel('Time to ramp offset (ms)')
            ylabel([{'Max input firing'},{'rate to GPA (Hz)'}])
            xlabel('Time to ramp offset (ms)')
%             title([nc_names{nc_id},...
%                 '-STR = ',num2str(GSSR(1,s_r_ind),'%i'),' (Hz)',...
%                 '-RELSG = ',num2str(GSSR(2,s_r_ind),'%i'),' (ms)',...
%                 '-RELSS = ',num2str(GSSR(3,s_r_ind),'%i'),' (ms)'])
            ax = colorbar();
            ax.Label.String = 'Average firing rate (Hz)';
            ax.Label.FontSize = 14;

            subplot(313)
            imagesc(cnt_times,rel_str_stn(1),cnt_str(sel_ind(1),:),...
                    [min(cnt(:)),...
                     max(cnt(:))])
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
                        num2str(GSSR(1,s_r_ind),'%i'),...
                        num2str(GSSR(2,s_r_ind),'%i'),...
                        num2str(GSSR(3,s_r_ind),'%i'),...
                        num2str(GSSR(4,s_r_ind),'%i')])
            if res_sep_dir
                fig_dir_GPA = [fig_dir,'GPA',num2str(GSSR(1,s_r_ind),'%i'),'/'];
                
                if exist(fig_dir_GPA,'dir') ~= 7 
                    mkdir(fig_dir_GPA)
                end
                
                fig_print(gcf,[fig_dir_GPA,nuclei{nc_id},...
                            num2str(GSSR(2,s_r_ind),'%i'),...
                            num2str(GSSR(3,s_r_ind),'%i'),...
                            num2str(GSSR(4,s_r_ind),'%i')])
                
                fig_dir_STN = [fig_dir,'STN',num2str(GSSR(2,s_r_ind),'%i'),'/'];
                
                if exist(fig_dir_STN,'dir') ~= 7 
                    mkdir(fig_dir_STN)
                end
                
                fig_print(gcf,[fig_dir_STN,nuclei{nc_id},...
                            num2str(GSSR(1,s_r_ind),'%i'),...
                            num2str(GSSR(3,s_r_ind),'%i'),...
                            num2str(GSSR(4,s_r_ind),'%i')])
                        
                fig_dir_STR = [fig_dir,'STR',num2str(GSSR(3,s_r_ind),'%i'),'/'];
                
                if exist(fig_dir_STR,'dir') ~= 7 
                    mkdir(fig_dir_STR)
                end
                
                fig_print(gcf,[fig_dir_STR,nuclei{nc_id},...
                            num2str(GSSR(1,s_r_ind),'%i'),...
                            num2str(GSSR(2,s_r_ind),'%i'),...
                            num2str(GSSR(4,s_r_ind),'%i')])
            end
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
