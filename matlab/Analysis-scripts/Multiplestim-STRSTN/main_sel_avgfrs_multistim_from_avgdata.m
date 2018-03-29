%% Createrd by Mohagheghi on 20.12.17

%% Analysis of spiking data

% The purpose of this file is to plot a specific average firing rate of all
% nuclei. The purpose of this script is to visualize the sequences of
% stimulations and how they show up in the average firing rate of nuclei.

function [] = main_sel_avgfrs_multistim_from_avgdata(stn_data_dir,gpa_stn_data_dir,res_dir)

    fig_dir = [res_dir,'/figs-comparison-avgs/'];
    if exist(fig_dir,'dir') ~= 7
        mkdir(fig_dir)
    end
    nuclei = {'SN','FS','GA','GI','M1','M2','ST'};
    nc_names = {'SNr','FSI','GPe Arky','GPe Proto',...
                'MSN D1','MSN D2','STN'};
    color_vec = {'magenta','green','blue','cyan','black','black','red'};
    linestyle_vec = {'-','-','-','-','-','--','-'};
    
    str_stim = 1500;
    stn_stim = 1000;        % Hz
    gpa_stim = 1000;
    rel_time_stn = -70;     % ms
    rel_time_gpa = -60;
    
    figure;
    
    avg_fr_flag = 0;         % 0: raw average firin rate
                             % 1: z-score
                             % 2: max deviation from baseline normalized
    trg_name = 'raw';
            
    for nc_id = 1:length(nuclei)
        
        %STN+STR data load
        
        stn_data = load([stn_data_dir,'/',nuclei{nc_id},'/avg_fr_data.mat']);
        cnt_times = stn_data.rel_time_to_rampoffset;
        stn_f = stn_data.stn_f;
        rel_time = stn_data.str_stn_lat;
        str_freq = stn_data.str_freq;
%         stn_rel = combvec(stn_f,rel_time);
        cnt = stn_data.cnt_rel_stn;
        cnt_str = stn_data.cnt_str_rel_stn;
        
        stn_f_ind = stn_f == stn_stim;
        rel_t_ind = find(rel_time == rel_time_stn);
        str_f_ind = str_freq == str_stim;
        
        
        cnt_sel = cnt(str_f_ind,:,...
                      rel_t_ind,...
                      stn_f_ind);
                  
        cnt_str_sel = cnt_str(str_f_ind,:,...
                              rel_t_ind,...
                              stn_f_ind);
                          
        
        
        cnt_sel = cnt(str_f_ind,:,...
                      rel_t_ind,...
                      stn_f_ind);
                  
        cnt_str_sel = cnt_str(str_f_ind,:,...
                              rel_t_ind,...
                              stn_f_ind);
                          
        time_baseline = cnt_times <= -250;
        cnt_sel_baseline = mean(cnt_sel(time_baseline));
                                    
        cnt_str_sel_baseline = mean(cnt_str_sel(time_baseline));

                          
        % The alternative to normalize the average firing rate is to use
        % average baseline firing rate instead of total average firing rate
                          
%         cnt_sel = (cnt_sel - mean(cnt_sel)/2)/mean(cnt_sel);
%         cnt_sel = (cnt_sel - mean(cnt_sel))/std(cnt_sel);
%         cnt_str_sel = (cnt_str_sel - mean(cnt_str_sel)/2)/mean(cnt_str_sel);
%         cnt_str_sel = (cnt_str_sel - mean(cnt_str_sel))/std(cnt_str_sel);

        % Normalization with baseline
        if avg_fr_flag == 2
            cnt_sel = cnt_sel - cnt_sel_baseline;
            cnt_sel = cnt_sel/max(abs(cnt_sel));

            cnt_str_sel = cnt_str_sel - cnt_str_sel_baseline;
    %         cnt_str_sel = cnt_str_sel/max((cnt_str_sel));
    %         cnt_str_sel = cnt_str_sel/(max(cnt_str_sel)-min(cnt_str_sel));
            cnt_str_sel = cnt_str_sel/max(abs(cnt_str_sel));
            trg_name = 'normed';
        elseif avg_fr_flag == 1
            cnt_sel = cnt_sel - cnt_sel_baseline;
            cnt_sel = cnt_sel/std(cnt_sel);

            cnt_str_sel = cnt_str_sel - cnt_str_sel_baseline;
    %         cnt_str_sel = cnt_str_sel/max((cnt_str_sel));
    %         cnt_str_sel = cnt_str_sel/(max(cnt_str_sel)-min(cnt_str_sel));
            cnt_str_sel = cnt_str_sel/max(abs(cnt_str_sel));
            trg_name = 'zscore';
        end
                 
        subplot(2,1,1)
        hold on
        plot(cnt_times,cnt_sel,'LineStyle',linestyle_vec{nc_id},...
                            'Color',color_vec{nc_id},...
                            'LineWidth',2)
                        
        xlim([-250,250])
        GCA = gca;
        GCA.FontSize = 14;
        GCA.Box = 'off';
        GCA.TickDir = 'out';
        ylabel(['Average firing rate'])
        title([ 'STR = ',num2str(str_freq(str_f_ind),'%i'),' (Hz)',...
                '-STN = ',num2str(stn_f(stn_f_ind),'%i'),' (Hz)',...
                '-REL = ',num2str(rel_time(rel_t_ind),'%i'),' (ms)'])
                        
        subplot(2,1,2)
        hold on
        plot(cnt_times,cnt_str_sel,'LineStyle',linestyle_vec{nc_id},...
                                    'Color',color_vec{nc_id},...
                                    'LineWidth',2)
        
        xlim([-250,250])
        GCA = gca;
        GCA.FontSize = 14;
        GCA.Box = 'off';
        GCA.TickDir = 'out';
        xlabel('Time to ramp offset (ms)')
        ylabel(['Average firing rate'])
        title('Go experiments without Stop signal')
                               
    end
    
    fig_print(gcf,[fig_dir,'avgfr_compare-',nuclei{nc_id},...
        '-STN',num2str(stn_f(stn_f_ind),'%i'),...
        '-REL',num2str(mod(rel_t_ind,length(rel_time) + 1),'%i'),...
        '-STR',num2str(str_freq(str_f_ind),'%i'),...
        '-',trg_name,...
        '-',num2str(10)])
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
        [IDs,spk_times] = spk_id_time_ex([stn_data_dir,'mat_data/',nuclei{nc_id},'-spikedata']);
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
