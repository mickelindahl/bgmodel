%% Analysis of spiking data

% The purpose of this file is to plot all average firing rates in a
% colorplot for different amplitude of stimulation for each nucleus.

function [] = main_all_avgfrs_multistim_avgdiff_alltrs(data_dir,res_dir)
    if isempty(data_dir)
        data_dir = [pwd,'/STN-dur10.0-1000.0-2000.0-50.0/'];%mat_data/']
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
    stimtimes_c = stimdata.stimvars_alltrs(1).go.STRramp.start_times + 150;
%     stimvars_c = load([data_dir_compare,'stimspec.mat']);
%     stimtimes_c = stimvars_c.STRramp.start_times + 150;    % The difference between
                                                    % start and stop times.
    stimrate_c = stimdata.stimvars_alltrs(1).go.STRramp.rates;% Rate in ramp sim.
    
    stimtimes = stimdata.stimvars_alltrs(1).gostop.STRramp.stop_times;
    stimrate = stimdata.stimvars_alltrs(1).gostop.STNstop.rates;
    str_f = unique(stimrate(1,:));      % Max cortical rate to STR
    stn_f = unique(stimrate(2,:));      % Max cortical rate to STN
%    stn_f = stn_f([5,10,end]);
    str_stn_lat = unique(stimrate(3,:));% Time interval between STR and STN

    % Averaging window 

    win_width = 10;
    overlap = 1;

    % nuclei_fr_hist(nuclei)
    
    all_diff_struc = struct([]);

    for nc_id = 1:length(nuclei)
        
        fig_dir = [res_dir,'/figs/',nuclei{nc_id},'/'];
        if exist(fig_dir,'dir') ~= 7
            mkdir(fig_dir)
        end
        
        diff_struc = struct([]);
        
        %% Data of ramping and stop-signal
        
        data = load([data_dir,nuclei{nc_id}]);
        data.gostop = rmfield(data.gostop,'trial_ids');
%         IDsall = data.gostop.N_ids;
%         numunits = length(unique(data.gostop.N_ids)); % Too slow with high
%                                                       % memory demand.
        numunits = double(max(data.gostop.N_ids)) - double(min(data.gostop.N_ids)) + 1;
        
        data.gostop = rmfield(data.gostop,'N_ids');
        
%         IDs = IDsall - min(IDsall) + 1;
        spk_times = double(data.gostop.spk_times)/10;
    %     spk_times_d = double(spk_times);
%         numunits = max(IDs) - min(IDs) + 1;
        
        %% Data of stop signal
        
        spk_times_c = double(data.go.spk_times)/10;
        
        clear data

        for stnf_ind = 1:length(stn_f)
            disp(['Stim STN = ',num2str(stn_f(stnf_ind))])
            h_sim_time = max(stimtimes(stimrate(2,:) == stn_f(stnf_ind)));
            l_sim_time = min(stimtimes(stimrate(2,:) == stn_f(stnf_ind)));
            spk_times_sel_init = spk_times(spk_times >= l_sim_time - 1000 &...
                                            spk_times <= h_sim_time + 1000);
            for rel_time_ind = 1:length(str_stn_lat)
                disp(['Rel time = ',num2str(str_stn_lat(rel_time_ind))])
                sel_stimtimes = stimtimes(...
                                stimrate(3,:) == str_stn_lat(rel_time_ind) & ...
                                stimrate(2,:) == stn_f(stnf_ind));

                for st_id = 1:length(sel_stimtimes)
                    str_freq(st_id) = stimrate(1,stimtimes == sel_stimtimes(st_id));
                    disp([nuclei{nc_id},'-',num2str(st_id)])
                    st_time = sel_stimtimes(st_id) - 500;
                    end_time = sel_stimtimes(st_id) + 500;
                    spk_times_sel = spk_times_sel_init(spk_times_sel_init >= st_time & ...
                                                spk_times_sel_init <= end_time );
                    [cnt(st_id,:),cnttimes] = PSTH_mov_win(spk_times_sel,...
                        win_width,overlap,st_time,end_time,numunits*numtrs,1);
                    
                    sel_stimtimes_c = stimtimes_c(stimrate_c == str_freq(st_id));
                    st_time = sel_stimtimes_c - 500;
                    end_time = sel_stimtimes_c + 500;
                    spk_times_sel = spk_times_c(spk_times_c >= st_time & ...
                                                spk_times_c <= end_time );
                    [cnt_str(st_id,:),cnttimes_str] = PSTH_mov_win(spk_times_sel,...
                        win_width,overlap,st_time,end_time,numunits*numtrs,1);
                end
                
                figure;
                subplot(211)
                imagesc(cnttimes - sel_stimtimes(st_id),str_freq,cnt,[min(cnt(:)),max(cnt(:))])
                GCA = gca;
                GCA.FontSize = 14;
                GCA.Box = 'off';
                GCA.TickDir = 'out';
%                 xlabel('Time to ramp offset (ms)')
                ylabel([{'Max input firing'},{'rate to STR (Hz)'}])
                title([nc_names{nc_id},...
                    '-STN = ',num2str(stn_f(stnf_ind)),' (Hz)',...
                    '-REL = ',num2str(str_stn_lat(rel_time_ind)),' (ms)'])
                ax = colorbar();
                ax.Label.String = 'Average firing rate (Hz)';
                ax.Label.FontSize = 14;
                        
                subplot(212)
                imagesc(cnttimes_str - sel_stimtimes_c,str_freq,cnt_str,[min(cnt(:)),max(cnt(:))])
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
                            '-STN',num2str(stn_f(stnf_ind)),...
                            '-REL',num2str(str_stn_lat(rel_time_ind)),...
                            '-',num2str(win_width)])
                close(gcf)
                cnt_rel(:,:,rel_time_ind) = cnt;
                cnt_str_rel(:,:,rel_time_ind) = cnt_str;
                    

            %         pause()
%                     diff_frs(:,rel_time_ind) = mean(cnt - cnt_str,2); % The differece is calculated to
                                                                  % to visualize the effect of STN
                                                                  % stimulation
            end
            
            cnt_rel_stn(:,:,:,stnf_ind) = cnt_rel;
            cnt_str_rel_stn(:,:,:,stnf_ind) = cnt_str_rel;

        end
        
        rel_time_to_rampoffset = cnttimes_str - sel_stimtimes_c;

%         all_diff_struc = [all_diff_struc,struct('Nucleus',nuclei(nc_id),'all_diff',diff_struc)];
        save([fig_dir,'avg_fr_data'],...
             'cnt_rel_stn','stn_f','str_freq','cnt_str_rel_stn',...
             'rel_time_to_rampoffset','str_stn_lat','numunits')
    end
%     save([fig_dir,'diffmat-tr',num2str(trial)],'all_diff_struc')
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
