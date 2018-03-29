%% Analysis of spiking data

function [] = main_spectrogram()
data_dir = [pwd,'/Ramp-dur140-amp200-1000-10/'];%mat_data/'];
fig_dir = [data_dir,'/figs/Autocorr-FFT-Freqlimited/'];
if exist(fig_dir,'dir') ~= 7
    mkdir(fig_dir)
end
nuclei = {'FS','GA','GI','M1','M2','SN','ST'};
nc_names = {'FSI','GPe Arky','GPe Proto',...
            'MSN D1','MSN D2','SNr','STN'};

stimvars = load([data_dir,'stimspec.mat']);
stimtimes = stimvars.C1.start_times;
stimrate = stimvars.C1.rates;
M1ids = stimvars.C1.stim_subpop;
M1all = stimvars.C1.allpop;
M2ids = stimvars.C2.stim_subpop;
M2all = stimvars.C2.allpop;
FSids = stimvars.CF.stim_subpop;
FSall = stimvars.CF.allpop;

% Averaging window 

win_width = 100;
overlap = 10;
fs = 1000/overlap;
hist_width = 1; %ms
fs = 1000/hist_width;

% nuclei_fr_hist(nuclei)

for nc_id = 1:length(nuclei)
    
    data = load([data_dir,'mat_data/',nuclei{nc_id},'-spikedata']);
    IDsall = double(data.N_ids);
    IDs = IDsall - min(IDsall) + 1;
    spk_times = double(data.spk_times)/10;
%     spk_times_d = double(spk_times);
    numunits = max(IDs) - min(IDs) + 1;
    switch nuclei{nc_id}
        case 'M1'
            M1ids = M1ids - min(M1all) + 1;
            [subpop_ids,subpop_spktimes] = ...
                spk_id_time_subpop_ex(M1ids,IDs,spk_times);
        case 'M2'
            M2ids = M2ids - min(M2all) + 1;
            [subpop_ids,subpop_spktimes] = ...
                spk_id_time_subpop_ex(M2ids,IDs,spk_times);
        case 'FS'
            FSids = FSids - min(FSall) + 1;
            [subpop_ids,subpop_spktimes] = ...
                spk_id_time_subpop_ex(FSids,IDs,spk_times);
    end
    
%     if strcmpi(nuclei{nc_id},'SN')
%         silent_snr_id(IDs,spk_times,fig_dir)
%     end
    
    for st_id = 1:1:length(stimtimes)
        disp([nuclei{nc_id},'-',num2str(st_id)])
        st_time = stimtimes(st_id) - 1000;
        end_time = stimtimes(st_id) + 1000;
        [cnt,cnttimes] = PSTH_mov_win(spk_times,win_width,overlap,st_time,end_time,numunits,1);
        figure;
        
        subplot(211)
        plot(cnttimes - stimtimes(st_id),cnt,...
            'LineWidth',2)
        GCA = gca;
        GCA.FontSize = 14;
        GCA.Box = 'off';
        GCA.TickDir = 'out';        
        title(['Stimulus max frequency = ',num2str(stimrate(st_id)),' Hz'])
        
        subplot(212)
%         spktimes_raster = ...
%             spk_times(spk_times>=st_time & spk_times<=end_time) - stimtimes(st_id);
%         ids_raster = IDs(spk_times>=st_time & spk_times<=end_time);
%         
%         if strcmpi(nuclei{nc_id},'SN')
%             silent_snr_id(ids_raster,spktimes_raster,fig_dir)
%         end
%         if numunits > 10000
%             random_sel = round(numunits/200);
%         elseif numunits > 1000
%             random_sel = round(numunits/20);
%         else
%             random_sel = round(numunits/10);
%         end
%         
%         raster_time_id_nest_rand(spktimes_raster,ids_raster,random_sel,'blue')




        spk_t = spk_times(spk_times >= st_time & spk_times <= end_time);
        hist_edges = (st_time):hist_width:(end_time);
        cnt_hist = histcounts(spk_t,hist_edges);
        spectimes = (hist_edges(1:end-1) + hist_width/2) - stimtimes(st_id);
        
%         hamm_filt_win = flattopwin(win_width);
%         cnt_f = conv(cnt_hist,hamm_filt_win,'same');
        cnt_f = autocorr(cnt_hist,length(cnt_hist)-1);
                %%FFT

        L = length(cnt_f);
        f = fs*(0:(L/2))/L;
        ft = fft(cnt_f);
        ftamp = abs(ft/L);
        ftamp = ftamp(1:(L/2) + 1);
        ftamp(2:end-1) = 2*ftamp(2:end-1);
        
        f_des = f <=100;
        
        subplot(212)
        plot(f(f_des),log10(ftamp(f_des)),'LineWidth',2)
        xlabel('Frequency (Hz)')
        ylabel('Amplitude')
        title(['rect-hamming-',nuclei{nc_id}])

%         spectimes = (cnttimes - stimtimes(st_id))/1000;
%         [Ker,frs] = mor_wav_gen_intfreq(fs);
%         Spec = spectrogram_int_freqs(cnt_f,Ker);
%         
% %                 [Spec,Freq,Time,Pow] = spectrogram(cnt,spec_win,stft_win-1,...
% %                                         [],fs);
% %         figure();
% %         imagesc(log10(Pow),[-2,0])
%         fr_lim = frs >= 0 & frs <= 50;
%         tim_lim = spectimes >= -500 & spectimes <= 500;
%         
%         imagesc(spectimes(tim_lim),frs(fr_lim),log10(Spec(fr_lim,tim_lim).^2))
%         GCA = gca;
%         GCA.FontSize = 14;
%         GCA.Box = 'off';
%         GCA.TickDir = 'out';
%         title(nuclei{nc_id})
%         ax = colorbar();
        fig_print(gcf,[fig_dir,'FFT-le100Hz-limited-',num2str(win_width),'ms-',nuclei{nc_id},'-',num2str(stimrate(st_id)),...
                        '-',num2str(win_width)])
        close(gcf)
        
        % subpopulation receiving stimuli plot
        
%         if sum(strcmpi(nuclei(nc_id),{'FS','M1','M2'})) > 0
%             num_units_insubpop = length(unique(subpop_ids));
%             [cnt,cnttimes] = PSTH_mov_win(subpop_spktimes,win_width,overlap,...
%                 st_time,end_time,num_units_insubpop,1);
%             figure;
% 
%             subplot(211)
%             plot(cnttimes - stimtimes(st_id),cnt,...
%                 'LineWidth',2)
%             GCA = gca;
%             GCA.FontSize = 14;
%             GCA.Box = 'off';
%             GCA.TickDir = 'out';        
%             title(['Stimulus max frequency = ',num2str(stimrate(st_id)),' Hz'])
% 
%             subplot(212)
%             spktimes_raster = ...
%                 subpop_spktimes(subpop_spktimes>=st_time & subpop_spktimes<=end_time) - stimtimes(st_id);
%             ids_raster = ids_renum_for_raster(subpop_ids(subpop_spktimes>=st_time & subpop_spktimes<=end_time));        
%             raster_time_id_nest(spktimes_raster,ids_raster,'blue')
%             GCA = gca;
%             GCA.FontSize = 14;
%             GCA.Box = 'off';
%             GCA.TickDir = 'out';
%             title(nuclei{nc_id})
%             fig_print(gcf,[fig_dir,'subpop-',nuclei{nc_id},'-',...
%                             num2str(stimrate(st_id)),'-',num2str(win_width)])
%             close(gcf)
%         end
%         pause()
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