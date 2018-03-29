%% Apply a function recursively


task = 'plot';

dir_name_range = 20:5:65;
num_trials = 20;
% dir_name_range(2) = [];

switch task
    
    
    case {'dataconcatenation','data-concatenate'}
        
        for dir_ind = 1:length(dir_name_range)
            data_dir =  ['/home/mohaghegh-data/temp-storage/18-03-12-gostop-onlySTN-STR500-1500-STN500-2000-gGPASTR-increased/',...
                         'GPASTR-Wmod',num2str(dir_name_range(dir_ind)),'-',num2str(dir_ind-1),'-STR-140.0-500.0-1500.0-500.0-STN-10.0-500.0-2000.0-500.0'];
            data_dir_str = ['/home/mohaghegh-data/temp-storage/18-03-12-go-STR500-1500-gGPASTR-increased/',...
                            'GPASTR-Wmod',num2str(dir_name_range(dir_ind)),'-STR-140.0-500.0-1500.0-500.0'];
            res_dir = ['/home/mohaghegh-data/temp-storage/18-03-12-gostop-onlySTN-STR500-1500-STN500-2000-gGPASTR-increased/',...
                       'W-',num2str(dir_name_range(dir_ind))];
            all_trs_data_concatenation_func(data_dir,data_dir_str,res_dir,num_trials)
        end
    
    case {'ISI','isi'}
        
        for dir_ind = 1:length(dir_name_range)
            disinh_w = 50;
            data_dir = ['/home/mohaghegh-data/temp-storage/18-03-12-gostop-onlySTN-STR500-1500-STN500-2000-gGPASTR-increased/W-',...
                        num2str(dir_name_range(dir_ind)),'/all-in-one-numtr20/'];
            disp(['Analyzing data in ',data_dir])
            res_dir = [data_dir,'w',num2str(disinh_w),'/'];
            main_avgfrs_multistim_eachtr_ISI_parfor(data_dir,res_dir,disinh_w)
        end
        
    case {'plot','PLOT'}
        
        for dir_id = 1:length(dir_name_range)
            data_dir = ['/home/mohaghegh-data/temp-storage/18-03-12-gostop-onlySTN-STR500-1500-STN500-2000-gGPASTR-increased/W-',...
                        num2str(dir_name_range(dir_id)),'/all-in-one-numtr20/figs/SN/'];
            disp(['Plotting data in ',data_dir])
            res_dir = data_dir;
%             comp_data_dir = ['/home/mohaghegh-data/temp-storage/18-02-15-gostop-onlyGPA-longsensorystim-gGPASTR-increased/W-',num2str(dir_id),'/all-in-one-numtr20/SN/'];
%             SNr_dec_latency_ISI_STNGPA_stop_vs_STN_RELvsRELvis(data_dir,res_dir)
            main_SNr_dec_latency_ISI_STRvsRELvis(data_dir,res_dir)
        end
        
        
    case {'plot-wo-comparison'}
        data_dir = ['/home/mohaghegh-data/temp-storage/18-02-28-gostop-onlySTN-STR500-2000-STN500-2000/',...
                        '/all-in-one-numtr20/figs/SN/'];
        disp(['Plotting data in ',data_dir])
        res_dir = [data_dir];
        main_SNr_dec_latency_ISI_STRvsRELvis(data_dir,res_dir)
        
        
    case {'delay-all-together'}
        res_dir = '/space2/mohaghegh-data/Working-Directory/PhD/Projects/BGmodel/MATLAB-data-analysis/Analysis/18-02-19-comparison-GPASTN-vs-STN-and-GPA/';
        comparison_region = 'STN';
        conds = dir_name_range/100;
        delay_difference_func(res_dir,comparison_region,conds)
        comparison_region = 'GPA';
        delay_difference_func(res_dir,comparison_region,conds)
        
    case {'all-avgfrs'}
        for dir_ind = 1:length(dir_name_range)
            data_dir = ['/home/mohaghegh-data/temp-storage/18-03-12-gostop-onlySTN-STR500-1500-STN500-2000-gGPASTR-increased/W-',...
                        num2str(dir_name_range(dir_ind)),'/all-in-one-numtr20/'];
            res_dir = data_dir;
            main_all_avgfrs_multistim_avgdiff_alltrs(data_dir,res_dir,'fig')
        end
        
    case {'all-avgfrs-visual'}
        
        for dir_ind = 1:length(dir_name_range)
            data_dir = ['/home/mohaghegh-data/temp-storage/18-03-12-gostop-onlySTN-STR500-1500-STN500-2000-gGPASTR-increased/W-',...
                        num2str(dir_name_range(dir_ind)),'/all-in-one-numtr20/figs/'];
            res_dir = data_dir;
            main_all_avgfrs_multistim_alltrs_from_avgdata(data_dir,res_dir)
        end
        
    case {'individual-traces'}
        stnstr_data_dir = '/home/mohaghegh-data/temp-storage/18-02-28-gostop-onlySTN-STR500-2000-STN500-2000/all-in-one-numtr20/figs/';
        for dir_id = dir_name_range
            gpastnstr_data_dir = ['/home/mohaghegh-data/temp-storage/18-02-14-gostop+GPA-longsensorystim-gGPASTR-increased/',...
                        num2str(dir_id),'/all-in-one-numtr20/SN/figs/'];
            res_dir = gpastnstr_data_dir;
            main_sel_avgfrs_multistim_from_avgdata(stnstr_data_dir,gpastnstr_data_dir,res_dir)
        end
end