%% Apply analysis functions recursively based on a different values of a parameter

task = 'plot';

dir_name_range = 35:5:65;
num_trials = 20;
% dir_name_range = 60;
% dir_name_range(2) = [];

switch task
    
    case {'dataconcatenation','data-concatenate'}
        for dir_ind = 1:length(dir_name_range)
           data_dir =  ['/home/mohaghegh-data/temp-storage/18-02-14-gostop+GPA-longsensorystim-gGPASTR-increased/',...
                        'GPASTR-Wmod',num2str(dir_name_range(dir_ind)),'-GPA-40.0-500.0-2000.0-500.0-STR-140.0-500.0-1500.0-500.0-STN-10.0-500.0-2000.0-500.0'];
           data_dir_str_c = ['/home/mohaghegh-data/temp-storage/18-03-12-go-STR500-1500-gGPASTR-increased/',...
                        'GPASTR-Wmod',num2str(dir_name_range(dir_ind)),'-STR-140.0-500.0-1500.0-500.0'];
           res_dir = ['/home/mohaghegh-data/temp-storage/18-02-14-gostop+GPA-longsensorystim-gGPASTR-increased/',...
                      'W-',num2str(dir_name_range(dir_ind))];
            all_trs_data_concatenation_func(data_dir,data_dir_str_c,res_dir,num_trials)
        end
        
    case {'ISI','isi'}
        for dir_ind = 1:length(dir_name_range)
            disinh_w = 50;
            data_dir = ['/home/mohaghegh-data/temp-storage/18-02-14-gostop+GPA-longsensorystim-gGPASTR-increased/W-',num2str(dir_name_range(dir_ind)),'/all-in-one-numtr20/'];
            disp(['Analyzing data in ',data_dir])
            res_dir = [data_dir,'w',num2str(disinh_w),'/'];
            main_avgfrs_multistim_eachtr_ISI_parfor_GPAinc(data_dir,res_dir,disinh_w)
        end
        
    case {'plot','PLOT'}    
        for dir_ind = 1:length(dir_name_range)
%             data_dir = ['/home/mohaghegh-data/temp-storage/18-02-14-gostop+GPA-longsensorystim-gGPASTR-increased/W-',...
%                         num2str(dir_name_range(dir_ind)),'/all-in-one-numtr20/SN/'];
                    
            data_dir = ['/home/mohaghegh-data/temp-storage/18-02-14-gostop+GPA-longsensorystim-gGPASTR-increased/W-',...
                        num2str(dir_name_range(dir_ind)),'/all-in-one-numtr20/SN/'];
            
            res_dir = data_dir;
            comp_data_dir_GPA = ['/home/mohaghegh-data/temp-storage/18-02-15-gostop-onlyGPA-longsensorystim-gGPASTR-increased/W-',num2str(dir_ind),'/all-in-one-numtr20/SN/'];
%             comp_data_dir_STN = ['/home/mohaghegh-data/temp-storage/18-03-12-gostop-onlySTN-STR500-1500-STN500-2000-gGPASTR-increased/',...
%                                  'W-',num2str(dir_name_range(dir_ind)),'/all-in-one-numtr20/figs/SN/'];
                             
            comp_data_dir_STN = ['/home/mohaghegh-data/temp-storage/18-03-12-gostop-onlySTN-STR500-1500-STN500-2000-gGPASTR-increased/',...
                                 'W-',num2str(dir_name_range(dir_ind)),'/all-in-one-numtr20/figs/SN/'];
            save('data_dirs','data_dir','comp_data_dir_STN');
            disp(['GPASTN ',data_dir])
            disp(['STN ',comp_data_dir_STN])
            SNr_dec_latency_ISI_STNGPA_stop_vs_STN_RELvsRELvis_avgcomp(data_dir,comp_data_dir_STN,res_dir)
%             SNr_dec_latency_ISI_STNGPA_stop_vs_GPA_RELvsRELvis_avgcomp(data_dir,comp_data_dir_GPA,res_dir)
        end
        
     case {'plot-nofsi','plot-noFSI'}
        for dir_ind = 1:length(dir_name_range)
%             data_dir = ['/home/mohaghegh-data/temp-storage/18-02-14-gostop+GPA-longsensorystim-gGPASTR-increased/W-',...
%                         num2str(dir_name_range(dir_ind)),'/all-in-one-numtr20/SN/'];
                    
            data_dir = ['/home/mohaghegh-data/temp-storage/18-02-14-gostop+GPA-longsensorystim-gGPASTR-increased/W-',...
                        num2str(dir_name_range(dir_ind)),'/all-in-one-numtr20/SN/'];
            
            res_dir = data_dir;
%             comp_data_dir_GPA = ['/home/mohaghegh-data/temp-storage/18-02-15-gostop-onlyGPA-longsensorystim-gGPASTR-increased/W-',num2str(dir_ind),'/all-in-one-numtr20/SN/'];
%             comp_data_dir_STN = ['/home/mohaghegh-data/temp-storage/18-03-12-gostop-onlySTN-STR500-1500-STN500-2000-gGPASTR-increased/',...
%                                  'W-',num2str(dir_name_range(dir_ind)),'/all-in-one-numtr20/figs/SN/'];
                             
            comp_data_dir_noFSI = ['/home/mohaghegh-data/temp-storage/18-03-12-gostop+GPA-longsensorystim-gGPASTR-increased-noFSIstim/',...
                                 'W-',num2str(dir_name_range(dir_ind)),'/all-in-one-numtr20/SN/'];
%             save('data_dirs','data_dir','comp_data_dir_STN');
            disp(['GPASTN ',data_dir])
            disp(['GPASTN-noFSI ',comp_data_dir_noFSI])
            SNr_dec_latency_ISI_STNGPA_stop_vs_noFSIstim_RELvsRELvis(data_dir,comp_data_dir_noFSI,res_dir)
%             SNr_dec_latency_ISI_STNGPA_stop_vs_GPA_RELvsRELvis(data_dir,comp_data_dir_GPA,res_dir)
        end
        
    case {'plot-relstn','plot-RELSTN'}
        for dir_ind = 1:length(dir_name_range)
            data_dir = ['/home/mohaghegh-data/temp-storage/18-02-14-gostop+GPA-longsensorystim-gGPASTR-increased/W-',...
                        num2str(dir_name_range(dir_ind)),'/all-in-one-numtr20/SN/'];
            disp(['Plotting data in ',data_dir])
            res_dir = data_dir;
            comp_data_dir_GPA = ['/home/mohaghegh-data/temp-storage/18-02-15-gostop-onlyGPA-longsensorystim-gGPASTR-increased/W-',num2str(dir_name_range(dir_ind)),'/all-in-one-numtr20/SN/'];
            comp_data_dir_STN = ['/home/mohaghegh-data/temp-storage/18-03-12-gostop-onlySTN-STR500-1500-STN500-2000-gGPASTR-increased/W-',...
                                num2str(dir_name_range(dir_ind)),'/all-in-one-numtr20/figs/SN/'];
            SNr_dec_latency_ISI_STNGPA_stop_vs_STN_RELvsRELvis_STNREL(data_dir,comp_data_dir_STN,res_dir)
%             SNr_dec_latency_ISI_STNGPA_stop_vs_GPA_RELvsRELvis_STNREL(data_dir,comp_data_dir_GPA,res_dir)
        end
        
    case {'plot-wo-comparison'}
        for dir_ind = dir_name_range
            data_dir = ['/home/mohaghegh-data/temp-storage/18-02-14-gostop+GPA-longsensorystim-gGPASTR-increased/',...
                            num2str(dir_ind),'/all-in-one-numtr20/SN/'];
            disp(['Plotting data in ',data_dir])
            res_dir = [data_dir,'mod/'];
            main_SNr_dec_latency_ISI_STNGPA_stop_RELvsRELvis(data_dir,res_dir)
        end
        
    case {'delay-all-together'}
        res_dir = '/space2/mohaghegh-data/Working-Directory/PhD/Projects/BGmodel/MATLAB-data-analysis/Analysis/18-03-12-comparison-GPASTN-vs-STN-and-GPA/';
%         data_dir = 
        comparison_region = 'STN';
        conds = dir_name_range/100;
        delay_difference_func(res_dir,comparison_region,conds)
%         comparison_region = 'GPA';
%         delay_difference_func(res_dir,comparison_region,conds)
    
    case {'all-avgfrs'}
%         num_proc = 2;
%         if num_proc > 1
%             parpool('local',num_proc)
%         end
        disp('Computing average firing rates among all trials for each stimulation parameter')
        
        for dir_ind = 1:length(dir_name_range)
            data_dir = ['/home/mohaghegh-data/temp-storage/18-03-12-gostop+GPA-longsensorystim-gGPASTR-increased-noFSIstim/W-',...
                            num2str(dir_name_range(dir_ind)),'/all-in-one-numtr20/'];
            disp(['Plotting data in ',data_dir])
            res_dir = data_dir;
            avgfrs_multistim_alltrs(data_dir,res_dir)
%             main_sel_avgfrs_multistim_from_avgdata
        end
        
    case {'all-avgfrs-visual'}
        
        disp('Showing the results of stop signal compared without stop')
        
        for dir_ind = 1:length(dir_name_range)
            data_dir = ['/home/mohaghegh-data/temp-storage/18-03-12-gostop+GPA-longsensorystim-gGPASTR-increased-noFSIstim/W-',...
                        num2str(dir_name_range(dir_ind)),'/all-in-one-numtr20/figs/'];
                    
            data_dir_stn = ['/home/mohaghegh-data/temp-storage/18-03-12-gostop-onlySTN-STR500-1500-STN500-2000-gGPASTR-increased/W-',...
                            num2str(dir_name_range(dir_ind)),'/all-in-one-numtr20/figs/'];
                        
            res_dir = ['/home/mohaghegh-data/temp-storage/18-03-12-gostop+GPA-longsensorystim-gGPASTR-increased-noFSIstim/W-',...
                        num2str(dir_name_range(dir_ind)),'/all-in-one-numtr20/'];
            main_all_avgfrs_multistim_GPASTR_alltrs_from_avgfrdata(data_dir,data_dir_stn,res_dir)
        end
        
    case {'individual-traces'}
        stnstr_data_dir = '/home/mohaghegh-data/temp-storage/18-02-28-gostop-onlySTN-STR500-2000-STN500-2000/all-in-one-numtr20/figs/';
        for dir_ind = dir_name_range
            gpastnstr_data_dir = ['/home/mohaghegh-data/temp-storage/18-02-14-gostop+GPA-longsensorystim-gGPASTR-increased/',...
                        num2str(dir_ind),'/all-in-one-numtr20/figs/'];
            res_dir = gpastnstr_data_dir;
            main_sel_avgfrs_multistim_from_avgdata(stnstr_data_dir,gpastnstr_data_dir,res_dir)
        end
end