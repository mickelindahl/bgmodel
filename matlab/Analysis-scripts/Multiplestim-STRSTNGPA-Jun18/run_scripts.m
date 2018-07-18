%% This simple script runs main_reset.m and main_vis.m after another.

function [] = run_scripts()

%     data_path = getenv('procdatapath');
    data_path = '/home/mohaghegh-data/temp-storage/18-06-14-randominput-fixedconn-reseting/W40';

    disp(['data is located in ',data_path])
    disp('Running main_reset.m script to preprocess the data ...')
    main_reset(data_path)

    disp('Running main_vis.m script to visualize the data ...')
    main_vis(data_path)
end