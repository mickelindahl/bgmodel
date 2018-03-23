%% Created by Mohagheghi 09.11.17
% This function reads gdf files
% Maximum value that uint16 can store is 65535 and the maximum value that
% uint32 can store is around 400000s with 0.1 ms resolution. Then becareful
% about using the right data format!

function [ids,spk_times] = readgdf_sep_stim(fl_name,sim_res,ref)
    
    data = importdata(fl_name);
    
    if isempty(data)
        ids = [];
        spk_times = [];
    else
        
        data = data(data(:,2) >= ref-1000 & data(:,2) <= ref+1000,:);

        if max(data(:,1)) > intmax('uint16')
            ids = uint32(data(:,1));
        else
            ids = uint16(data(:,1));
        end

        if ceil(max(data(:,2))/sim_res) > intmax('uint32')
            spk_times = uint64(data(:,2)/sim_res);
        else
            spk_times = uint32(data(:,2)/sim_res);
        end
    end
