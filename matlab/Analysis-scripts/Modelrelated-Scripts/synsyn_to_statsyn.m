%% Date created 25.5.18 by M. Mohagheghi

% This script analyzes the weight data obtained from the simulation
% together with the membrane potentials recorded from postsynaptic neurons.
% Ultimately, this script provide estimates of the weights required for the
% conversion of dynamic weights to static ones.

clc
clear

w_path = ['/space2/mohaghegh-data/Working-Directory/PhD/Projects/',...
          'BGmodel/bgmodel/results/example/eneuro/3000/',...
          'activation-control/GPASTR-Wmod4-0-tr40'];

v_path = fullfile(w_path,'nest');

rest_pot = struct('M1',-74,'M2',-74,'FS',-74,'SN',struct('GI',-72,'M1',-80,'ST',0));

wdata = load(fullfile(w_path,'weights'));
t_vec = wdata.time;
wdata = rmfield(wdata,'time');
wdata_mod = wdata;

dat_file = dir(fullfile(w_path,'nest','*.dat'));
vdata = load(fullfile(v_path,dat_file.name));

t_steps = 10;
t_start_volt = 9;

%     for src_ind = 
srcs = fieldnames(wdata);
% wdata_mod = []
for src_ind = 1:length(srcs)
    tmp_dat = getfield(wdata,srcs{src_ind});
    tmp_dat_mod = tmp_dat;
    trgs = fieldnames(tmp_dat);
    for trg_ind = 1:length(trgs)
        disp(['source: ',srcs{src_ind},', target:',trgs{trg_ind}])
        s_t_w = getfield(tmp_dat,trgs{trg_ind});
        s_t_w_mod = s_t_w;
        if ~isempty(s_t_w.w)
            for t_ind = 1:length(t_vec)
                src_ids = s_t_w.s(t_ind,:);
                trg_ids = s_t_w.t(t_ind,:);
                weights = s_t_w.w(t_ind,:);
                v_time = t_start_volt + (t_ind-1)*t_steps;
                v_ind = vdata(:,2)==v_time;

                if strcmpi(trgs(trg_ind),'SN')
                    v_rest = getfield(rest_pot.SN,srcs{src_ind});
                else
                    v_rest = getfield(rest_pot,trgs{trg_ind});
                end

                delta_v = voltage_finder(vdata(v_ind,:),trg_ids,v_rest);
                actual_weight = weights./delta_v;
                s_t_w_mod.w(t_ind,:) = actual_weight;
            end
            setfield(tmp_dat_mod,trgs{trg_ind},s_t_w_mod)
        end
    end
    setfield(wdata_mod,srcs{src_ind},tmp_dat_mod)
end

%% Visualizing weights

% for src_ind = 1:length(srcs)
% %     tmp_dat = getfield(wdata,srcs{src_ind});
% %     tmp_dat_mod = tmp_dat;
%     trgs = fieldnames(tmp_dat);
%     for trg_ind = 1:length(trgs)
%         disp(['source: ',srcs{src_ind},', target:',trgs{trg_ind}])
% %         s_t_w = getfield(tmp_dat,trgs{trg_ind});
% %         s_t_w_mod = s_t_w;
%         if ~isempty(s_t_w.w)
% %             for t_ind = 1:length(t_vec)
%                 src_ids = s_t_w.s(t_ind,:);
%                 trg_ids = s_t_w.t(t_ind,:);
%                 weights = s_t_w.w(t_ind,:);
%                 v_time = t_start_volt + (t_ind-1)*t_steps;
%                 v_ind = vdata(:,2)==v_time;
% 
%                 if strcmpi(trgs(trg_ind),'SN')
%                     v_rest = getfield(rest_pot.SN,srcs{src_ind});
%                 else
%                     v_rest = getfield(rest_pot,trgs{trg_ind});
%                 end
% 
%                 delta_v = voltage_finder(vdata(v_ind,:),trg_ids,v_rest);
%                 actual_weight = weights./delta_v;
%                 s_t_w_mod.w(t_ind,:) = actual_weight;
%             end
%             setfield(tmp_dat_mod,trgs{trg_ind},s_t_w_mod)
%         end
%     end
%     setfield(wdata_mod,srcs{src_ind},tmp_dat_mod)
% end