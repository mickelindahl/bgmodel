'''
Created on Aug 12, 2013

@author: lindahlm
'''
from core import monkey_patch as mp
mp.patch_for_milner()

from scripts_inhibition import config
from scripts_inhibition.base_simulate import (pert_add_go_nogo_ss,
                      get_path_rate_runs,
                      get_args_list_Go_NoGo_compete,
                      get_kwargs_list_indv_nets)

from core.network.manager import Builder_Go_NoGo_with_lesion_FS as Builder
from core.parallel_excecution import loop
from core import directories as dr
from core import my_socket

import scripts_inhibition.base_Go_NoGo_compete as module
import sys
import fig_01_and_02_pert as op
import pprint
pp=pprint.pprint

path_rate_runs=get_path_rate_runs('fig_01_and_02_sim_inh/')
ops=[op.get()[0]]
FILE_NAME=__file__.split('/')[-1][0:-3]
FROM_DISK_0=int(sys.argv[1]) if len(sys.argv)>1 else 0
proportion_connected=[0.1, 0.5, 1.]
NUM_RUNS=len(proportion_connected)
NUM_NETS=5
num_sims=NUM_NETS*NUM_RUNS

dc=my_socket.determine_computer
CORES=40*4 if dc()=='milner' else 10
JOB_ADMIN=config.Ja_milner if dc()=='milner' else config.Ja_else
LOCAL_NUM_THREADS= 20 if dc()=='milner' else 10
WRAPPER_PROCESS=config.Wp_milner if dc()=='milner' else config.Wp_else

kwargs={
        'Builder':Builder,
        
        'cores':CORES,
        
        'debug':False,
        'do_not_record':['M1', 'M2', 'FS','GA','GI', 'ST'], 
        'do_nets':['Net_'+str(i) for i in range(NUM_NETS)], #none means all
        'do_runs':[2], #range(NUM_RUNS),#range(NUM_RUNS), #none means all
        'do_obj':False,
        'duration':[900.,100.0],
        
        'file_name':FILE_NAME,
        'from_disk_0':FROM_DISK_0,
        
        'i0':FROM_DISK_0,
        
        'job_admin':JOB_ADMIN, #user defined clas
        'job_name':'fig6_sr',

        'l_hours':['12','01','00'],
        'l_mean_rate_slices':['mean_rate_slices'],
        'l_minutes':['00','00','05'],
        'l_seconds':['00','00','00'],            
        'labels':['Only D1', 
                   'D1,D2',
                   'MSN lesioned (D1, D2)',
                   'FSN lesioned (D1, D2)',
                   'GPe TA lesioned (D1,D2)'],
         
        'laptime':1000.0,
        'local_num_threads':LOCAL_NUM_THREADS,
        
        'max_size':20000.,
        'module':module,
        
        'nets':['Net_{}'.format(i) for i in range(NUM_NETS)],
        'nets_to_run':['Net_{}'.format(i) for i in range(NUM_NETS)],
        
        'other_scenario':True, #channels (set-set) also in GPE-SNR
        
        'path_rate_runs':path_rate_runs,
        'path_results':dr.HOME_DATA+ '/'+ FILE_NAME + '/',
        'perturbation_list':ops,
        
        'proportion_connected':proportion_connected, #related to toal number fo runs
        
        'p_sizes':[1.]*NUM_RUNS,
        'p_subsamp':[1.]*NUM_RUNS,
        
        'res':10,
        'rep':40,
                
        'time_bin':100,

        'wrapper_process':WRAPPER_PROCESS, #user defined wrapper of subprocesses
        }


p_list=pert_add_go_nogo_ss(**kwargs)

for i, p in enumerate(p_list): print i, p

a_list=get_args_list_Go_NoGo_compete(p_list, **kwargs)
k_list=get_kwargs_list_indv_nets(len(p_list), kwargs)

loop(5, [num_sims,num_sims, NUM_RUNS], a_list, k_list )
# loop(5, [5,5, NUM_RUNS], a_list, k_list )
        