'''
Created on Aug 12, 2013

@author: lindahlm
'''
from toolbox import monkey_patch as mp
mp.patch_for_milner()

from scripts_inhibition import config
from simulate import (pert_add_go_nogo_ss,
                      get_path_rate_runs, 
                      get_args_list_Go_NoGo_compete,
                      get_kwargs_list_indv_nets)

from toolbox.network.manager import Builder_Go_NoGo_with_lesion_FS_ST_pulse as Builder
from toolbox.parallel_excecution import get_loop_index, loop
from toolbox import directories as dr
from toolbox import my_socket

import scripts_inhibition.Go_NoGo_compete as module
import sys
import fig_01_and_02_pert as op
import pprint
pp=pprint.pprint

from copy import deepcopy

path_rate_runs=get_path_rate_runs('fig_01_and_02_sim_inh/')
ops=[op.get()[0]]
FILE_NAME=__file__.split('/')[-1][0:-3]
FROM_DISK_0=int(sys.argv[1]) if len(sys.argv)>1 else 0
LOAD_MILNER_ON_SUPERMICRO=False
NUM_NETS=1
NUM_RUNS=1
num_sims=NUM_NETS*NUM_RUNS

dc=my_socket.determine_computer
CORES=40*4 if dc()=='milner' else 10
JOB_ADMIN=config.Ja_milner if dc()=='milner' else config.Ja_else
LOCAL_NUM_THREADS= 40 if dc()=='milner' else 10
WRAPPER_PROCESS=config.Wp_milner if dc()=='milner' else config.Wp_else

kwargs={
        'Builder':Builder,
        
        'cores':CORES,
        
        'debug':False,
        'do_not_record':['M1', 'M2', 'FS','GA','GI', 'ST'], 
        'do_runs':range(NUM_RUNS),
        'do_obj':False,
        'duration':[900.,100.0],
        
        'file_name':FILE_NAME,
        'from_disk_0':FROM_DISK_0,
        
        'i0':FROM_DISK_0,
                
        'job_admin':JOB_ADMIN, #user defined class
        'job_name':'fig6_0.2_STp',

        'l_hours':['10','02','00'],
        'l_mean_rate_slices':['mean_rate_slices'],
        'l_minutes':['00','00','05'],
        'l_seconds':['00','00','00'],   
                  
        'labels':['D1,D2 puls=5',], 
        'laptime':1000.0,
        
        'local_num_threads':LOCAL_NUM_THREADS,
                 
        'max_size':20000,
        'module':module,
        
        'nets':['Net_0'],
        'nets_to_run':['Net_0'],
                
        'other_scenario':True,
        
        'path_rate_runs':path_rate_runs,
        'path_results':dr.HOME_DATA+ '/'+ FILE_NAME + '/',
        'perturbation_list':ops,
        
        'proportion_connected':[0.2]*1, #related to toal number fo runs
        
        'p_pulses':[5],
        'p_sizes':[
                    1,
                ],
        'p_subsamp':[
                     1., 
                    ],
        'res':10,
        'rep':80,

        'time_bin':100,
       
        'wrapper_process':WRAPPER_PROCESS, #user defined wrapper of subprocesses

        }


p_list=pert_add_go_nogo_ss(**kwargs)

for i, p in enumerate(p_list): print i, p

a_list=get_args_list_Go_NoGo_compete(p_list, **kwargs)
k_list=get_kwargs_list_indv_nets(len(p_list), kwargs)
        
loop(1, [num_sims, num_sims, NUM_RUNS], a_list, k_list )