'''
Created on Aug 12, 2013

@author: lindahlm
'''
from core import monkey_patch as mp
mp.patch_for_milner()

from scripts_inhibition.base_simulate import (pert_add_go_nogo_ss, get_path_logs, 
                      par_process_and_thread,
                      get_args_list_Go_NoGo_compete,
                      get_kwargs_list_indv_nets)
from core.network import default_params
from core.network.manager import Builder_Go_NoGo_with_lesion_FS_ST_pulse as Builder
from core.parallel_excecution import loop

import scripts_inhibition.base_Go_NoGo_compete as module


# from scripts inhibition import scripts_inhibition.base_Go_NoGo_compete as module
import oscillation_perturbations4 as op
import pprint
pp=pprint.pprint

from copy import deepcopy

FILE_NAME=__file__.split('/')[-1][0:-3]
FROM_DISK_0=0
LOAD_MILNER_ON_SUPERMICRO=False
NUM_NETS=3 #
NUM_RUNS=2
num_sims=NUM_NETS*NUM_RUNS
kwargs={
        'Builder':Builder,
        
        'cores_milner':40*4,
        'cores_superm':16,
        
        'debug':False,
        'do_runs':[0, 1],
        'do_obj':False,
        'duration':[900.,100.0],
        
        'file_name':FILE_NAME,
        'from_disk_0':FROM_DISK_0,
        
        'i0':FROM_DISK_0,
        
        'job_name':'_'.join(FILE_NAME.split('_')[1:]),

        'l_hours':['10','01','00'],
        'l_mean_rate_slices':['mean_rate_slices'],
        'l_minutes':['00','00','05'],
        'l_seconds':['00','00','00'],             
        'labels':['D1,D2 puls=5',
                  'D1,D2 puls=7.5',
                  'D1,D2 puls=10.',], 
        'laptime':1000.0,
        'local_threads_milner':20,
        'local_threads_superm':4,
                 
        'max_size':5000,
        'module':module,
        
        'nets':['Net_{}'.format(i) for i in range(NUM_NETS)],
        
        'other_scenario':True,
        
        'path_code':default_params.HOME_CODE,
        'path_results':get_path_logs(LOAD_MILNER_ON_SUPERMICRO, 
                                     FILE_NAME),
        'perturbation_list':[op.get()[4+3]],
        'p_pulses':[5., 7.5, 10.], #A network for each, pulse amplitude STN
        'p_sizes':[
                    1,
#                     0.523,   
                    0.4278185787,
                  ],
        'p_subsamp':[
                     1., 
#                     2.5,
                     5.,
                     ],
        'res':10,
        'rep':40,

       'time_bin':100,
        }

d_process_and_thread=par_process_and_thread(**kwargs)
kwargs.update(d_process_and_thread)

p_list=pert_add_go_nogo_ss(**kwargs)

for i, p in enumerate(p_list): print i, p

a_list=get_args_list_Go_NoGo_compete(p_list, **kwargs)
k_list=get_kwargs_list_indv_nets(len(p_list), kwargs)

# loop(get_loop_index(5, [15,15,3]), a_list, k_list )
        
loop(6, [num_sims,num_sims,NUM_RUNS], a_list, k_list )