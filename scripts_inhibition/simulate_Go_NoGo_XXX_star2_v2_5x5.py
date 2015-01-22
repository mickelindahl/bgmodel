'''
Created on Aug 12, 2013

@author: lindahlm
'''
from toolbox import monkey_patch as mp
mp.patch_for_milner()

from simulate import (pert_add_go_nogo_ss, get_path_logs, 
                      par_process_and_thread,
                      get_args_list_Go_NoGo_compete,
                      get_kwargs_list_indv_nets)
from toolbox.network import default_params
from toolbox.network.manager import Builder_Go_NoGo_with_lesion_FS as Builder
from toolbox.parallel_excecution import get_loop_index,loop

import scripts_inhibition.Go_NoGo_compete as module
# from scripts inhibition import Go_NoGo_compete as module
import oscillation_perturbations4 as op
import pprint
pp=pprint.pprint

from copy import deepcopy

FILE_NAME=__file__.split('/')[-1][0:-3]
FROM_DISK_0=0
NUM_NETS=5
NUM_RUNS=1
num_sim=NUM_NETS*NUM_RUNS
kwargs={
        'Builder':Builder,
        
        'cores_milner':40*4,
        'cores_superm':40,
        
        'debug':False,
        'do_not_record':[],
        'do_nets':[],
        'do_runs':range(NUM_RUNS),
        'do_obj':False,
        'duration':[900.,100.0],
        
        'file_name':FILE_NAME,
        'from_disk_0':FROM_DISK_0,
        
        'i0':FROM_DISK_0,
        
        'job_name':'_'.join(FILE_NAME.split('_')[1:]),

        'l_hours':['02','01','00'],
        'l_mean_rate_slices':['mean_rate_slices'],
        'l_minutes':['00','00','05'],
        'l_seconds':['00','00','00'],             
        'laptime':1000.0,
        'local_threads_milner':40,
        'local_threads_superm':4,
                 
        'max_size':4000,
        'module':module,
        
        'nets':['Net_{}'.format(i) for i in range(NUM_NETS)],
        
        'other_scenario':True,
        
        'path_code':default_params.HOME_CODE,
        'path_results':get_path_logs(False,FILE_NAME),
        'perturbation_list':[op.get()[4+3]],
        
        'p_sizes':[
                   0.200,  
#                    0.161,    
#                    0.123, 
#                    0.085
                  ],
        'p_subsamp':[
                     6.25, 
#                      8.3, 
#                      12.5, 
#                      25
                     ],
        'res':5,
        'rep':40,
        'to_memory':False,
        'to_file':True,
        'time_bin':5,
        

        
        }

d_process_and_thread=par_process_and_thread(**kwargs)
kwargs.update(d_process_and_thread)

p_list=pert_add_go_nogo_ss(**kwargs)

for i, p in enumerate(p_list): print i, p

a_list=get_args_list_Go_NoGo_compete(p_list, **kwargs)
k_list=get_kwargs_list_indv_nets(len(p_list), kwargs)

loop(5, [num_sim, num_sim, NUM_RUNS], 
     a_list, k_list )
    