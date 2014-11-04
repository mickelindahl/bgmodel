'''
Created on Aug 12, 2013

@author: lindahlm
'''
from toolbox import monkey_patch as mp
mp.patch_for_milner()

from simulate import (get_path_logs, 
                      pert_add,
                      pert_add_go_nogo_ss, 
                      par_process_and_thread,
                      get_args_list_Go_NoGo_compete,
                      get_kwargs_list_indv_nets)
from toolbox.network import default_params
from toolbox.network.manager import Builder_Go_NoGo_with_nodop_FS as Builder
from toolbox.parallel_excecution import get_loop_index, loop

import scripts_inhibition.Go_NoGo_compete as module


# from scripts inhibition import Go_NoGo_compete as module
import oscillation_perturbations4 as op
import oscillation_perturbations_dop as op_dop
import pprint
pp=pprint.pprint

for p in [op_dop.get()[5]]:
    print p

FILE_NAME=__file__.split('/')[-1][0:-3]
FROM_DISK_0=0
LOAD_MILNER_ON_SUPERMICRO=False

kwargs={
        'Builder':Builder,
        
        'cores_milner':40*8,
        'cores_superm':12,
        
        'debug':False,
        'do_runs':[0],
        'do_obj':False,
        'duration':[900.,100.0],
        
        'file_name':FILE_NAME,
        'from_disk_0':FROM_DISK_0,
        
        'i0':FROM_DISK_0,
        
        'job_name':'_'.join(FILE_NAME.split('_')[1:]),

        'l_hours':['04','02','00'],
        'l_mean_rate_slices':['mean_rate_slices'],
        'l_minutes':['00','00','05'],
        'l_seconds':['00','00','00'],            
        'labels':['Only D1', 
                  'D1,D2',
                  'Only D1 no dop', 
                  'D1,D2 no dop'], 
        'laptime':1000.0,
        'local_threads_milner':40,
        'local_threads_superm':4,
                 
        'max_size':20000.,
        'module':module,
        
        'nets':['Net_0', 'Net_1', 'Net_2', 'Net_3'],
        
        'op_pert_add':[op_dop.get()[5]],    
    
        'path_code':default_params.HOME_CODE,
        'path_results':get_path_logs(LOAD_MILNER_ON_SUPERMICRO, 
                                     FILE_NAME),
        'perturbation_list':[op.get()[7]],
        'proportion_connected':[0.08]*4, #related to toal number fo runs
        'p_sizes':[
                   1.
                  ],
        'p_subsamp':[
                     1.
                     ],
        'res':10,
        'rep':40,
        
        'time_bin':100,
        }

d_process_and_thread=par_process_and_thread(**kwargs)
kwargs.update(d_process_and_thread)

p_list=pert_add_go_nogo_ss(**kwargs)
p_list=pert_add(p_list, **kwargs)

for i, p in enumerate(p_list): print i, p

a_list=get_args_list_Go_NoGo_compete(p_list, **kwargs)
k_list=get_kwargs_list_indv_nets(len(p_list), kwargs)

loop(get_loop_index(4, [4,4,1]), a_list, k_list )
        