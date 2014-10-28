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
from toolbox.parallel_excecution import loop

import scripts_inhibition.Go_NoGo_compete as module
# from scripts inhibition import Go_NoGo_compete as module
import oscillation_perturbations4 as op
import pprint
pp=pprint.pprint

from copy import deepcopy

FILE_NAME=__file__.split('/')[-1][0:-3]
FROM_DISK_0=0
LOAD_MILNER_ON_SUPERMICRO=True

kwargs={
        'Builder':Builder,
        
        'cores_milner':40*4,
        'cores_superm':8,
        
        'debug':False,
        'do_runs':[0],
        'do_obj':False,
        'duration':[900.,100.0],
        
        'file_name':FILE_NAME,
        'freqs':[0.5, 1.0, 1.5],
        'freq_oscillation':20.,
        'from_disk_0':FROM_DISK_0,
        
        'i0':FROM_DISK_0,
        
        'job_name':'_'.join(FILE_NAME.split('_')[1:]),

        'l_hours':['04','01','00'],
        'l_mean_rate_slices':['mean_rate_slices'],
        'l_minutes':['00','00','05'],
        'l_seconds':['00','00','00'],             
        'laptime':1000.0,
        'local_threads_milner':20,
        'local_threads_superm':4,
                 
        'max_size':4000,
        'module':module,
        
        'nets':['Net_0', 'Net_1', 'Net_2', 'Net_3', 'Net_4'],
        
        'other_scenario':True,
        
        'path_code':default_params.HOME_CODE,
        'path_results':get_path_logs(LOAD_MILNER_ON_SUPERMICRO, 
                                     FILE_NAME),
        'perturbation_list':[op.get()[4+3]],
        
        'p_sizes':[
                   0.200,  
                   0.161,    
                   0.123, 
                   0.085
                  ],
        'p_subsamp':[
                     6.25, 
                     8.3, 
                     12.5, 
                     25
                     ],
        'res':10,
        'rep':40,
        'to_memory':False,
        'to_file':True
        }

d_process_and_thread=par_process_and_thread(**kwargs)
kwargs.update(d_process_and_thread)

p_list=pert_add_go_nogo_ss(**kwargs)

for i, p in enumerate(p_list): print i, p

a_list=get_args_list_Go_NoGo_compete(p_list, **kwargs)
k_list=get_kwargs_list_indv_nets(len(p_list), kwargs)

loop(5, a_list, k_list )
        