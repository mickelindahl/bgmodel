'''
Created on Aug 12, 2013

@author: lindahlm

'''

from toolbox import monkey_patch as mp
mp.patch_for_milner()

from simulate import (get_path_rate_runs,
                      get_path_logs, get_args_list_oscillation,
                      get_kwargs_list_indv_nets,
                      par_process_and_thread,
                      pert_set_data_path_to_milner_on_supermicro, 
                      pert_add_oscillations) 

from toolbox.network import default_params
from toolbox.network.manager import Builder_beta as Builder
from toolbox.parallel_excecution import loop

import simulate_beta as module
import oscillation_perturbations4 as op
import pprint
pp=pprint.pprint

FILE_NAME=__file__.split('/')[-1][0:-3]
FROM_DISK_0=0
LOAD_MILNER_ON_SUPERMICRO=False

kwargs={
        'Builder':Builder,
        
        'cores_milner':40*1,
        'cores_superm':8,
        
        'debug':True,
        'do_runs':range(10), #A run for each perturbation
        'do_obj':True,
        
        'file_name':FILE_NAME,
        'freqs':[1., 2.0],
        'freq_oscillation':20.,
        'from_disk_0':FROM_DISK_0,
        
        'i0':FROM_DISK_0,
        
        'jobb_name':'_'.join(FILE_NAME.split('_')[1:]),
        
        'l_hours':  ['01','01','00'],
        'l_minutes':['00','00','5'],
        'l_seconds':['00','00','00'],

        'local_threads_milner':10,
        'local_threads_superm':2,
        
        'module':module,
        
        'nets':['Net_0','Net_1'], #The nets for each run
        
        'path_code':default_params.HOME_CODE,
        'path_rate_runs':get_path_rate_runs('simulate_inhibition_ZZZ4/'),
        'path_results':get_path_logs(LOAD_MILNER_ON_SUPERMICRO, 
                                     FILE_NAME),
        'perturbation_list':op.get(),
        
        'sim_time':10000.0,
        'size':20000.0 ,
        }

d_process_and_thread=par_process_and_thread(**kwargs)
pp(d_process_and_thread)
kwargs.update(d_process_and_thread)

pp(kwargs)
path_code=default_params.HOME_CODE
path_rate_runs = get_path_rate_runs('simulate_inhibition_ZZZ/')
path_result_logs = get_path_logs(LOAD_MILNER_ON_SUPERMICRO, 
                                 FILE_NAME)

p_list = pert_add_oscillations(**kwargs)
p_list = pert_set_data_path_to_milner_on_supermicro(p_list,
                                                  LOAD_MILNER_ON_SUPERMICRO)

for i, p in enumerate(p_list): print i, p

a_list=get_args_list_oscillation(p_list, **kwargs)
k_list=get_kwargs_list_indv_nets(len(p_list), kwargs)

loop(5, a_list, k_list )

        