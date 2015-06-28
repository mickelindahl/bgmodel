'''
Created on Aug 12, 2013

@author: lindahlm
'''

from core.network.manager import Builder_inhibition_striatum as Builder
from core.parallel_excecution import loop
from core.network import default_params

from scripts_inhibition.simulate import (get_path_logs, 
                      get_args_list_inhibition,
                      get_kwargs_list_indv_nets,
                      par_process_and_thread,
                      pert_set_data_path_to_milner_on_supermicro, 
                      pert_add_inhibition) 

import scripts_inhibition.base_inhibition_striatum as module
import oscillation_perturbations41_slow_sw as op
import pprint
pp=pprint.pprint

FILE_NAME=__file__.split('/')[-1][0:-3]
FROM_DISK_0=2
LOAD_MILNER_ON_SUPERMICRO=False
NUM_NETS=6
NUM_RUNS=1 #A run for each perturbation
num_sim=NUM_NETS*NUM_RUNS

kwargs={
        'Builder':Builder,
        
        'cores_milner':40*1,
        'cores_superm':20,
        
        'file_name':FILE_NAME,
        'from_disk_0':FROM_DISK_0,
        
        'debug':False,
        'do_runs':range(NUM_NETS/2), #A run for each perturbation
        'do_obj':False,
        
        'i0':FROM_DISK_0,
        
        'job_name':'inh_YYY',
        
        'l_hours':  ['00','00','00'],
        'l_minutes':['15','10','5'],
        'l_seconds':['00','00','00'],
        
        'lower':0.8,
        'local_threads_milner':20,
        'local_threads_superm':1,

        
        'module':module,    
        
        'nets':['Net_{}'.format(i) for i in range(NUM_NETS)],
        
        'resolution':10,
        'repetitions':5,
        
        'path_code':default_params.HOME_CODE,
        'path_results':get_path_logs(LOAD_MILNER_ON_SUPERMICRO, 
                                     FILE_NAME),
        'perturbation_list':[op.get()[5]],
        
        'size':3000,
        
        'upper':1.8}

d_process_and_thread=par_process_and_thread(**kwargs)
pp(d_process_and_thread)
kwargs.update(d_process_and_thread)

p_list = pert_add_inhibition(**kwargs)
p_list = pert_set_data_path_to_milner_on_supermicro(p_list,
                                                  LOAD_MILNER_ON_SUPERMICRO)

for i, p in enumerate(p_list): print i, p

a_list=get_args_list_inhibition(p_list, **kwargs)
k_list=get_kwargs_list_indv_nets(len(p_list), kwargs)

for obj in a_list:
    print obj.kwargs['setup'].nets_to_run


loop(2,[NUM_NETS,NUM_NETS,1], a_list, k_list )

        