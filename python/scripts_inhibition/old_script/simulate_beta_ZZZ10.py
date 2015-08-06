'''
Created on Aug 12, 2013

@author: lindahlm

'''

from core import monkey_patch as mp
mp.patch_for_milner()

from scripts_inhibition.base_simulate import (get_path_rate_runs,
                      get_path_logs, get_args_list_oscillation,
                      get_kwargs_list_indv_nets,
                      par_process_and_thread,
                      pert_set_data_path_to_milner_on_supermicro, 
                      pert_add_oscillations) 

from core.network import default_params
from core.network.manager import Builder_beta as Builder
from core.parallel_excecution import loop

import sys
import scripts_inhibition.base_oscillation_beta as module
import oscillation_perturbations10 as op
import pprint
pp=pprint.pprint

path_rate_runs=get_path_rate_runs('simulate_inhibition_ZZZ10/')
                                  
FILE_NAME=__file__.split('/')[-1][0:-3]
FROM_DISK_0=int(sys.argv[1]) if len(sys.argv)>1 else 0
LOAD_MILNER_ON_SUPERMICRO=False

# ops=[op.get()[0]]
ops=op.get()
NUM_RUNS=len(ops)
NUM_NETS=2
num_sims=NUM_RUNS*NUM_NETS
kwargs={
        'amp_base':[1.2], #From ZZZ61
        
        'Builder':Builder,
        
        'cores_milner':40*1,
        'cores_superm':10,
        
        'debug':False,
        'do_runs':range(NUM_RUNS), #A run for each perturbation
        'do_obj':False,
        
        'file_name':FILE_NAME,
        'freqs':[1.5],  #amplitude frequencies
        'freq_oscillation':20.,
        'from_disk_0':FROM_DISK_0,
        
        'i0':FROM_DISK_0,
        
        'job_name':'beta_ZZZ9',
        
        'l_hours':  ['00','00','00'],
        'l_minutes':['15','10','5'],
        'l_seconds':['00','00','00'],

        'local_threads_milner':20,
        'local_threads_superm':1,
        
        'module':module,
        
        'nets':['Net_0','Net_1'], #Nets for each run
        'no_oscillations_control':True,
        
        'path_code':default_params.HOME_CODE,
        'path_rate_runs':path_rate_runs,
        'path_results':get_path_logs(LOAD_MILNER_ON_SUPERMICRO, 
                                     FILE_NAME),
        'perturbation_list':ops,
        
        'sim_time':10000.0,
        'size':20000.0 ,
        }

d_process_and_thread=par_process_and_thread(**kwargs)
pp(d_process_and_thread)
kwargs.update(d_process_and_thread)


p_list = pert_add_oscillations(**kwargs)
p_list = pert_set_data_path_to_milner_on_supermicro(p_list,
                                                  LOAD_MILNER_ON_SUPERMICRO)

for i, p in enumerate(p_list): print i, p

a_list=get_args_list_oscillation(p_list, **kwargs)
k_list=get_kwargs_list_indv_nets(len(p_list), kwargs)

for i, obj in enumerate(a_list):
    print i, obj.kwargs['from_disk']

loop(10,[num_sims,num_sims,NUM_RUNS], a_list, k_list )

        