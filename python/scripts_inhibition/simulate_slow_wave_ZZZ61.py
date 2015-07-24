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
from core.network.manager import Builder_slow_wave2 as Builder
from core.parallel_excecution import loop


import numpy
import scripts_inhibition.base_oscillation_sw as module
import oscillation_perturbations6 as op
import pprint
pp=pprint.pprint

FILE_NAME=__file__.split('/')[-1][0:-3]
FROM_DISK_0=1
LOAD_MILNER_ON_SUPERMICRO=False

NUM_NETS=2*29
amp_base=numpy.arange(0.8, 1.6, 0.1)
freqs=numpy.arange(0.5, 3, 0.2)
n=len(amp_base)
amp_base=[n*[v] for v in amp_base] 
freqs=freqs*n

kwargs={
        'amp_base':amp_base,
        
        'Builder':Builder,
        
        'cores_milner':40*1,
        'cores_superm':40,
        
        'debug':False,
        'do_runs':range(NUM_NETS/2), #A run for each perturbation
        'do_obj':False,
        
        'file_name':FILE_NAME,
        'freqs':freqs,
        'freq_oscillation':1.,
        'from_disk_0':FROM_DISK_0,
        
        'i0':FROM_DISK_0,
        
        'job_name':'sw_ZZZ6',
        
        'l_hours':  ['00','00','00'],
        'l_minutes':['45','10','5'],
        'l_seconds':['00','00','00'],

        'local_threads_milner':10,
        'local_threads_superm':4,
        
        'module':module,
        
        'nets':['Net_'+str(i) for i in range(2)], #The nets for each run
  
        
        'path_code':default_params.HOME_CODE,
        'path_rate_runs':get_path_rate_runs('simulate_inhibition_ZZZ6/'),
        'path_results':get_path_logs(LOAD_MILNER_ON_SUPERMICRO, 
                                     FILE_NAME),
        'perturbation_list':[op.get()[0]],
        
        'sim_time':40000.0,
        'size':20000.0 ,
        }

d_process_and_thread=par_process_and_thread(**kwargs)
pp(d_process_and_thread)
kwargs.update(d_process_and_thread)

pp(kwargs)

p_list = pert_add_oscillations(**kwargs)
p_list = pert_set_data_path_to_milner_on_supermicro(p_list,
                                                  LOAD_MILNER_ON_SUPERMICRO)

for i, p in enumerate(p_list): print i, p

a_list=get_args_list_oscillation(p_list, **kwargs)
k_list=get_kwargs_list_indv_nets(len(p_list), kwargs)

loop(10,[NUM_NETS,NUM_NETS,NUM_NETS/2], a_list, k_list )
        