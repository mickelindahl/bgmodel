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
from toolbox.network.manager import Builder_slow_wave2 as Builder
from toolbox.parallel_excecution import loop


import numpy
import simulate_slow_wave as module
import sys
import oscillation_perturbations41_slow_sw as op
import pprint
pp=pprint.pprint


path_rate_runs=get_path_rate_runs('simulate_inhibition_ZZZ41_slow_sw/')
FILE_NAME=__file__.split('/')[-1][0:-3]
FROM_DISK_0=int(sys.argv[1]) if len(sys.argv)>1 else 2
LOAD_MILNER_ON_SUPERMICRO=False
ops=op.get()[4:7]
NUM_NETS=2
amp_base=[0.85]#numpy.arange(.85, 1.0, 0.05)
freqs=[0.45, 0.5, .55 ]#numpy.arange(0.5, .8, 0.2)
# amp_base=[1.0] #numpy.arange(0.7, 1.1, 0.05)
# freqs=numpy.arange(0.1, 1.0, 0.1)


n=len(amp_base)
m=len(freqs)
amp_base=list(numpy.array([m*[v] for v in amp_base]).ravel()) 
freqs=list(freqs)*n

num_runs=len(freqs)*len(ops)
num_sims=NUM_NETS*num_runs

kwargs={
        'amp_base':amp_base,
        
        'Builder':Builder,
        
        'cores_milner':40*1,
        'cores_superm':40,
        
        'debug':False,
#         'database_save':True,
        'do_runs':range(num_runs), #A run for each perturbation
        'do_obj':False,
        
        'external_input_mod':['EA', 'EI'],
        
        'file_name':FILE_NAME,
        'freqs':freqs,
        'freq_oscillation':1.,
        'from_disk_0':FROM_DISK_0,
        
        'i0':FROM_DISK_0,
        
        'job_name':'sw_ZZZ35_slow_sw',
        
        'l_hours':  ['00','01','00'],
        'l_minutes':['45','00','05'],
        'l_seconds':['00','00','00'],

        'local_threads_milner':10,
        'local_threads_superm':4,
        
        'module':module,
        
        'nets':['Net_'+str(i) for i in range(2)], #The nets for each run
  
        'path_code':default_params.HOME_CODE,
        'path_rate_runs':path_rate_runs,
        'path_results':get_path_logs(LOAD_MILNER_ON_SUPERMICRO, 
                                     FILE_NAME),
        'perturbation_list':ops,
        
        'sim_time':40000.0,
        'size':20000.0,
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

print 'from disk', FROM_DISK_0


loop(20,[num_sims,num_sims,num_sims/2], a_list, k_list )
        