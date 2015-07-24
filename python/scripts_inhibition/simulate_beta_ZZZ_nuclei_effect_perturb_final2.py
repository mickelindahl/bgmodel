'''
Created on Aug 12, 2013

@author: lindahlm

'''

from core import monkey_patch as mp
mp.patch_for_milner()

from scripts_inhibition.base_simulate import (get_path_rate_runs,
                      get_path_logs, get_args_list_oscillation,
                      get_kwargs_list_indv_nets,
                      pert_add,
                      par_process_and_thread,
                      pert_add_oscillations) 

from core.network import default_params
from core.network.manager import Builder_beta as Builder
from core.parallel_excecution import loop

import sys
import scripts_inhibition.base_oscillation_beta as module
import oscillation_perturbations41_slow as op
import oscillation_perturbations_nuclei as op_neuclei
import pprint
pp=pprint.pprint

FILE_NAME=__file__.split('/')[-1][0:-3]
FROM_DISK_0=int(sys.argv[1]) if len(sys.argv)>1 else 0
LOAD_MILNER_ON_SUPERMICRO=False



amp_base=[1.05] #numpy.arange(1.05, 1.2, 0.05)
freqs=[0.7] #numpy.arange(0.5, .8, 0.2)
#Total number of runs 18*2*2+18
STN_amp_mod=[3.]
NUM_NETS=2
NUM_RUNS=len(op_neuclei.get()) #A run for each perturbation
num_sim=NUM_NETS*NUM_RUNS
path_rate_runs=get_path_rate_runs('simulate_inhibition_ZZZ41_slow/')

kwargs={
        
        'amp_base':amp_base,
        
        'Builder':Builder,
        
        'cores_milner':40*1,
        'cores_superm':40,
        
        'debug':False,#173-86, 109-54, 45-22
        'do_runs':range(NUM_RUNS), #A run for each perturbation
        'do_obj':False,
        
        'file_name':FILE_NAME,
        'freqs':freqs,
        'freq_oscillation':20.,
        'from_disk_0':FROM_DISK_0,
        
        'external_input_mod':['EI','EA'],
        
        'i0':FROM_DISK_0,
        
        'job_name':'be_nepf',
        
        'l_hours':  ['00','00','00'],
        'l_minutes':['45','45','5'],
        'l_seconds':['00','00','00'],

        'local_threads_milner':10,
        'local_threads_superm':4,
        
        'module':module,
        
        'nets':['Net_'+str(i) for i in range(NUM_NETS)], #The nets for each run
        
        'no_oscillations_control':True,
        
        'op_pert_add':op_neuclei.get(),
        
        'path_code':default_params.HOME_CODE,
        'path_rate_runs':path_rate_runs,
        'path_results':get_path_logs(LOAD_MILNER_ON_SUPERMICRO, 
                                     FILE_NAME),
        'perturbation_list':[op.get()[5]],
        
        'sim_time':40000.0,
        'size':20000.0 ,

        'STN_amp_mod':STN_amp_mod,
        }

d_process_and_thread=par_process_and_thread(**kwargs)
pp(d_process_and_thread)
kwargs.update(d_process_and_thread)

pp(kwargs)

p_list = pert_add_oscillations(**kwargs)

p_list=pert_add(p_list, **kwargs)

for i, p in enumerate(p_list): 
    print i, p
#     for l in p.list:
#         if len(l.val)==2:
#             print l

a_list=get_args_list_oscillation(p_list, **kwargs)
k_list=get_kwargs_list_indv_nets(len(p_list), kwargs)


loop(10,[num_sim, num_sim, NUM_RUNS], a_list, k_list )

        