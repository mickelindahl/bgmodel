'''
Created on Aug 12, 2013

@author: lindahlm

'''

from core import monkey_patch as mp
mp.patch_for_milner()

from scripts_inhibition.simulate import (get_path_rate_runs,
                      get_path_logs, get_args_list_oscillation,
                      get_kwargs_list_indv_nets,
                      pert_add,
                      par_process_and_thread,
                      pert_add_oscillations) 

from core.network import default_params
from core.network.manager import Builder_slow_wave2 as Builder
from core.parallel_excecution import loop

import scripts_inhibition.base_oscillation_sw as module
import oscillation_perturbations4 as op
import oscillation_perturbations_conns as op_conns
import pprint
pp=pprint.pprint

FILE_NAME=__file__.split('/')[-1][0:-3]
FROM_DISK_0=0
LOAD_MILNER_ON_SUPERMICRO=False

#Total number of runs 18*2*2+18
NUM_NETS=139*2

kwargs={
        
        'amp_base':[1.],
        
        'Builder':Builder,
        
        'cores_milner':40*1,
        'cores_superm':40,
        
        'debug':False,#173-86, 109-54, 45-22
        'do_runs':range(NUM_NETS/2), #A run for each perturbation
        'do_obj':False,
        
        'file_name':FILE_NAME,
        'freqs':[0.75],
        'freq_oscillation':1.,
        'from_disk_0':FROM_DISK_0,
        
        'i0':FROM_DISK_0,
        
        'job_name':'sw_conn_pert',
        
        'l_hours':  ['00','00','00'],
        'l_minutes':['45','10','5'],
        'l_seconds':['00','00','00'],

        'local_threads_milner':10,
        'local_threads_superm':4,
        
        'module':module,
        
        'nets':['Net_'+str(i) for i in range(2)], #The nets for each run
        
        'op_pert_add':op_conns.get(),
        
        'path_code':default_params.HOME_CODE,
        'path_rate_runs':get_path_rate_runs('simulate_inhibition_ZZZ4/'),
        'path_results':get_path_logs(LOAD_MILNER_ON_SUPERMICRO, 
                                     FILE_NAME),
        'perturbation_list':[op.get()[7]],
        
        'sim_time':40000.0,
        'size':20000.0 ,
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


loop(30,[NUM_NETS, NUM_NETS, NUM_NETS/2], a_list, k_list )

        