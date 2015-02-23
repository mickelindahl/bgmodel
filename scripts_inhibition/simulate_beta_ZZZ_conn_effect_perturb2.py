'''
Created on Aug 12, 2013

@author: lindahlm

'''

from toolbox import monkey_patch as mp
mp.patch_for_milner()

from simulate import (get_path_rate_runs,
                      get_path_logs, get_args_list_oscillation,
                      get_kwargs_list_indv_nets,
                      pert_add,
                      par_process_and_thread,
                      pert_set_data_path_to_milner_on_supermicro, 
                      pert_add_oscillations) 

from toolbox.network import default_params
from toolbox.network.manager import Builder_beta_EI_EA as Builder
from toolbox.parallel_excecution import loop

import sys
import simulate_beta as module
import oscillation_perturbations13 as op
import oscillation_perturbations_conns as op_conns
import pprint
pp=pprint.pprint

FILE_NAME=__file__.split('/')[-1][0:-3]
FROM_DISK_0=int(sys.argv[1]) if len(sys.argv)>1 else 0
LOAD_MILNER_ON_SUPERMICRO=False

#Total number of runs 18*2*2+18
NUM_NETS=len(op_conns.get()*2)
path_rate_runs=get_path_rate_runs('simulate_inhibition_ZZZ13/')

kwargs={
        
        'amp_base':[1.15],
        
        'Builder':Builder,
        
        'cores_milner':40*1,
        'cores_superm':40,

        'debug':False,#173-86, 109-54, 45-22
        'do_runs':range(NUM_NETS/2), #A run for each perturbation
        'do_obj':False,

        'external_input_mod':['EI','EA'],
                
        'file_name':FILE_NAME,
        'freqs':[0.8],
        'freq_oscillation':20.,
        'from_disk_0':FROM_DISK_0,
        
        'i0':FROM_DISK_0,
        
        'job_name':'be_conn_pert2',
        
        'l_hours':  ['00','00','00'],
        'l_minutes':['45','45','5'],
        'l_seconds':['00','00','00'],

        'local_threads_milner':10,
        'local_threads_superm':4,
        
        'module':module,
        
        'nets':['Net_'+str(i) for i in range(2)], #The nets for each run
        'no_oscillations_control':True,
        
        'op_pert_add':op_conns.get(),
        
        'path_code':default_params.HOME_CODE,
        'path_rate_runs':path_rate_runs,
        'path_results':get_path_logs(LOAD_MILNER_ON_SUPERMICRO, 
                                     FILE_NAME),
        'perturbation_list':op.get(),
        
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
p_list=pert_add(p_list, **kwargs)

for i, p in enumerate(p_list): 
    print i, p
#     for l in p.list:
#         if len(l.val)==2:
#             print l

a_list=get_args_list_oscillation(p_list, **kwargs)
k_list=get_kwargs_list_indv_nets(len(p_list), kwargs)


loop(10,[NUM_NETS, NUM_NETS, NUM_NETS/2], a_list, k_list )

        