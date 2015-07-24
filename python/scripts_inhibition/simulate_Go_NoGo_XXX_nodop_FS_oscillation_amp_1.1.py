'''
Created on Aug 12, 2013

@author: lindahlm
'''
from core import monkey_patch as mp

mp.patch_for_milner()

from scripts_inhibition.base_simulate import (pert_add_go_nogo_ss, get_path_logs, 
                      par_process_and_thread,
                      get_args_list_Go_NoGo_compete,
                      get_kwargs_list_indv_nets)
from core.network import default_params
from core.network.manager import Builder_Go_NoGo_with_nodop_FS_oscillation as Builder
from core.parallel_excecution import loop
from core import my_socket

import sys
import scripts_inhibition.base_Go_NoGo_compete as module
import oscillation_perturbations41_slow as op
import pprint
pp=pprint.pprint


FILE_NAME=__file__.split('/')[-1][0:-3]
FROM_DISK_0=int(sys.argv[1]) if len(sys.argv)>1 else 0
LOAD_MILNER_ON_SUPERMICRO=False
NUM_NETS=2

amp_base=1.1
freqs= 0.0
STN_amp_mod=3.
kwargs={
        'Builder':Builder,

        'amp_base':amp_base,

        'cores_milner':40*4,
        'cores_superm':40,
        
        'debug':False,
        'do_runs':range(NUM_NETS/2),
        'do_obj':False,
        
        'do_not_record':[],#['M1', 'M2', 'FS','GA','GI', 'ST'], 
        
        'file_name':FILE_NAME,
        'freqs':freqs,
        'freq_oscillations':20.,
        'from_disk_0':FROM_DISK_0,
        
        'i0':FROM_DISK_0,
        'input_type':'burst3_oscillations',
         
        'job_name':'fig6_nodop',


        'l_mean_rate_slices':['mean_rate_slices'],
        'l_hours':['05','01','00'],
        'l_minutes':['00','00','05'],
        'l_seconds':['00','00','00'],            
        'labels':['Only D1 no dop', 
                  'D1,D2 no dop'], 

        'local_threads_milner':40,
        'local_threads_superm':4,
                 
        'max_size':20000.,
        'module':module,
        
        'nets':['Net_0', 'Net_1'],
        
        'other_scenario':True,
        
        'path_code':default_params.HOME_CODE,
        'path_results':get_path_logs(LOAD_MILNER_ON_SUPERMICRO, 
                                     FILE_NAME),
        'perturbation_list':[op.get()[5]],
        'proportion_connected':[0.2]*NUM_NETS, #related to toal number fo runs
        'p_sizes':[
                   1.
                  ],
        'p_subsamp':[
                     1.
                     ],
        'STN_amp_mod':STN_amp_mod,
        
        }
if my_socket.determine_computer()=='milner':
    kw_add={
            'duration':[907.,100.0],            
            'laptime':1007.0,
            'res':10,
            'rep':40,
            'time_bin':100.,

            }
elif my_socket.determine_computer()=='supermicro':
    kw_add={
            'duration':[357., 100.0],
            'laptime':457.,
            'res':3, 
            'rep':5,
            'time_bin':1000/256.,
            }
kwargs.update(kw_add)

d_process_and_thread=par_process_and_thread(**kwargs)
kwargs.update(d_process_and_thread)

p_list=pert_add_go_nogo_ss(**kwargs)

for i, p in enumerate(p_list): print i, p

a_list=get_args_list_Go_NoGo_compete(p_list, **kwargs)
k_list=get_kwargs_list_indv_nets(len(p_list), kwargs)


loop(1, [NUM_NETS, NUM_NETS, 1], a_list, k_list )
        