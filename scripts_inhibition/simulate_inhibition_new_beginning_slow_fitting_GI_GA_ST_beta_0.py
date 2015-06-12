'''
Created on Aug 12, 2013

@author: lindahlm
'''
from toolbox import monkey_patch as mp
mp.patch_for_milner()

from toolbox.network.manager import Builder_striatum as Builder
from toolbox.parallel_excecution import loop
from toolbox import directories as dr

from simulate import (
                      get_args_list_inhibition,
                      get_kwargs_list_indv_nets,

                      pert_add_inhibition) 
from toolbox import my_socket

import config
import inhibition_striatum as module
from scripts_inhibition import oscillation_perturbations_new_beginning_slow_fitting_GI_GA_ST_beta_0 as op
import pprint
pp=pprint.pprint


FILE_NAME=__file__.split('/')[-1][0:-3]
FROM_DISK_0=0

NUM_NETS=1
ops=op.get()
NUM_RUNS=len(ops) #A run for each perturbation
num_sim=NUM_NETS*NUM_RUNS

dc=my_socket.determine_computer
CORES=40 if dc()=='milner' else 4
JOB_ADMIN=config.Ja_milner if dc()=='milner' else config.Ja_else
LOCAL_NUM_THREADS= 20 if dc()=='milner' else 4
WRAPPER_PROCESS=config.Wp_milner if dc()=='milner' else config.Wp_else

kwargs={
        'Builder':Builder,
                             
        'cores':CORES,
        
        'file_name':FILE_NAME,
        'from_disk_0':FROM_DISK_0,
        
        'debug':False,
        'do_runs':range(NUM_RUNS), #A run for each perturbation
        'do_obj':False,
                 
        'i0':FROM_DISK_0,
        
        'job_admin':JOB_ADMIN, #user defined class
        'job_name':'inh_YYY',
        
        'l_hours':  ['00','00','00'],
        'l_minutes':['15','10','5'],
        'l_seconds':['00','00','00'],
        
        'lower':1,
        'local_num_threads':LOCAL_NUM_THREADS,

        'module':module,    
        
        'nets':['Net_{}'.format(i) for i in range(NUM_NETS)],
        'nets_to_run':['Net_0'], 
        
        'resolution':5,
        'repetitions':1,
        
        'path_results':dr.HOME_DATA+ '/'+ FILE_NAME + '/',
        'perturbation_list':ops,
                
        'size':3000,
        
        'upper':3,
        
        'wrapper_process':WRAPPER_PROCESS, #user defined wrapper of subprocesses
        }


p_list = pert_add_inhibition(**kwargs)

for i, p in enumerate(p_list): print i, p

a_list=get_args_list_inhibition(p_list, **kwargs)
k_list=get_kwargs_list_indv_nets(len(p_list), kwargs)

for obj in a_list:
    print obj.kwargs['setup'].nets_to_run

# for i, a in enumerate(args_list):
#     print i, a
loop(min(num_sim, 1),[NUM_RUNS, NUM_RUNS, NUM_RUNS], a_list, k_list, **{'config':config} )


