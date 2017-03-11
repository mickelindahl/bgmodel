'''
Created on Aug 12, 2013

@author: lindahlm
'''

from core import monkey_patch as mp
mp.patch_for_milner()

from core.network.manager import Builder_MSN_cluster_compete as Builder
from core.parallel_excecution import loop
from core import directories as dr

from scripts_inhibition.base_simulate import (
                      get_args_list_inhibition,
                      get_kwargs_list_indv_nets,
                      pert_add_MSN_cluster_compete) 
from core import my_socket

import config
import scripts_inhibition.base_MSN_cluster_compete as module
import eNeuro_fig_01_and_02_pert as op
import pprint
import sys
pp=pprint.pprint


FILE_NAME=__file__.split('/')[-1][0:-3]
FROM_DISK_0=int(sys.argv[1]) if len(sys.argv)>1 else 0

NUM_NETS=10
ops=op.get()
NUM_RUNS=len(ops) #A run for each perturbation
num_sims=NUM_NETS*NUM_RUNS

dc=my_socket.determine_computer
CORES=40 if dc()=='milner' else 4
JOB_ADMIN=config.Ja_milner if dc()=='milner' else config.Ja_else
LOCAL_NUM_THREADS= 40 if dc()=='milner' else 4 # seems to be important to put 40 here for milner
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
        'job_name':'fig_04B',
        
        'l_hours':  ['02','01','00'],
        'l_minutes':['00','00','5'],
        'l_seconds':['00','00','00'],
        
        'lower':1,
        'local_num_threads':LOCAL_NUM_THREADS,
 
        'module':module,    
        
        'nets':['Net_{}'.format(i) for i in range(NUM_NETS)],
        'nets_to_run':['Net_{}'.format(i) for i in range(NUM_NETS)],       
      
        'repetitions':5,
        
        'path_results':dr.HOME_DATA+ '/'+ FILE_NAME + '/',
        'perturbation_list':ops,
        
        'size':3000,
        
        'upper':1.5,
        
        'wrapper_process':WRAPPER_PROCESS, #user defined wrapper of subprocesses
        }

p_list = pert_add_MSN_cluster_compete(**kwargs)

for i, p in enumerate(p_list): print i, p

a_list=get_args_list_inhibition(p_list, **kwargs)
k_list=get_kwargs_list_indv_nets(len(p_list), kwargs)

for obj in a_list:
    print obj.kwargs['setup'].nets_to_run


loop(5, [num_sims, num_sims,NUM_RUNS], a_list, k_list )

      