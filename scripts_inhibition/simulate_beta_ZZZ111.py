'''
Created on Aug 12, 2013

@author: lindahlm

'''

from toolbox import monkey_patch as mp
mp.patch_for_milner()

from simulate import (get_type_of_run, get_path_rate_runs,
                      get_path_logs, get_runs_oscillation,
                      pert_set_data_path_to_milner_on_supermicro, 
                      pert_add_oscillations) 

from toolbox.network import default_params
from toolbox.network.manager import Builder_beta as Builder
from toolbox.parallel_excecution import loop

import simulate_beta
import oscillation_perturbations as op
import pprint
pp=pprint.pprint

FILE_NAME=__file__.split('/')[-1][0:-3]
FREQS=[0.5, 1.0, 1.5]
FREQ_OSCILLATION=20.
FROM_DISK_0=0

LOAD_MILNER_ON_SUPERMICRO=False

SIM_TIME=10000.0
SIZE=5000.0 
DO_RUNS=[0,1] #if not empty these runs are simulated
DO_OBJ=True
 
THREADS_MILNER=40 #Should be multiple of 20
THREADS_SUPERMICRO=10

perturbation_list=op.get()
type_of_run=get_type_of_run(shared=True) 
threads=get_threads(THREADS_SUPERMICRO, THREADS_MILNER)

path_code=default_params.HOME_CODE
path_rate_runs = get_path_rate_runs('simulate_inhibition_ZZZ/')
path_result_logs = get_path_logs(LOAD_MILNER_ON_SUPERMICRO, 
                             FILE_NAME)

p_list = pert_add_oscillations(SIM_TIME, SIZE, threads, 
                               FREQS, path_rate_runs, FREQ_OSCILLATION, 
                               perturbation_list)
 
p_list=pert_set_data_path_to_milner_on_supermicro(p_list,
                                                  LOAD_MILNER_ON_SUPERMICRO)

for i, p in enumerate(p_list): print i, p

args_list=get_runs_oscillation(Builder, DO_OBJ, DO_RUNS, FILE_NAME,
                               FREQ_OSCILLATION,  FROM_DISK_0, 
                               simulate_beta, p_list, threads,
                               type_of_run)

# for i, a in enumerate(args_list):
#     print i, a
n_tasks_per_node=20
loop(args_list, path_result_logs, path_code, 2,
     **{'type_of_run':type_of_run,
        'threads':threads,
        'memory_per_node':int(819*(40/n_tasks_per_node)),
        'n_nodes':int(threads/n_tasks_per_node),
        'n_tasks_per_node':n_tasks_per_node, # 40 is maximum
        'n_mpi_processes':threads,
        
        'i0':FROM_DISK_0, 
        'debug':False,
        'l_hours':['01','01','00'],
        'job_name':'_'.join(FILE_NAME.split('_')[1:]),
        'l_minutes':['00','00','5'],
        'l_seconds':['00','00','00']})

        