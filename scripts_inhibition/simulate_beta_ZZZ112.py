'''
Created on Aug 12, 2013

@author: lindahlm

'''

from toolbox import monkey_patch as mp
mp.patch_for_milner()
28126 
from simulate import (get_threads_postprocessing, get_type_of_run, get_path_rate_runs,
                      get_path_logs, get_runs_oscillation,
                      par_mpi_sim,
                      pert_set_data_path_to_milner_on_supermicro, 
                      pert_add_oscillations) 

from toolbox.network import default_params
from toolbox.network.manager import Builder_beta as Builder
from toolbox.parallel_excecution import loop

import simulate_beta as module
import oscillation_perturbations as op
import pprint
pp=pprint.pprint

FILE_NAME=__file__.split('/')[-1][0:-3]
FREQS=[0.5, 1.0, 1.5]
FREQ_OSCILLATION=20.
FROM_DISK_0=0

LOAD_MILNER_ON_SUPERMICRO=False

SIM_TIME=10000.0
SIZE=20000.0 
DO_RUNS=[0] #if not empty these runs are simulated
DO_OBJ=False


NUM_CORES_MILNER=40*2 #multiple of 40 for milner
NUM_CORES_SUPERM=20
NUM_LOCAL_THREADS_MILNER=10


NO_MPI_PROCESSES=20 #Should be multiple of 20
NO_MPI_THREADS_PER_PROCESS=1
NO_SHARED_MEMORY_THREADS=2





SHARED=False

total_no_threads_mpi=NO_MPI_PROCESSES*NO_MPI_THREADS_PER_PROCESS

perturbation_list=op.get()
type_of_run=get_type_of_run(shared=SHARED) 
print type_of_run
v=get_threads_postprocessing(NO_SHARED_MEMORY_THREADS, total_no_threads_mpi, SHARED)
no_threads_postprocessing=v
# threads=get_threads(THREADS_SUPERMICRO, THREADS_MILNER)

path_code=default_params.HOME_CODE
path_rate_runs = get_path_rate_runs('simulate_inhibition_ZZZ/')
path_result_logs = get_path_logs(LOAD_MILNER_ON_SUPERMICRO, 
                             FILE_NAME)

p_list = pert_add_oscillations(
                               FREQS, 
                               FREQ_OSCILLATION, 
                               NO_MPI_THREADS_PER_PROCESS, 
                               NO_SHARED_MEMORY_THREADS, 
                               path_rate_runs, 
                               perturbation_list,
                               SIM_TIME, 
                               SIZE, 
                               )
 
p_list=pert_set_data_path_to_milner_on_supermicro(p_list,
                                                  LOAD_MILNER_ON_SUPERMICRO)

for i, p in enumerate(p_list): print i, p

args_list=get_runs_oscillation(Builder, 
                               DO_OBJ, 
                               DO_RUNS, 
                               FILE_NAME,
                               FREQ_OSCILLATION,  
                               FROM_DISK_0, 
                               module,
                               no_threads_postprocessing, 
                               p_list, 
                               type_of_run)

kwargs_list=[]

l_hours=['01','01','00']
l_minutes=['00','00','5']
l_seconds=['00','00','00']

n=0
for args in args_list:
    for _ in args:
        n+=1
 
# i=FROM_DISK_0
for j in range(n*(3-FROM_DISK_0)):
    
    i=int(j/n)+FROM_DISK_0
    print i,j
    if i==0:
        
        d=par_mpi_sim(NUM_CORES)
        dd={
            'n_threads_shared_memory':NO_SHARED_MEMORY_THREADS,
            'depth':NO_MPI_THREADS_PER_PROCESS,
            'memory_per_node':int(819*NO_MPI_THREADS_PER_PROCESS),
            'n_nodes':int(total_no_threads_mpi/40),
            'n_tasks_per_node':40/NO_MPI_THREADS_PER_PROCESS, # 40 is maximum
            'n_mpi_processes':NO_MPI_PROCESSES,
#             'n_threads_shared_memory':NO_SHARED_MEMORY_THREADS,
#             'depth':2,
#             'memory_per_node':int(819*2),
#             'n_nodes':int(2),
#             'n_tasks_per_node':20, # 40 is maximum
#             'n_mpi_processes':40
            }
    else:
        n_tasks_per_node=20
        dd={'n_threads_shared_memory':NO_SHARED_MEMORY_THREADS,
            'depth':1,
            'memory_per_node':int(819*(40/n_tasks_per_node)),
            'n_nodes':int(total_no_threads_mpi/40),
            'n_tasks_per_node':n_tasks_per_node, # 40 is maximum
            'n_mpi_processes':total_no_threads_mpi/2}
        
    d={'type_of_run':type_of_run,
        'i0':FROM_DISK_0, 
        'debug':False,
        'job_name':'_'.join(FILE_NAME.split('_')[1:]),
        'hours':l_hours[i],
        'seconds':l_seconds[i],
        'minutes':l_minutes[i]
        }
    d.update(dd)
    pp(d)
    kwargs_list.append(d)
    

# for i, a in enumerate(args_list):
#     print i, a

loop(args_list, path_result_logs, path_code, 1, kwargs_list)

        