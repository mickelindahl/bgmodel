'''
Created on Aug 12, 2013

@author: lindahlm
'''

from toolbox.network.manager import Builder_striatum as Builder
from toolbox.parallel_excecution import loop
from toolbox import directories as dr

from simulate import (
                      get_args_list_inhibition,
                      get_kwargs_list_indv_nets,
                      par_process_and_thread,
                      pert_add_inhibition) 
from toolbox import my_socket

import config
import inhibition_striatum as module
import oscillation_perturbations0 as op
import pprint
pp=pprint.pprint


FILE_NAME=__file__.split('/')[-1][0:-3]
FROM_DISK_0=0

NUM_NETS=1
ops=op.get()
NUM_RUNS=len(ops) #A run for each perturbation
num_sim=NUM_NETS*NUM_RUNS

JOB_ADMIN=config.Ja_else #if my_socket.determine_computer()=='milner' else config.Ja_else
PROCESS_TYPE='else' #if my_socket.determine_computer()=='milner' else 'else'
WRAPPER_PROCESS=config.Wp_else #if my_socket.determine_computer()=='milner' else config.Wp_else

kwargs={
        'Builder':Builder,
                             
        'cores_milner':40*1,
        'cores_else':2,
        
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
        'local_threads_milner':20,
        'local_threads_else':2,

        'module':module,    
        
        'nets':['Net_{}'.format(i) for i in range(NUM_NETS)],
        
        'resolution':5,
        'repetitions':1,
        
        'path_results':dr.HOME_DATA+ '/'+ FILE_NAME + '/',
        'perturbation_list':ops,
        'process_type':PROCESS_TYPE,
                
        'size':3000,
        
        'upper':3,
        
        'wrapper_process':WRAPPER_PROCESS, #user defined wrapper of subprocesses
        }

d_process_and_thread=par_process_and_thread(**kwargs)
pp(d_process_and_thread)
kwargs.update(d_process_and_thread)

p_list = pert_add_inhibition(**kwargs)

for i, p in enumerate(p_list): print i, p

a_list=get_args_list_inhibition(p_list, **kwargs)
k_list=get_kwargs_list_indv_nets(len(p_list), kwargs)

for obj in a_list:
    print obj.kwargs['setup'].nets_to_run

# for i, a in enumerate(args_list):
#     print i, a
loop(min(num_sim, 10),[num_sim, num_sim, NUM_RUNS], a_list, k_list, **{'config':config} )


