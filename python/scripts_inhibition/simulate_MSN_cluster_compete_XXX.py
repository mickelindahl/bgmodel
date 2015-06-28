'''
Created on Aug 12, 2013

@author: lindahlm
'''

from MSN_cluster_compete import Setup
from core.network.default_params import Perturbation_list as pl
from core.network.manager import Builder_MSN_cluster_compete as Builder
from core.parallel_excecution import loop
from core.network import default_params

from scripts_inhibition.simulate import (get_path_logs, 
                      get_args_list_inhibition,
                      get_kwargs_list_indv_nets,
                      par_process_and_thread,
                      pert_set_data_path_to_milner_on_supermicro, 
                      pert_add_MSN_cluster_compete) 

import MSN_cluster_compete as module
import oscillation_perturbations4 as op
import pprint
pp=pprint.pprint

from copy import deepcopy

FILE_NAME=__file__.split('/')[-1][0:-3]
FROM_DISK_0=0
LOAD_MILNER_ON_SUPERMICRO=False
NUM_NETS=10
NUM_RUNS=1 #A run for each perturbation
num_sim=NUM_NETS*NUM_RUNS

kwargs={
        'Builder':Builder,
        
        'cores_milner':40*1,
        'cores_superm':20,
        
        
        'from_disk':0,
        
        'debug':False,
        'do_runs':range(NUM_NETS/2), #A run for each perturbation
        'do_obj':False,
        
        'i0':FROM_DISK_0,
        
        'job_name':'inh_YYY',
        
        'l_hours':  ['00','00','00'],
        'l_minutes':['15','10','5'],
        'l_seconds':['00','00','00'],
        
        'lower':1,
        'local_threads_milner':20,
        'local_threads_superm':5,

        
        'module':module,    
        
        'nets':['Net_{}'.format(i) for i in range(NUM_RUNS)],
        
        'repetitions':5,
        
        'path_code':default_params.HOME_CODE,
        'path_results':get_path_logs(LOAD_MILNER_ON_SUPERMICRO, 
                                     FILE_NAME),
        'perturbation_list':[op.get()[7]],
        
        'size':3000,
        
        'upper':1.5}

d_process_and_thread=par_process_and_thread(**kwargs)
pp(d_process_and_thread)
kwargs.update(d_process_and_thread)

p_list = pert_add_MSN_cluster_compete(**kwargs)
p_list = pert_set_data_path_to_milner_on_supermicro(p_list,
                                                  LOAD_MILNER_ON_SUPERMICRO)

for i, p in enumerate(p_list): print i, p

a_list=get_args_list_inhibition(p_list, **kwargs)
k_list=get_kwargs_list_indv_nets(len(p_list), kwargs)

def perturbations():

    threads=8

    l=[]
    
#     l.append(op.get()[0])
    l.append(op.get()[7])

    ll=[]
    

    l[-1]+=pl({'simu':{'threads':threads}},'=')
             
    return l, threads



rep=5
p_list, threads=perturbations()
for i, p in enumerate(p_list):
    print i, p
args_list=[]
 

from os.path import expanduser
home = expanduser("~")
   
path=(home + '/results/papers/inhibition/network/'
      +__file__.split('/')[-1][0:-3]+'/')

n=len(p_list)



for j in range(2,3):
    for i, p in enumerate(p_list):
        
# #         if i<n-9:
#         if i!=1:
#             continue

        from_disk=j

        fun=MSN_cluster_compete.main
        script_name=(__file__.split('/')[-1][0:-3]+'/script_'+str(i)+'_'+p.name)
#         fun(*[Builder, from_disk, p, script_name, 
#               Setup(**{'threads':threads,
#                         'repetition':rep})])
        args_list.append([fun,script_name]
                         +[Builder, from_disk, p, 
                           script_name, 
                           Setup(**{'threads':threads,
                                    'repetition':rep})])

# for i, a in enumerate(args_list):
#     print i, a

loop(args_list, path, 1)
        