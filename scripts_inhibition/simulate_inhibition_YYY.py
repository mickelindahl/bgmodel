'''
Created on Aug 12, 2013

@author: lindahlm
'''

from toolbox.network.manager import Builder_inhibition_striatum as Builder
from toolbox.parallel_excecution import loop
from toolbox.network import default_params

from simulate import (get_path_logs, 
                      get_args_list_inhibition,
                      get_kwargs_list_indv_nets,
                      par_process_and_thread,
                      pert_set_data_path_to_milner_on_supermicro, 
                      pert_add_inhibition) 

import inhibition_striatum as module
import oscillation_perturbations4 as op
import pprint
pp=pprint.pprint

FILE_NAME=__file__.split('/')[-1][0:-3]
FROM_DISK_0=0
LOAD_MILNER_ON_SUPERMICRO=False
NUM_NETS=6
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
        
        'nets':['Net_{}'.format(i) for i in range(NUM_NETS)],
        
        'resolution':14,
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

p_list = pert_add_inhibition(**kwargs)
p_list = pert_set_data_path_to_milner_on_supermicro(p_list,
                                                  LOAD_MILNER_ON_SUPERMICRO)

for i, p in enumerate(p_list): print i, p

a_list=get_args_list_inhibition(p_list, **kwargs)
k_list=get_kwargs_list_indv_nets(len(p_list), kwargs)

for obj in a_list:
    print obj.kwargs['setup'].nets_to_run

# 
# def perturbations(rep,res):
#     sim_time=rep*res*1000.0
#     size=3000.0
#     threads=16
# 
#     
#     l=[op.get()[7]]
# 
#     for i in range(len(l)):
#         l[i]+=pl({'simu':{'sim_time':sim_time,
#                           'sim_stop':sim_time,
#                           'threads':threads},
#                   'netw':{'size':size}},
#                   '=')
#         
#     return l, threads
# 
# 
# 
# res, rep, low, upp=14, 5, 1, 2.
# p_list, threads=perturbations(rep, res)
# for i, p in enumerate(p_list):
#     print i, p
# args_list=[]
#  
# 
# from os.path import expanduser
# home = expanduser("~")
#    
# path=(home + '/results/papers/inhibition/network/'
#       +__file__.split('/')[-1][0:-3]+'/')
# 
# n=len(p_list)
# 
# 
# 
# for j in range(2,3):
#     for i, p in enumerate(p_list):
#         
# # #         if i<n-9:
# #         if i>18:
# #             continue
# 
#         from_disk=j
# 
#         fun=inhibition_striatum.main
#         script_name=(__file__.split('/')[-1][0:-3]+'/script_'+str(i)+'_'+p.name)
# #         fun(*[Builder, from_disk, p, script_name, 
# #               Setup(threads, res, rep, low, upp)])
#         args_list.append([fun,script_name]
#                          +[Builder, from_disk, p, 
#                            script_name, 
#                            Setup(**{'threads':threads,
#                                     'resolution':res,
#                                     'repetition':rep,
#                                     'lower':low,
#                                     'upper':upp})])

# for i, a in enumerate(args_list):
#     print i, a
loop(2,[NUM_NETS,NUM_NETS,1], a_list, k_list )
# loop(args_list, path, 1)
        