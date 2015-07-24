'''
Created on Aug 12, 2013

@author: lindahlm
'''

from scripts_inhibition.base_MSN_cluster_compete import Setup
from core.network.default_params import Perturbation_list as pl
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
import fig_01_and_02_pert as op
import pprint
import sys
pp=pprint.pprint

from copy import deepcopy

ops=[op.get()[1]]
FILE_NAME=__file__.split('/')[-1][0:-3]
FROM_DISK_0=int(sys.argv[1]) if len(sys.argv)>1 else 0
NUM_NETS=10
NUM_RUNS=1 #A run for each perturbation
num_sim=NUM_NETS*NUM_RUNS

dc=my_socket.determine_computer
CORES=40 if dc()=='milner' else 6
JOB_ADMIN=config.Ja_milner if dc()=='milner' else config.Ja_else
LOCAL_NUM_THREADS= 20 if dc()=='milner' else 5
WRAPPER_PROCESS=config.Wp_milner if dc()=='milner' else config.Wp_else

kwargs={
        'Builder':Builder,
        
        'cores':CORES,
        
        'file_name':FILE_NAME,
        'from_disk':FROM_DISK_0,
        
        'debug':False,
        'do_runs':range(1), #A run for each perturbation
        'do_obj':False,
        
        'i0':FROM_DISK_0,

        'job_admin':JOB_ADMIN, #user defined class
        'job_name':'fig_04B',
        
        'l_hours':  ['00','00','00'],
        'l_minutes':['15','10','5'],
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


loop(10,[NUM_NETS,NUM_NETS,1], a_list, k_list )

# d_process_and_thread=par_process_and_thread(**kwargs)
# pp(d_process_and_thread)
# kwargs.update(d_process_and_thread)
# 
# p_list = pert_add_MSN_cluster_compete(**kwargs)
# p_list = pert_set_data_path_to_milner_on_supermicro(p_list,
#                                                   LOAD_MILNER_ON_SUPERMICRO)
# 
# for i, p in enumerate(p_list): print i, p
# 
# a_list=get_args_list_inhibition(p_list, **kwargs)
# k_list=get_kwargs_list_indv_nets(len(p_list), kwargs)
# 
# def perturbations():
# 
#     threads=8
# 
#     l=[]
#     
# #     l.append(op.get()[0])
#     l.append(op.get()[7])
# 
#     ll=[]
#     
# 
#     l[-1]+=pl({'simu':{'threads':threads}},'=')
#              
#     return l, threads
# 
# 
# 
# rep=5
# p_list, threads=perturbations()
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
# #         if i!=1:
# #             continue
# 
#         from_disk=j
# 
#         fun=MSN_cluster_compete.main
#         script_name=(__file__.split('/')[-1][0:-3]+'/script_'+str(i)+'_'+p.name)
# #         fun(*[Builder, from_disk, p, script_name, 
# #               Setup(**{'threads':threads,
# #                         'repetition':rep})])
#         args_list.append([fun,script_name]
#                          +[Builder, from_disk, p, 
#                            script_name, 
#                            Setup(**{'threads':threads,
#                                     'repetition':rep})])
# 
# # for i, a in enumerate(args_list):
# #     print i, a
# 
# loop(args_list, path, 1)
        