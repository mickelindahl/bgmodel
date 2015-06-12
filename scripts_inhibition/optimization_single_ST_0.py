'''
Created on Jun 4, 2015

@author: mikael
'''
from toolbox import monkey_patch as mp
mp.patch_for_milner()

from simulate import (pert_add_single,
                      get_args_list_opt_single,
                      get_kwargs_list_indv_nets,
                      ) 

from toolbox.network.manager import Builder_single as Builder
# from toolbox.network.manager import Builder_beta_GA_GI_ST as Builder
from toolbox.parallel_excecution import loop
from toolbox import directories as dr
from toolbox import my_socket

from scripts_inhibition import config
import numpy
import sys
import optimization_single as module
import oscillation_perturbations_new_beginning_slow_fitting_ST as op
import pprint
pp=pprint.pprint

# path_rate_runs=get_path_rate_runs('simulate_inhibition_new_beginning_slow0/')
FILE_NAME=__file__.split('/')[-1][0:-3]
FROM_DISK_0=int(sys.argv[1]) if len(sys.argv)>1 else 0
# LOAD_MILNER_ON_SUPERMICRO=False

NUM_NETS=2

# amp_base=[0.9] #numpy.arange(1.05, 1.2, 0.05)
# freqs=[ 0.3] #numpy.arange(0.5, .8, 0.2)
# n=len(amp_base)
# m=len(freqs)
# amp_base=list(numpy.array([m*[v] for v in amp_base]).ravel()) 
# freqs=list(freqs)*n
# STN_amp_mod=[1.]#range(1, 6, 2)
# 
# num_runs=len(freqs)*len(STN_amp_mod)*len(ops)
ops=[op.get()[0]]*2
n_ops=len(ops)
num_runs=len(ops)
num_sims=NUM_NETS*num_runs

dc=my_socket.determine_computer
CORES=40 if dc()=='milner' else 1
JOB_ADMIN=config.Ja_milner if dc()=='milner' else config.Ja_else
LOCAL_NUM_THREADS= 40 if dc()=='milner' else 1
WRAPPER_PROCESS=config.Wp_milner if dc()=='milner' else config.Wp_else

kwargs={
#         'amp_base':amp_base,
        
        'Builder':Builder,
        
        'cores':CORES,
        
        'debug':False,
        'do_runs':range(num_runs), #A run for each perturbation
        'do_obj':False,
        'do_reset':False,
        
#         'external_input_mod':[],
        
        'file_name':FILE_NAME,
#         'freqs':freqs,
#         'freq_oscillation':1.,
        'from_disk_0':FROM_DISK_0,
        
        'i0':FROM_DISK_0,
        
        'job_admin':JOB_ADMIN, #user defined class
        'job_name':'opt_sw0',
        
        'l_hours':  ['01','01','00'],
        'l_minutes':['00','00','05'],
        'l_seconds':['00','00','00'],

        'local_num_threads':LOCAL_NUM_THREADS,
        
        'module':module,
        'nets_to_run':['Net_0', 'Net_1'],
        'nets':['Net_0', 'Net_1'], #The nets for each run
#         'no_oscillations_control':True,
        
        'opt':[
               {'Net_0':{'f':['ST'],
                         'x':['node.CSp.rate'],
                         'x0':[400.0]},
                'Net_1':{'f':['ST'],
                         'x':['node.CSp.rate'],
                         'x0':[400.0]}}]*n_ops, #Same size as ops

#         'opt':[{'Net_1':{'f':['GI'],
#                          'x':['node.M2p.rate'],
#                          'x0':[2.9]}}]*n_ops, #Same size as ops
        
#         'path_rate_runs':path_rate_runs,
        'path_results':dr.HOME_DATA+ '/'+ FILE_NAME + '/',
        'perturbation_list':ops,
        
        
        'sim_time':20000.0,
        'single_unit':['ST']*n_ops, #Same size as ops
        'size':1.0 ,
#         'STN_amp_mod':STN_amp_mod,
        
        'tp_names':['beta']*(n_ops/2)+['sw']*(n_ops/2), #Same size as ops
#         'tuning_freq_amp_to':'M2',
        
        'wrapper_process':WRAPPER_PROCESS, #user defined wrapper of subprocesses
        }


p_list = pert_add_single(**kwargs)

for i, p in enumerate(p_list): 
    print i, p

a_list=get_args_list_opt_single(p_list, **kwargs)
k_list=get_kwargs_list_indv_nets(len(p_list), kwargs)

for i, obj in enumerate(a_list):
    print i, obj.kwargs['from_disk']

loop(4,[num_sims], a_list, k_list )


