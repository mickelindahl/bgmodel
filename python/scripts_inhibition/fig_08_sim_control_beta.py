'''
Created on Aug 12, 2013

@author: lindahlm

'''

from core import monkey_patch as mp
mp.patch_for_milner()

from scripts_inhibition.base_simulate import (get_path_rate_runs,
                      get_args_list_oscillation,
                      get_kwargs_list_indv_nets,
                      pert_add_oscillations,
                      pert_add) 

from core import directories as dr
from core.network.manager import Builder_beta as Builder
from core.parallel_excecution import loop
from core import my_socket

from scripts_inhibition import config
import numpy
import sys
import scripts_inhibition.base_oscillation_beta as module
import fig_01_and_02_pert as op
import fig_08_pert_control as op_control
import pprint
pp=pprint.pprint

path_rate_runs=get_path_rate_runs('fig_01_and_02_sim_inh/')
FILE_NAME=__file__.split('/')[-1][0:-3]
FROM_DISK_0=int(sys.argv[1]) if len(sys.argv)>1 else 0
LOAD_MILNER_ON_SUPERMICRO=False

NUM_NETS=2

amp_base=[1.1] #numpy.arange(1.05, 1.2, 0.05)
freqs=[0.4] #numpy.arange(0.5, .8, 0.2)
n=len(amp_base)
m=len(freqs)
ops=[op.get()[0]]
STN_amp_mod=[3.]#range(1, 6, 2)

amp_base=list(numpy.array([m*[v] for v in amp_base]).ravel()) 
freqs=list(freqs)*n
num_runs=len(freqs)*len(STN_amp_mod)*len(ops)*len(op_control.get())
num_sims=NUM_NETS*num_runs

dc=my_socket.determine_computer
CORES=40 if dc()=='milner' else 10
JOB_ADMIN=config.Ja_milner if dc()=='milner' else config.Ja_else
LOCAL_NUM_THREADS= 20 if dc()=='milner' else 10
WRAPPER_PROCESS=config.Wp_milner if dc()=='milner' else config.Wp_else

kwargs={
        'amp_base':amp_base,
        'amp_base_skip':['CS'],
        
        'Builder':Builder,
        
        'cores':CORES,
        
        'debug':False,
        'do_runs':range(num_runs), #A run for each perturbation
        'do_obj':False,
        
        'external_input_mod':[],
        
        'file_name':FILE_NAME,
        'freqs':freqs,
        'freq_oscillation':20.,
        'from_disk_0':FROM_DISK_0,
        
        'i0':FROM_DISK_0,
        
        'job_admin':JOB_ADMIN, #user defined class
        'job_name':'fig08_beta',
        
        'l_hours':  ['00','01','00'],
        'l_minutes':['45','00','05'],
        'l_seconds':['00','00','00'],

        'local_num_threads':LOCAL_NUM_THREADS,
        
        'module':module,
        
        'nets':['Net_0','Net_1'], #The nets for each run
        'nets_to_run':['Net_0','Net_1'],
        'no_oscillations_control':True,
        
        'op_pert_add':op_control.get(),
        
        'path_rate_runs':path_rate_runs,
        'path_results':dr.HOME_DATA+ '/'+ FILE_NAME + '/',
        'perturbation_list':ops,
        
        'sim_time':40000.0,
        'size':20000.0 ,
        
        'STN_amp_mod':STN_amp_mod,

        'tuning_freq_amp_to':'M2',

        'wrapper_process':WRAPPER_PROCESS, #user defined wrapper of subprocesses
        }

p_list = pert_add_oscillations(**kwargs)
p_list=pert_add(p_list, **kwargs)

for i, p in enumerate(p_list): 
    print i, p

a_list=get_args_list_oscillation(p_list, **kwargs)
k_list=get_kwargs_list_indv_nets(len(p_list), kwargs)

for i, obj in enumerate(a_list):
    print i, obj.kwargs['from_disk']

loop(30,[num_sims,num_sims,num_sims/2], a_list, k_list )

        