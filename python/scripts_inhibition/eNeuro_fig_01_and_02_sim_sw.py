'''
Created on Aug 12, 2013

@author: lindahlm

'''


# Patch for milner. Avoid importing nest in shell.
from core import monkey_patch as mp
mp.patch_for_milner()

# Helper modules for simulation
from scripts_inhibition.base_simulate import (
      get_path_rate_runs,
      get_args_list_oscillation,
      get_kwargs_list_indv_nets,
      pert_add_oscillations)

# Import builder defining a special build version of the model
# from the defautl model. In this case a model that have
# slow wave input instead of no modulation
from core.network.manager import Builder_slow_wave2 as Builder

# Tool for running multiple simulations at once (for example
# when you have different parameter combinations which
# results in multiple simulations. Here we only have to
# scenarios normal and dopamine depleted.)
from core.parallel_excecution import loop

# Set som global variables
from core import directories as dr

# Module for detemining computer type
from core import my_socket

# Job configureations fordifferent type of
# computers e.g. desktop vs milner supercomputer
from scripts_inhibition import config

# Default values for eNeuro simlkations
import eNeuro_fig_defaults as fd

# Model perturbations. You will end up with
# as many simulations as the number of
# perturbations lists you have.
import eNeuro_fig_01_and_02_pert as op

import numpy
import sys

import scripts_inhibition.base_oscillation_sw as module
import pprint
pp=pprint.pprint

path_rate_runs=get_path_rate_runs('eNeuro_fig_01_and_02_sim_inh/')
FILE_NAME=__file__.split('/')[-1][0:-3]
FROM_DISK_0=int(sys.argv[1]) if len(sys.argv)>1 else 0
LOAD_MILNER_ON_SUPERMICRO=False

NUM_NETS=2

amp_base=[fd.amp_sw] #numpy.arange(1.05, 1.2, 0.05)
freqs=[ fd.freq_sw ] #numpy.arange(0.5, .8, 0.2) // rate shift
#ops=op.get()['sw']
ops=[op.get()[fd.idx_sw]]
n=len(amp_base)
m=len(freqs)
amp_base=list(numpy.array([m*[v] for v in amp_base]).ravel()) 
freqs=list(freqs)*n
STN_amp_mod=[fd.STN_amp_mod_sw]#range(1, 6, 2)
num_runs=len(freqs)*len(STN_amp_mod)*len(ops)
num_sims=NUM_NETS*num_runs

dc=my_socket.determine_computer
CORES=40*4 if dc()=='milner' else 4
JOB_ADMIN=config.Ja_milner if dc()=='milner' else config.Ja_else
LOCAL_NUM_THREADS= 20 if dc()=='milner' else 4
WRAPPER_PROCESS=config.Wp_milner if dc()=='milner' else config.Wp_else

kwargs={
        'amp_base':amp_base,
        'amp_base_skip':['CS'],
        
        'Builder':Builder,
        
        'cores':CORES,
        
        'debug':False,
        'do_runs':range(num_runs), #A run for each perturbation
        'do_obj':True,
        
        'external_input_mod':[],
        
        'file_name':FILE_NAME,
        'freqs':freqs,
        'freq_oscillation':1.,
        'from_disk_0':FROM_DISK_0,
        
        'i0':FROM_DISK_0,
        
        'job_admin':JOB_ADMIN, #user defined class
        'job_name':'eNf1_2_sw',
        
        'l_hours':  ['00','01','00'],
        'l_minutes':['45','00','05'],
        'l_seconds':['00','00','00'],

        'local_num_threads':LOCAL_NUM_THREADS,
        
        'module':module,
        
        'nets':['Net_0','Net_1'], #The nets for each run
        'nets_to_run':['Net_0','Net_1'],
        
        'path_rate_runs':path_rate_runs,
        'path_results':dr.HOME_DATA+ '/'+ FILE_NAME + '/',
        'perturbation_list':ops,
                
        'sim_time':20000.0,
        'size':5000.0 ,
        'STN_amp_mod':STN_amp_mod,
        
        'tuning_freq_amp_to':'M2',
        
        'wrapper_process':WRAPPER_PROCESS, #user defined wrapper of subprocesses
        }

p_list = pert_add_oscillations(**kwargs)

for i, p in enumerate(p_list): 
    print i, p

a_list=get_args_list_oscillation(p_list, **kwargs)
k_list=get_kwargs_list_indv_nets(len(p_list), kwargs)

for i, obj in enumerate(a_list):
    print i, obj.kwargs['from_disk']

loop(num_sims,[num_sims,num_sims,num_sims/2], a_list, k_list )

        
