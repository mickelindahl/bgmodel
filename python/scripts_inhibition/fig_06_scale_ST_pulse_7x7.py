'''
Created on Aug 12, 2013

@author: lindahlm
'''
from core import monkey_patch as mp
mp.patch_for_milner()

from scripts_inhibition import config
from scripts_inhibition.base_simulate import (
                      pert_add_go_nogo_ss, 
                      get_args_list_Go_NoGo_compete_oscillation,
                      get_kwargs_list_indv_nets,
                      get_path_rate_runs)

from core.network.manager import Builder_Go_NoGo_with_lesion_FS_ST_pulse_oscillation as Builder
from core.parallel_excecution import loop
from core import directories as dr
from core import my_socket

import fig_defaults as fd
import fig_01_and_02_pert as op
import scripts_inhibition.base_Go_NoGo_compete as module
import sys
import pprint
pp=pprint.pprint

path_rate_runs=get_path_rate_runs('fig_01_and_02_sim_inh/')
ops=[op.get()[fd.idx_beta]] #0 is beta

FILE_NAME=__file__.split('/')[-1][0:-3]
FROM_DISK_0=int(sys.argv[1]) if len(sys.argv)>1 else 0
LOAD_MILNER_ON_SUPERMICRO=False
proportion_connected=[0.1, 0.5, 1.]
NUM_RUNS=len(proportion_connected)
NUM_NETS=1
num_sims=NUM_NETS*NUM_RUNS

dc=my_socket.determine_computer
CORES=40*4if dc()=='milner' else 10
JOB_ADMIN=config.Ja_milner if dc()=='milner' else config.Ja_else
LOCAL_NUM_THREADS= 40 if dc()=='milner' else 10
WRAPPER_PROCESS=config.Wp_milner if dc()=='milner' else config.Wp_else

amp_base=fd.amp_beta#1.1
freq= 0.0
STN_amp_mod=fd.STN_amp_mod_beta#3.
kwargs={
        'amp_base':amp_base,
        
        'Builder':Builder,
        
        'cores':CORES,
        
        'debug':False,
        'do_not_record':[],#,['M1', 'M2', 'FS','GA','GI', 'ST'], 
        'do_runs':range(NUM_RUNS),
        'do_obj':False,
#         'duration':[900.,100.0],
        
        'file_name':FILE_NAME,
        'freqs':[freq], #need to be length  1
        'freq_oscillations':20.,
        'from_disk_0':FROM_DISK_0,
        
        'i0':FROM_DISK_0,
        'input_type':'burst3_oscillations',
        
        'job_admin':JOB_ADMIN, #user defined class    
        'job_name':'fig6_scl_STp',

        'l_mean_rate_slices':['mean_rate_slices'],
        'l_hours':  ['02','02','00'], #12 because of 1.0 proportion connected
        'l_minutes':['00','00','05'],
        'l_seconds':['00','00','00'],             
        'labels':['D1,D2 puls=5',], 
        
        'local_num_threads':LOCAL_NUM_THREADS,
                 
        'max_size':20000,
        'module':module,
        
        'nets':['Net_0'],
        'nets_to_run':['Net_0'],
                
        'other_scenario':True,
        
        'path_rate_runs':path_rate_runs,
        'path_results':dr.HOME_DATA+ '/'+ FILE_NAME + '/',
        'perturbation_list':ops,
        'proportion_connected':proportion_connected, #related to toal number fo runs
        
        'p_pulses':[5]*NUM_NETS, #size of labels

        'p_sizes':[1.]*NUM_RUNS,
        'p_subsamp':[1.]*NUM_RUNS,
        
        'STN_amp_mod':STN_amp_mod,
 
        'threshold':14.,
        'tuning_freq_amp_to':'M2',
        
        'wrapper_process':WRAPPER_PROCESS, #user defined wrapper of subprocesses

        }
if my_socket.determine_computer()=='milner':
    kw_add={
            'duration':[907.,100.0],            
            'laptime':1007.0,
            'res':7,
            'rep':10,
            'time_bin':100.,

            }
elif my_socket.determine_computer() in ['thalamus','supermicro']:
    kw_add={
            'duration':[357., 100.0],
            'laptime':457.,
            'res':3, 
            'rep':5,
            'time_bin':1000./256,
            }
    
kwargs.update(kw_add)

p_list=pert_add_go_nogo_ss(**kwargs)

for i, p in enumerate(p_list): print i, p

a_list=get_args_list_Go_NoGo_compete_oscillation(p_list, **kwargs)
k_list=get_kwargs_list_indv_nets(len(p_list), kwargs)

loop(3, [num_sims,num_sims,NUM_RUNS], a_list, k_list )