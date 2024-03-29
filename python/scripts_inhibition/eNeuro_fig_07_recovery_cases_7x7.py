'''
Created on Aug 12, 2013

@author: lindahlm
'''
from core import monkey_patch as mp
mp.patch_for_milner()

from scripts_inhibition import config
from scripts_inhibition.base_simulate import (
                      pert_add,
                      pert_add_go_nogo_ss, 
                      get_args_list_Go_NoGo_compete_oscillation,
                      get_kwargs_list_indv_nets,
                      get_path_rate_runs)

from core.network.manager import Builder_Go_NoGo_only_D1D2_nodop_FS_oscillations as Builder
from core.parallel_excecution import loop
from core import directories as dr
from core import my_socket
from core.network.default_params import Perturbation_list as pl

import eNeuro_fig_defaults as fd
import eNeuro_fig_01_and_02_pert as op
import eNeuro_fig_07_pert_conn as op_conn
import eNeuro_fig_07_pert_nuclei as op_nuc
import eNeuro_fig_03_pert_dop as op_dop
import scripts_inhibition.base_Go_NoGo_compete as module
import sys
import pprint
pp=pprint.pprint

path_rate_runs=get_path_rate_runs('eNeuro_fig_01_and_02_sim_inh/')
ops=[op.get()[fd.idx_beta]] # 0 is beta
# l_op_conn=[17, 51, 57, 67, 107, 137, 161, 167, 171, 177, 181, 187, 201, 211, 217, 221, 227]#12, 97, 108, 109, 127, 132 ]    
# l_op_nuc=[ 8, 16, 17, 25, 32, 33, 40, 49, 56, 57, 64]#16, 33, 49, 57, 64]
l_op_conn=[8, 11, 12, 35, 37, 38, 40, 47, 48, 49, 50] #12, 97, 108, 109, 127, 132 ]    
l_op_nuc=[2,4,5,8,9, 10,11,13,15,16]#16, 33, 49, 57, 64]

l_op_dop=range(3,23)

l_op_conn2=[4]
l_op_nuc2=[11,12,14]
# [4, #CTX-M1
#           5, #CTX-M2"
#           6, #MS-MS
#           9, #FS-M2
#           10, #CTX-ST
#           12, #GP
#           14, #GP-GP
#           17] #M2-TI

'''
Issuse wtih:
2 (51)     GA_M2_pert_0.0    OOM
5 (137)    GA_FS_pert_5      time
7 (161)    ST_GA_pert_0.0    OOM
8 (171)    ST_GA_pert_5      time
10 (187)   GI_GA_pert_0.0    ?
11 (201)   M2_GI_pert_0.0    OOM
12 (211)   M2_GI_pert_5      OOM
15 (16)    M1_pert_mod7      OOM
18 (40)    M2_pert_mod7      Mem python
19 (49)    GI_pert_mod0      OOM
20 (56)    GI_pert_mod7      ?
21 (57)    ST_pert_mod0      OOM
22 (64)    ST_pert_mod7      ?
'''

op_pert_add=[pl(**{'name':'Control'})]

opc=op_conn.get([0,6])
op_pert_add+=[opc[i] for i in l_op_conn]


opn=op_nuc.get([0,7])
op_pert_add+=[opn[i] for i in l_op_nuc]

opn=op_dop.get()
op_pert_add+=[opn[i] for i in l_op_dop]

opc=op_conn.get([0,6])
op_pert_add+=[opc[i] for i in l_op_conn2]


opn=op_nuc.get([0,7])
op_pert_add+=[opn[i] for i in l_op_nuc2]

for i, o in enumerate(op_pert_add):
    print i, o

FILE_NAME=__file__.split('/')[-1][0:-3]
FROM_DISK_0=int(sys.argv[1]) if len(sys.argv)>1 else 0
LOAD_MILNER_ON_SUPERMICRO=False
NUM_RUNS=len(op_pert_add)


dc=my_socket.determine_computer
CORES=40*6 if dc()=='milner' else 10
JOB_ADMIN=config.Ja_milner if dc()=='milner' else config.Ja_else
LOCAL_NUM_THREADS= 40 if dc()=='milner' else 10
WRAPPER_PROCESS=config.Wp_milner if dc()=='milner' else config.Wp_else

amp_base=fd.amp_beta#1.1
freq= fd.freq_beta#0.6 
STN_amp_mod=fd.STN_amp_mod_beta#3.
kwargs={
        
        'amp_base':amp_base,
        'amp_base_skip':['CS'],
           
        'Builder':Builder,
        
        'cores':CORES,
        
        'debug':False,
        'do_runs':[42],#range(NUM_RUNS-3, NUM_RUNS),#[2,5,7,8,10,11,12,15,18,19,20,21,22],#range(NUM_RUNS),#NUM_NETS),
        'do_obj':False,

        'do_not_record':['M1', 'M2', 'FS','GA','GI', 'ST'], 
        'file_name':FILE_NAME,
        'freqs':[freq], #need to be length  1
        'freq_oscillations':20.,
        'from_disk_0':FROM_DISK_0,
        
        'i0':FROM_DISK_0,
        'input_type':'burst3_oscillations',
        
        'job_admin':JOB_ADMIN, #user defined clas
        'job_name':'fig7_rec',

        'l_hours':['04','03','00'],
        'l_mean_rate_slices':['mean_rate_slices'],
        'l_minutes':['00','00','05'],
        'l_seconds':['00','00','00'],            
        'labels':['D1,D2',], 
        
        'local_num_threads':LOCAL_NUM_THREADS,
                 
        'max_size':20000.,
        'module':module,
        
        'nets':['Net_0'],
        'nets_to_run':['Net_0'],
        
        'op_pert_add':op_pert_add,    
        'other_scenario':True,
                 
        'path_rate_runs':path_rate_runs,
        'path_results':dr.HOME_DATA+ '/'+ FILE_NAME + '/',      
        
        'perturbation_list':ops,
        'proportion_connected':[0.2]*NUM_RUNS, #related to toal number fo runs
        'p_sizes':[1. ],
        'p_subsamp':[1. ],
        
        'STN_amp_mod':STN_amp_mod,
 
        'tuning_freq_amp_to':'M2',
        'wrapper_process':WRAPPER_PROCESS, #user defined wrapper of subprocesses
         
        }

if my_socket.determine_computer()=='milner':
    kw_add={
            'duration':[907.,100.0],            
            'laptime':1007.0,
            'res':7,
            'rep':10,
            'time_bin':10000.,

            }
elif my_socket.determine_computer() in ['thalamus','supermicro']:
    kw_add={
            'duration':[357., 100.0],
            'laptime':457.,
            'res':3, 
            'rep':5,
            'time_bin':1000/256.,
            }

kwargs.update(kw_add)

p_list=pert_add_go_nogo_ss(**kwargs)
p_list=pert_add(p_list, **kwargs)

for i, p in enumerate(p_list): print i, p

a_list=get_args_list_Go_NoGo_compete_oscillation(p_list, **kwargs)
k_list=get_kwargs_list_indv_nets(len(p_list), kwargs)

loop(NUM_RUNS, [NUM_RUNS,NUM_RUNS,NUM_RUNS], a_list, k_list )
        