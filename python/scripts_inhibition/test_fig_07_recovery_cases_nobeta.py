'''
Created on Aug 12, 2013

@author: lindahlm
'''
from core import monkey_patch as mp
mp.patch_for_milner()

from scripts_inhibition import config
from scripts_inhibition.base_simulate import (pert_add,
                                         pert_add_go_nogo_ss,
                      get_path_rate_runs,
                      get_args_list_Go_NoGo_compete,
                      get_kwargs_list_indv_nets)

from core.network.manager import Builder_Go_NoGo_only_D1D2_nodop_FS as Builder
from core.parallel_excecution import loop
from core import directories as dr
from core import my_socket
from core.network.default_params import Perturbation_list as pl


import scripts_inhibition.base_Go_NoGo_compete as module
import fig_01_and_02_pert as op
import fig_07_pert_conn as op_conn
import fig_07_pert_nuclei as op_nuc
import fig_03_pert_dop as op_dop

import sys
import pprint
pp=pprint.pprint

path_rate_runs=get_path_rate_runs('fig_01_and_02_sim_inh/')
ops=[op.get()[0]]

l_op_conn=[17, 51, 57, 107, 137, 161, 171, 177, 187, 201, 211, 217, 221]#12, 97, 108, 109, 127, 132 ]    
l_op_nuc=[32, 16, 17, 33, 40, 49, 56, 57, 64]#16, 33, 49, 57, 64]
l_op_dop=[5,6]


op_pert_add=[pl(**{'name':'Control'})]

opc=op_conn.get()
op_pert_add+=[opc[i] for i in l_op_conn]

opn=op_nuc.get()
op_pert_add+=[opn[i] for i in l_op_nuc]

opn=op_dop.get()
op_pert_add+=[opn[i] for i in l_op_dop]


for i, o in enumerate(op_pert_add):
    print i, o

FILE_NAME=__file__.split('/')[-1][0:-3]
FROM_DISK_0=int(sys.argv[1]) if len(sys.argv)>1 else 0
LOAD_MILNER_ON_SUPERMICRO=False
NUM_NETS=len(op_pert_add)


dc=my_socket.determine_computer
CORES=40*4 if dc()=='milner' else 2
JOB_ADMIN=config.Ja_milner if dc()=='milner' else config.Ja_else
LOCAL_NUM_THREADS= 40 if dc()=='milner' else 2
WRAPPER_PROCESS=config.Wp_milner if dc()=='milner' else config.Wp_else


kwargs={
        'Builder':Builder,
        
        'cores':CORES,
        
        'debug':False,
        'do_not_record':[], 
        'do_runs':[1],#range(5),
        'do_obj':False,
#         'duration':[900.,100.0],
        
        'file_name':FILE_NAME,
#         'freqs':[0.5, 1.0, 1.5],
#         'freq_oscillation':20.,
        'from_disk_0':FROM_DISK_0,
        
        'i0':FROM_DISK_0,
        
        'job_admin':JOB_ADMIN, #user defined class   
        'job_name':'fig6_0.2rest',

        'l_hours':['10','02','00'],
        'l_mean_rate_slices':['mean_rate_slices'],
        'l_minutes':['00','00','05'],
        'l_seconds':['00','00','00'],            
        'labels':['D1,D2',], 
        
#         'laptime':1000.0,

        'local_num_threads':LOCAL_NUM_THREADS,
                 
        'max_size':5000.,
        'module':module,
        
        'nets':['Net_0'],
        'nets_to_run':['Net_0'],

        'op_pert_add':op_pert_add,  
        'other_scenario':True,
        
        'path_rate_runs':path_rate_runs,
        'path_results':dr.HOME_DATA+ '/'+ FILE_NAME + '/',
        'perturbation_list':ops,
        
        'proportion_connected':[0.2]*NUM_NETS, #related to toal number fo runs
        'p_sizes':[1.], #*NUM_NETS,
        'p_subsamp':[1.],#*NUM_NETS,
#         'res':10,
#         'rep':40,
        
        'time_bin':100,

        'wrapper_process':WRAPPER_PROCESS, #user defined wrapper of subprocesses

        }


if my_socket.determine_computer()=='milner':
    kw_add={
            'duration':[907.,100.0],            
            'laptime':1007.0,
            'res':10,
            'rep':40,
            'time_bin':100.,

            }
elif my_socket.determine_computer()in ['supermicro', 'thalamus']:
    kw_add={
            'duration':[357., 100.0],
            'laptime':457.,
            'res':3, 
            'rep':5,
            'time_bin':1000./256,
            }

kwargs.update(kw_add)

p_list=pert_add_go_nogo_ss(**kwargs)
p_list=pert_add(p_list, **kwargs)

for i, p in enumerate(p_list): print i, p

a_list=get_args_list_Go_NoGo_compete(p_list, **kwargs)
k_list=get_kwargs_list_indv_nets(len(p_list), kwargs)

loop(5, [NUM_NETS,NUM_NETS,NUM_NETS], a_list, k_list )
        