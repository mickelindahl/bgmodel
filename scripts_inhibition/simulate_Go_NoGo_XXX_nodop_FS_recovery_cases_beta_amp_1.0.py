'''
Created on Aug 12, 2013

@author: lindahlm
'''
from toolbox import monkey_patch as mp
mp.patch_for_milner()

from simulate import (get_path_logs, 
                      pert_add,
                      pert_add_go_nogo_ss, 
                      par_process_and_thread,
                      get_args_list_Go_NoGo_compete_oscillation,
                      get_kwargs_list_indv_nets,
                      get_path_rate_runs)
from toolbox.network import default_params
from toolbox.network.manager import Builder_Go_NoGo_only_D1D2_nodop_FS_oscillations as Builder
from toolbox.parallel_excecution import loop
from toolbox import my_socket
import scripts_inhibition.Go_NoGo_compete as module


# from scripts inhibition import Go_NoGo_compete as module
import oscillation_perturbations41_slow as op
import oscillation_perturbations_conns as op_conn
import oscillation_perturbations_nuclei as op_nuc
import oscillation_perturbations_dop as op_dop

import sys
import pprint
pp=pprint.pprint

# for p in [op_dop.get()[5]]:
#     print p


path_rate_runs=get_path_rate_runs('simulate_inhibition_ZZZ41_slow/')
oscillation_returbations_index=5
# op_pert_add=[op_dop.get()[5]]
# l_op_conn=[36, 66, 97, 114, 121, 127, 138]    
# l_op_nuc=[33, 49, 57]
# l_op_dop=[5,6]
l_op_conn=[]#12, 97, 108, 109, 127, 132 ]    
l_op_nuc=[]#16, 33, 49, 57, 64]
l_op_dop=[-1]#,6,6]

opc=op_conn.get()
op_pert_add=[opc[i] for i in l_op_conn]

opn=op_nuc.get()
op_pert_add+=[opn[i] for i in l_op_nuc]

opn=op_dop.get()
op_pert_add+=[opn[i] for i in l_op_dop]


for i, o in enumerate(op_pert_add):
    print i, o

FILE_NAME=__file__.split('/')[-1][0:-3]
FROM_DISK_0=int(sys.argv[1]) if len(sys.argv)>1 else 0
LOAD_MILNER_ON_SUPERMICRO=False
NUM_NETS=len(l_op_conn)+len(l_op_nuc)+len(l_op_dop)


amp_base=1.0
freqs= 0.6 
STN_amp_mod=3.
kwargs={
        
        'amp_base':amp_base,
        
#         'ax_4x1':True,
#         'add_midpoint':False,
        
        'Builder':Builder,
        
        'cores_milner':40*4,
        'cores_superm':40,
        
        'debug':False,
        'do_runs':range(10),
        'do_obj':False,

#         'do_not_record':['M1', 'M2', 'FS','GA','GI', 'ST'], 
        'do_not_record':[],#['FS','GA','GI', 'ST'], 
        'file_name':FILE_NAME,
        'freqs':freqs,
        'freq_oscillations':20.,
        'from_disk_0':FROM_DISK_0,
        
        'i0':FROM_DISK_0,
        'input_type':'burst3_oscillations',
        
        'job_name':'fig6_dco',

        'l_hours':['06','01','00'],
        'l_mean_rate_slices':['mean_rate_slices'],
        'l_minutes':['00','00','05'],
        'l_seconds':['00','00','00'],            
        'labels':['D1,D2',], 
        
      
        'local_threads_milner':40,
        'local_threads_superm':4,
                 
        'max_size':20000.,
        'module':module,
        
        'nets':['Net_0'],
        
        'op_pert_add':op_pert_add,    
#         'oscillation_returbations_index':oscillation_returbations_index,
        'other_scenario':True,
        
        'path_code':default_params.HOME_CODE,
        'path_results':get_path_logs(LOAD_MILNER_ON_SUPERMICRO, 
                                     FILE_NAME),
        'path_rate_runs':path_rate_runs,
        'perturbation_list':[op.get()[oscillation_returbations_index]],
        'proportion_connected':[0.2]*NUM_NETS, #related to toal number fo runs
        'p_sizes':[
                   1.
                  ],
        'p_subsamp':[
                     1.
                     ],
        
        'STN_amp_mod':STN_amp_mod,
        
        }

if my_socket.determine_computer()=='milner':
    kw_add={
            'duration':[907.,100.0],            
            'laptime':1007.0,
            'res':10,
            'rep':40,
            'time_bin':100.,

            }
elif my_socket.determine_computer()=='supermicro':
    kw_add={
            'duration':[357., 100.0],
            'laptime':457.,
            'res':3, 
            'rep':5,
            'time_bin':1000/256.,
            }
kwargs.update(kw_add)

d_process_and_thread=par_process_and_thread(**kwargs)
kwargs.update(d_process_and_thread)

p_list=pert_add_go_nogo_ss(**kwargs)
p_list=pert_add(p_list, **kwargs)

for i, p in enumerate(p_list): print i, p

a_list=get_args_list_Go_NoGo_compete_oscillation(p_list, **kwargs)
k_list=get_kwargs_list_indv_nets(len(p_list), kwargs)

loop(1, [NUM_NETS,NUM_NETS,NUM_NETS], a_list, k_list )
        