'''
Created on Aug 12, 2013

@author: lindahlm
'''
from toolbox import monkey_patch as mp
mp.patch_for_milner()

from scripts_inhibition.simulate import (get_path_logs, 
                      pert_add,
                      pert_add_go_nogo_ss, 
                      par_process_and_thread,
                      get_args_list_Go_NoGo_compete,
                      get_kwargs_list_indv_nets)
from toolbox.network import default_params
from toolbox.network.manager import Builder_Go_NoGo_only_D1D2_nodop_FS as Builder
from toolbox.parallel_excecution import loop

import scripts_inhibition.Go_NoGo_compete as module


# from scripts inhibition import scripts_inhibition.Go_NoGo_compete as module
import oscillation_perturbations41_slow as op
import oscillation_perturbations_conns as op_conn
import oscillation_perturbations_nuclei as op_nuc
import oscillation_perturbations_dop as op_dop

import sys
import pprint
pp=pprint.pprint

# for p in [op_dop.get()[5]]:
#     print p


# op_pert_add=[op_dop.get()[5]]
# l_op_conn=[36, 66, 97, 114, 121, 127, 138]    
# l_op_nuc=[33, 49, 57]
# l_op_dop=[5,6]
l_op_conn=[12, 97, 108, 109, 127, 132 ]    
l_op_nuc=[16, 33, 49, 57, 64]
l_op_dop=[5,6]

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
kwargs={
         'ax_4x1':True,
        'add_midpoint':False,
        
        'Builder':Builder,
        
        'cores_milner':40*4,
        'cores_superm':12,
        
        'debug':False,
        'do_runs':range(10),
        'do_obj':False,
        'duration':[900.,100.0],
        'do_not_record':['M1', 'M2', 'FS','GA','GI', 'ST'], 
        
        'file_name':FILE_NAME,
        'from_disk_0':FROM_DISK_0,
        
        'i0':FROM_DISK_0,
        
        'job_name':'fig6_dcases',

        'l_hours':['06','01','00'],
        'l_mean_rate_slices':['mean_rate_slices'],
        'l_minutes':['00','00','05'],
        'l_seconds':['00','00','00'],            
        'labels':['D1,D2',], 
        'laptime':1000.0,
        'local_threads_milner':40,
        'local_threads_superm':4,
                 
        'max_size':20000.,
        'module':module,
        
        'nets':['Net_0'],
        
        'op_pert_add':op_pert_add,    
        'other_scenario':True,
        
        'path_code':default_params.HOME_CODE,
        'path_results':get_path_logs(LOAD_MILNER_ON_SUPERMICRO, 
                                     FILE_NAME),
        'perturbation_list':[op.get()[5]],
        'proportion_connected':[0.2]*NUM_NETS, #related to toal number fo runs
        'p_sizes':[
                   1.
                  ],
        'p_subsamp':[
                     1.
                     ],
        'res':10,
        'rep':40,
        
        'time_bin':100,
        }

d_process_and_thread=par_process_and_thread(**kwargs)
kwargs.update(d_process_and_thread)

p_list=pert_add_go_nogo_ss(**kwargs)
p_list=pert_add(p_list, **kwargs)

for i, p in enumerate(p_list): print i, p

a_list=get_args_list_Go_NoGo_compete(p_list, **kwargs)
k_list=get_kwargs_list_indv_nets(len(p_list), kwargs)

loop(2, [NUM_NETS,NUM_NETS,NUM_NETS], a_list, k_list )
        