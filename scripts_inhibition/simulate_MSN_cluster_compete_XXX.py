'''
Created on Aug 12, 2013

@author: lindahlm
'''

from MSN_cluster_compete import Setup
from toolbox.network.default_params import Perturbation_list as pl
from toolbox.network.manager import Builder_MSN_cluster_compete as Builder
from toolbox.parallel_excecution import loop

import MSN_cluster_compete
import oscillation_perturbations3 as op3
import pprint
pp=pprint.pprint

from copy import deepcopy

def perturbations():

    threads=2

    l=[]
    
#     l.append(op.get()[0])
    l.append(op3.get()[3])

    ll=[]
    

    l[-1]+=pl({'simu':{'threads':threads}},'=')
             
    return l, threads



rep=5
p_list, threads=perturbations()
for i, p in enumerate(p_list):
    print i, p
args_list=[]
 

from os.path import expanduser
home = expanduser("~")
   
path=(home + '/results/papers/inhibition/network/'
      +__file__.split('/')[-1][0:-3]+'/')

n=len(p_list)



for j in range(0,3):
    for i, p in enumerate(p_list):
        
# #         if i<n-9:
#         if i!=1:
#             continue

        from_disk=j

        fun=MSN_cluster_compete.main
        script_name=(__file__.split('/')[-1][0:-3]+'/script_'+str(i)+'_'+p.name)
#         fun(*[Builder, from_disk, p, script_name, 
#               Setup(**{'threads':threads,
#                         'repetition':rep})])
        args_list.append([fun,script_name]
                         +[Builder, from_disk, p, 
                           script_name, 
                           Setup(**{'threads':threads,
                                    'repetition':rep})])

# for i, a in enumerate(args_list):
#     print i, a

loop(args_list, path, 1)
        