'''
Created on Aug 12, 2013

@author: lindahlm
'''

from Go_NoGo_compete import Setup
from toolbox.network.default_params import Perturbation_list as pl
from toolbox.network.manager import Builder_Go_NoGo_with_lesion as Builder
from toolbox.parallel_excecution import loop

import Go_NoGo_compete
import oscillation_perturbations as op
import oscillation_perturbations3 as op3
import pprint
pp=pprint.pprint

from copy import deepcopy

def perturbations(rep,res):

    threads=2

    l=[]
    
    l.append(op.get()[0])
    l.append(op3.get()[3])

    ll=[]
    for ss in [25, 50, 75]: 
        
        for i, _l in enumerate(l):
            _l=deepcopy(_l)
            per=pl({'netw':{
                            'sub_sampling':{'M1':ss,
                                            'M2':ss},}},
                      '=', 
                      **{'name':'ss-'+str(ss)})
            _l+=per
    
            _l+=pl({'simu':{'threads':threads}},'=')
            ll.append(_l)
            
    return ll, threads



res, rep=5, 1
p_list, threads=perturbations(rep, res)
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
#         if i>18:
#             continue

        from_disk=j

        fun=Go_NoGo_compete.main
        script_name=(__file__.split('/')[-1][0:-3]+'/script_'+str(i)+'_'+p.name)
#         fun(*[Builder, from_disk, p, script_name, 
#               Setup(**{'threads':threads,
#                         'resolution':res,
#                         'repetition':rep})])
        args_list.append([fun,script_name]
                         +[Builder, from_disk, p, 
                           script_name, 
                           Setup(**{'threads':threads,
                                    'resolution':res,
                                    'repetition':rep})])

# for i, a in enumerate(args_list):
#     print i, a

loop(args_list, path, 6)
        