'''
Created on Aug 12, 2013

@author: lindahlm
'''

from Go_NoGo_compete import Setup
from toolbox.network.default_params import Perturbation_list as pl
from toolbox.network.manager import Builder_Go_NoGo_with_lesion_FS as Builder
from toolbox.parallel_excecution import loop

import Go_NoGo_compete
import oscillation_perturbations4 as op
import pprint
pp=pprint.pprint

from copy import deepcopy

def perturbations(rep,res):

    threads=12

    l=[]
    
#     l.append(op.get()[0])
#     l.append(op.get()[4::3])
    l.append(op.get()[4+3])

    ll=[]
    
    p_sizes=[
#              0.1989460102,  
            0.1608005821,    
#             0.122655154, 
#             0.0845097259
              ]
    p_sizes=[p/p_sizes[0] for p in p_sizes]
    max_size=4000
    for ss, p_size in zip([
                           6.25, 
                           8.3 , 
                           12.5, 
                           25
                           ], p_sizes): 
        
        for i, _l in enumerate(l):
            _l=deepcopy(_l)
            per=pl({'netw':{'size':int(p_size*max_size), 
                            'sub_sampling':{'M1':ss,
                                            'M2':ss},}},
                      '=', 
                      **{'name':'ss-'+str(ss)})
            _l+=per
    
            _l+=pl({'simu':{'threads':threads}},'=')
            ll.append(_l)
    
    for _l in deepcopy([ll[0]]):
        _l=deepcopy(_l)
        _l.update_list(
            pl({'nest':{'GA_M1_gaba':{'weight':5*2.*0.4},
#                         'GA_FS_gaba':{'weight':0.1}
                        }},'*', **{'name':'GA-XX-equal'}))
        ll.append(_l)
 
        _l=deepcopy(_l)
        _l.update_list(
            pl({'nest':{'GA_M1_gaba':{'weight':5*4.*0.4},
#                         'GA_FS_gaba':{'weight':0.1}
                        }},'*', **{'name':'GA-XX-equal'}))
        ll.append(_l)
    
    return ll, threads



res, rep=10, 1
duration=[900.,100.0]
laptime=1000.0
l_mean_rate_slices= ['mean_rate_slices']
p_list, threads=perturbations(rep, res)
for i, p in enumerate(p_list):
    print i, p
args_list=[]
 

from os.path import expanduser
home = expanduser("~")
   
path=(home + '/results/papers/inhibition/network/'
      +__file__.split('/')[-1][0:-3]+'/')

n=len(p_list)



for j in range(2, 3):
    for i, p in enumerate(p_list):
#         
#         if i<3:
# #         if i!=1:
#             continue

        from_disk=j

        fun=Go_NoGo_compete.main
        script_name=(__file__.split('/')[-1][0:-3]+'/script_'+str(i)+'_'+p.name)
#         fun(*[Builder, from_disk, p, script_name, 
#               Setup(**{'duration':duration,
#                     'l_mean_rate_slices':l_mean_rate_slices,
#                      'threads':threads,
#                      'resolution':res,
#                      'repetition':rep})])

        args_list.append([fun,script_name]
                         +[Builder, from_disk, p, 
                           script_name, 
                           Setup(**{'duration':duration,
                                    'laptime':laptime,
                                    'l_mean_rate_slices':l_mean_rate_slices,
                                    'threads':threads,
                                    'resolution':res,
                                    'repetition':rep})])

# for i, a in enumerate(args_list):
#     print i, a

loop(args_list, path, 1)
        