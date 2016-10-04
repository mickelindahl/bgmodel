'''
Created on Aug 12, 2013

@author: lindahlm
'''

from core.network.default_params import Perturbation_list as pl
from core import misc
from scripts_inhibition.base_perturbations import get_solution
from scripts_inhibition.fig_01_and_02_pert import get as _get

import numpy
import pprint

pp=pprint.pprint

d0=0.8
f_beta_rm=lambda f: (1-f)/(d0+f*(1-d0))

def get():
    
    l=[[],[]]
    labels=['beta', 'sw']
    for i, p in enumerate(_get('dictionary')):
    
    
        # Test STN-STN
        for f1, f2, f3 in [
                            [1., .0, 1.], 
#                             [.0, 1., 1.2],
#                             [.0, 1., 1.4], 
#                             [.0, 1., 1.6], 
#                             [.0, 1., 1.8], 
#                             [.0, 1., 2.], 
#                             [.0, 1., 2.2], 
#                             [.0, 1., 2.4], 
#                             [.0, 1., 2.6]
                            
#                             [.9, .1, 1.02],  
                            [.8, .2, 1.04],  
#                             [.7, .3, 1.06],  
                            [.6, .4, 1.08],
                            [.5, .5, 1.1], 
                            [.4, .6, 1.12],  
#                             [.3, .7, 1.14],  
                            [.2, .8, 1.16],  
#                             [.1, .9, 1.18], 
                            [.0, 1., 2.2]
                           ]:
             
            
            d={'nest':{'GA_FS_gaba':{'weight':f1},
                       'GF_FS_gaba':{'weight':f2}},
               'node':{'CF':{'rate':f3}}}
                    
            d=misc.dict_update(p['mul'], d) 
            
            l[i]+=[pl(d, '*', **{'name':''})]
            
            d={'conn':{'GF_FS_gaba':{'lesion':False}}}
            d=misc.dict_update(p['equal'], d) 
    
            s='GA_{0}_GF_{1}_FS_{2}_{3}'.format( f1, f2, f3, labels[i] )
            
            l[i][-1]+=pl(d, '=', **{'name':s})     
    
    
    
    beta, sw=l
    return {'beta':l[0], 'sw':l[1]}
    

ld=get()
pp(ld)

