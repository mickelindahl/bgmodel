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
        for f1, f2, decay in [
                            [1., .0, 12.*5],   
                            [.8, .2, 12.*5],    
                            [.6, .4, 12.*5],
                            [.5, .5, 12.*5], 
                            [.4, .6, 12.*5],    
                            [.2, .8, 12.*5],  
                            [.1, .9, 12.*5], 
                            [.05, .95, 12.*5], 
                            [.01, .99, 12.*5], 
                            [.005, .995, 12.*5],
                            [.001, .999, 12.*5],
                            [.0, 1., 12.*5],
                            
                            [1., .0, 12.*5],   
                            [.8, .0, 12.*5],    
                            [.6, .0, 12.*5],
                            [.5, .0, 12.*5], 
                            [.4, .0, 12.*5],    
                            [.2, .0, 12.*5],  
                            [.1, .0, 12.*5], 
                            [.05, .0, 12.*5], 
                            [.01, .0, 12.*5], 
                            [.005, .0, 12.*5],
                            [.001, .0, 12.*5],
                            [.0, .0, 12.*5],
                            
                            
                            [.5, .5, 12.*5],
                            [.5, .5, 12.*4],
                            [.5, .5, 12.*3],
                            [.5, .5, 12.*2],
                            [.5, .5, 12.*1],
                           ]:
             
            
            d={'nest':{'GA_FS_gaba':{'weight':f1},
                       'GF_FS_gaba':{'weight':f2}},  
               'node':{'CF':{'rate':1.}}}
                    
            d=misc.dict_update(p['mul'], d) 
            
            l[i]+=[pl(d, '*', **{'name':''})]
            
            d={'conn':{'GF_FS_gaba':{'lesion':False}},
               'nest':{'FS_low':{'GABAA_2_Tau_decay':decay}}}
            d=misc.dict_update(p['equal'], d) 
    
            s='GA_{0}_GF_{1}_decay_{2}_{3}'.format( f1, f2, decay, labels[i] )
            
            l[i][-1]+=pl(d, '=', **{'name':s})     
    
    
    
    beta, sw=l
    return {'beta':l[0], 'sw':l[1]}
    

ld=get()
pp(ld)

