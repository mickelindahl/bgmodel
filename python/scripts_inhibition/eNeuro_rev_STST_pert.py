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
        for w, f in [[0.05, 0.9], [0.1,0.8], [0.2, 0.7], [0.3, 0.6], [0.4, 0.5], [0.5, 0.4],
                     [0.05, 0.95], [0.1,0.85], [0.2, 0.75], [0.3, 0.65], [0.4, 0.55], [0.5, 0.45],
                     [0.05, 1.], [0.1,0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5]]:
             
            d= {'node':{'CS':{'rate':f}}}
            d=misc.dict_update(p['mul'], d) 
            
            l[i]+=[pl(d, '*', **{'name':''})]
            
            d={'nest':{'ST_ST_ampa':{'weight': w}}}
            d=misc.dict_update(p['equal'], d) 
    
            s='STST_{0}_{1}'.format( w, labels[i] )
            
            l[i][-1]+=pl(d, '=', **{'name':s})     
    
    
    
    beta, sw=l
    return {'beta':l[0], 'sw':l[1]}
    

ld=get()
pp(ld)