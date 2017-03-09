'''
Created on Aug 12, 2013

@author: lindahlm
'''

from core.network.default_params import Perturbation_list as pl
from core import misc
from scripts_inhibition.eNeuro_fig_01_and_02_pert import get as _get

import numpy
import pprint

pp=pprint.pprint

d0=0.8
f_beta_rm=lambda f: (1-f)/(d0+f*(1-d0))

def get():
    
    l=[[],[]]
    labels=['beta', 'sw']
    for i, p in enumerate(_get('dictionary')):
    
    
        for f in [1., 2., 2.5, 3., 3.5,  4.]:
            
            
            # TA-MS decay as Glajch 2016
            d={'nest':{
                       'GA_M1_gaba':{'weight':f},
                       'GA_M2_gaba':{'weight':f}}}
            d=misc.dict_update( p['mul'], d) 
             
            l[i]+=[pl(d, '*', **{'name':''})]
            # Dopamine effect on GA-MS
            d={}
            
            d=misc.dict_update(p['equal'], d) 
    
            s='TAMS_{0}_{1}'.format(f,  labels[i] )
            
            l[i][-1]+=pl(d, '=', **{'name':s})   
   
   
    
    beta, sw=l
    return {'beta':l[0], 'sw':l[1]}

ld=get()
pp(ld)
