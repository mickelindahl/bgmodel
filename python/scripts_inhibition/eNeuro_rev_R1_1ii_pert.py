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
    
    """
    First: To explain why increase firing rate STN is beneficial (DBS). 
    Simulate without GPe TA MSN and compare to simulation when this 
    connection is intact. Show how much the TA MSN contribute
    to the reduction in oscillations seen from increase in STN
    firing rate.
    
    Second: To explain why lesioning STN is benifitial. Simulate without 
    oscillatory input to STN and STN intact vs without oscillatory input 
    to striatum and STN lesioned. Compare all against intact dopamine 
    depleted model. Is it beause a reduction of oscillations through 
    STN from cortex, from MSNs through STN from striatum? Compare no 
    oscillations in to both striatum and STN for intact dopamine lesion 
    model vs when STN lesioned in is intact dopamine lesion model. Does
    the recution in firing rate make the system less prone to exhibit 
    oscillations?
    
    """
    
    l=[[],[]]
    labels=['beta', 'sw']
    for i, p in enumerate(_get('dictionary')):
    
        #Control
        d={} 
        d=misc.dict_update(p['mul'], d) 
        l[i]+=[pl(d, '*', **{'name':''})]
        
        d={}
        d=misc.dict_update(p['equal'], d) 

        s='GA_MS_intact_{0}'.format( labels[i] )
        
        l[i][-1]+=pl(d, '=', **{'name':s})     
             
        # GA-MS lesion
        d={} 
        d=misc.dict_update(p['mul'], d) 
        l[i]+=[pl(d, '*', **{'name':''})]
        
        d={'conn':{
                   'GA_M1_gaba':{'lesion':True},
                   'GA_M2_gaba':{'lesion':True}
                   }
           }
        d=misc.dict_update(p['equal'], d) 

        s='GA_MS_lesion_{0}'.format( labels[i] )
        
        l[i][-1]+=pl(d, '=', **{'name':s})     


        # STN lesion
        d={} 
        d=misc.dict_update(p['mul'], d) 
        l[i]+=[pl(d, '*', **{'name':''})]
        
        d={'node':{'CS':{'rate':0.}}}
        d=misc.dict_update(p['equal'], d) 

        s='STN_lesion_{0}'.format( labels[i] )
        
        l[i][-1]+=pl(d, '=', **{'name':s})

    
    beta, sw=l
    return {'beta':l[0], 'sw':l[1]}
    

ld=get()
pp(ld)