'''
Created on Aug 12, 2013

@author: lindahlm
'''


from core.network.default_params import Perturbation_list as pl
import numpy
import pprint
from core_old.toolbox import misc
pp=pprint.pprint

from oscillation_perturbations8 import get_solution, update
# from oscillation_perturbations15 import STN_ampa_gaba_input_magnitude

def get():
    
    
    l=[]
    solution, s_mul, s_equal=get_solution()
    
    d0=0.8
    f_beta_rm=lambda f: (1-f)/(d0+f*(1-d0))
    
    
#     v=STN_ampa_gaba_input_magnitude()[0:5]
    
    for _, u, w in v:

        v0=numpy.arange(1.25, 3.75, 1.)
        v1=numpy.arange(1.5, 4, 1.)
        v=[[x,y] for x in v0 for y in v1]
        for x, y in v:
            
            d={}
            for keys in s_mul: update(solution, d, keys)  
            l+=[pl(d, '*', **{'name':''})]
              
            d={}
            for keys in s_equal: update(solution, d, keys) 
    
            misc.dict_update(d,{'nest':{'ST':{'beta_I_GABAA_1': f_beta_rm(x)}}})        
            misc.dict_update(d,{'nest':{'ST':{'beta_I_AMPA_1': f_beta_rm(y)}}})
            misc.dict_update(d,{'nest':{'ST':{'beta_I_NMDA_1': f_beta_rm(y)}}})
            misc.dict_update(d,{'node':{'CS':{'rate': u}}})
            misc.dict_update(d,{'nest':{'GI_ST_gaba':{'weight':w }}})             
            d['node']['EA']['rate']*=0.7
            l[-1]+=pl(d, '=', **{'name':('mod_ST_beta_'
                                         +str(u)+'_'+str(w)+'_'
                                         +str(x)+'_'+str(y))
                                 })    

               
    
    return l

get()