'''
Created on Aug 12, 2013

@author: lindahlm
'''


from toolbox.network.default_params import Perturbation_list as pl
import numpy
import pprint
from core.toolbox import misc
pp=pprint.pprint

from oscillation_perturbations8 import get_solution_slow_GP_striatum, update


def get():
    
    
    l=[]
    solution, s_mul, s_equal=get_solution_slow_GP_striatum()
    
    d0=0.8
    f_beta_rm=lambda f: (1-f)/(d0+f*(1-d0))
    
   
    d={}
    for x in numpy.arange(3.4, 6, 0.4):
#         x=3.4
        y=2.25
            
        for keys in s_mul: update(solution, d, keys)  
        l+=[pl(d, '*', **{'name':''})]
          
        d={}
        for keys in s_equal: update(solution, d, keys) 
        
        misc.dict_update(d,{'nest':{'ST':{'beta_I_AMPA_1': f_beta_rm(x)}}})
        misc.dict_update(d,{'nest':{'ST':{'beta_I_NMDA_1': f_beta_rm(x)}}})
        misc.dict_update(d,{'nest':{'ST':{'beta_I_GABAA_1': f_beta_rm(y)}}})
        
        d['node']['EA']['rate']*=0.7    
        
        l[-1]+=pl(d, '=', **{'name':'mod_CS_STbeta_'+str(x)+'_GA_STbeta_'+str(y)})    

    
    return l

get()