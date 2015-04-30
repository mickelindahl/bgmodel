'''
Created on Aug 12, 2013

@author: lindahlm
'''


from toolbox.network.default_params import Perturbation_list as pl
import numpy
import pprint
from core.toolbox import misc
pp=pprint.pprint

from oscillation_perturbations8 import get_solution_slow_GP_striatum_2, update


def get():
    
    
    l=[]
    solution=get_solution_slow_GP_striatum_2()
    d0=0.8
    f_beta_rm=lambda f: (1-f)/(d0+f*(1-d0))
    
    for y in numpy.arange(1., 3., 1.):       
        d={}
        misc.dict_update(d, solution['mul'])
        l+=[pl(d, '*', **{'name':''})]
          
        d={}
        misc.dict_update(d, solution['equal']) 
        
        misc.dict_update(d,{'nest':{'GI':{'beta_I_GABAA_1': f_beta_rm(y)}}})
    
        
        l[-1]+=pl(d, '=', **{'name':'mod_GI_beta_'+str(y)})  
    
    return l

get()