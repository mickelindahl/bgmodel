'''
Created on Aug 12, 2013

@author: lindahlm
'''


from core.network.default_params import Perturbation_list as pl
import numpy
import pprint
from core_old.toolbox import misc
pp=pprint.pprint

from oscillation_perturbations8 import get_solution_slow_GP_striatum_2, update


def get():
    
    
    l=[]
    solution, s_mul, s_equal=get_solution_slow_GP_striatum_2()
  
    for y in numpy.arange(1.0,-0.1,-0.25):
        for z in numpy.arange(1.0,-0.1,-0.25):
            x=2.5
            d={}
            for keys in s_mul: update(solution, d, keys)  
    
            misc.dict_update(d,{'nest':{'GI_GA_gaba':{'weight':y}}})
            misc.dict_update(d,{'nest':{'ST_GA_ampa':{'weight':z}}})
            
            l+=[pl(d, '*', **{'name':''})]
        
    
              
            d={}
            for keys in s_equal: update(solution, d, keys) 
        
            l[-1]+=pl(d, '=', **{'name':'GIGA_'+str(y)+'_STGA_'+str(z)})  
    
         
    return l

get()