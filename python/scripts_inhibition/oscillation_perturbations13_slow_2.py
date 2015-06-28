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
    
    d={}
    for keys in s_mul: update(solution, d, keys)  
    l+=[pl(d, '*', **{'name':''})]
      
    d={}
    for keys in s_equal: update(solution, d, keys) 
    
    l[-1]+=pl(d, '=', **{'name':''})    
            
    
    return l

get()