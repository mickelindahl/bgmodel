'''
Created on Aug 12, 2013

@author: lindahlm
'''


from toolbox.network.default_params import Perturbation_list as pl
import numpy
import pprint
from core.toolbox import misc
pp=pprint.pprint

from oscillation_perturbations8 import get_solution_slow_GP_striatum_2


def get():
    
    
    l=[]
    solution=get_solution_slow_GP_striatum_2()
    misc.dict_update(d, solution['mul']) 
    l+=[pl(d, '*', **{'name':''})]
    misc.dict_update(d, solution['equal']) 
    l[-1]+=pl(d, '=', **{'name':'myname'}) 
    
    
    return l


get()