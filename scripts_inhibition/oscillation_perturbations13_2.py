'''
Created on Aug 12, 2013

@author: lindahlm
'''


from toolbox.network.default_params import Perturbation_list as pl
import pprint
pp=pprint.pprint

from oscillation_perturbations8 import get_solution_2, update


def get():
    
    
    l=[]
    solution, s_mul, s_equal=get_solution_2()
    
    d={}
    for keys in s_mul: update(solution, d, keys)  
    l+=[pl(d, '*', **{'name':''})]
      
    d={}
    for keys in s_equal: update(solution, d, keys) 
    
    l[-1]+=pl(d, '=', **{'name':''})    
            
    
    return l

get()