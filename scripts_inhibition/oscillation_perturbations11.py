'''
Created on Aug 12, 2013

@author: lindahlm
'''


from toolbox.network.default_params import Perturbation_list as pl
import numpy
import pprint
from toolbox import misc
pp=pprint.pprint



from oscillation_perturbations8 import get_solution, update

def get():
        
    l=[]
    solution, s_mul, s_equal=get_solution()
    
    # Decrease/increase C1,C2,CF,CS, EI and EA. Test idea that at
    # slow wave cortical and thalamic input is decreased. 
    mod=numpy.arange(0.7, 1.3, 0.05)
    for x in mod:

        
        d={}
        for keys in s_mul: update(solution, d, keys) 
        l+=[pl(d, '*', **{'name':''})]
          
        d={}
        for keys in s_equal: update(solution, d, keys)

        d['node']['EI']['rate']*=x
        d['node']['EA']['rate']*=x
        l[-1]+=pl(d, '=', **{'name':'mod_EI_EA_'+str(x)}) 
              

    return l

get()