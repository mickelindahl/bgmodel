'''
Created on Aug 12, 2013

@author: lindahlm
'''


from core.network.default_params import Perturbation_list as pl
import numpy
import pprint
from core import misc
pp=pprint.pprint



from oscillation_perturbations8 import get_solution, update

def get():
        
    l=[]
    solution, s_mul, s_equal=get_solution()
    
    # Decrease/increase C1,C2,CF,CS, EI and EA. Test idea that at
    # slow wave cortical and thalamic input is decreased. 
    mod=numpy.arange(0.5, 1.5, 0.05)
    for x in mod:

        
        d={}
        for keys in s_mul: update(solution, d, keys) 
        misc.dict_update(d, {'node':{'CF':{'rate':x}}} )
        misc.dict_update(d, {'node':{'CS':{'rate':x}}} )

        l+=[pl(d, '*', **{'name':''})]
          
        d={}
        for keys in s_equal: update(solution, d, keys)

        d['node']['C1']['rate']*=x
        d['node']['C2']['rate']*=x        
        d['node']['EI']['rate']*=x
        d['node']['EA']['rate']*=x
        l[-1]+=pl(d, '=', **{'name':'mod_C1_C2_CF_CS_EI_EA_'+str(x)}) 
              

    return l

get()