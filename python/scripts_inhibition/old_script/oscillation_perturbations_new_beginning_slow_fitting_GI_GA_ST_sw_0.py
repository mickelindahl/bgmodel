'''
Created on Aug 12, 2013

@author: lindahlm
'''

from oscillation_perturbations_new_beginning_slow0 import get_solution
from core.network.default_params import Perturbation_list as pl
import numpy
import pprint
from core_old.toolbox import misc
pp=pprint.pprint

d0=0.8
f_beta_rm=lambda f: (1-f)/(d0+f*(1-d0))

def STN_ampa_gaba_input_magnitude():
    l=[
       [20, 188, 0.08], # 20 Hz 
#        [20, 210, 0.08], # 20 Hz 
#        [25, 290, 0.119], #25 Hz
#        [30, 430, 0.18],      #30 Hz
#        [35, 540, 0.215],     #35 Hz,
#        [40, 702, 0.28],     #40 Hz
#        [45, 830., 0.336],   #45 Hz 
#        [46, 876.7, 0.349],  #46 Hz
#        [50, 1000.8, 0.3957],     # 50 Hz
#        [55, 1159., 0.458],  #50 Hz
#        [80, 2102, 0.794] # 80 Hz] 
       ]
    
    return l

def get():
    
    
    l=[]
    solution=get_solution()
    
    
    '''
    From
    script_0000_rEI_1600.0_rEA_0.0_rCS_250.0_wGA_1.0-amp_0.08_1.0_stn_3.0_0
    in new_beginning beta slow (1.77)
    '''
#     misc.dict_update(solution, {'equal':{'node':{'M2p':{'rate':2.9}}}})
    '''
    From estimate single GA, ST, subtract  since single not estimate overestimated
    exitation needed. IN network inhibition onto ST from GP is weaker at the same
    frequency. And ST_GA_ampa need to be stronger in full network than in single 
    mode to optain similar firing rates.
    '''
    misc.dict_update(solution, {'equal':{'node':{'CS':{'rate':190-10}}}})
    misc.dict_update(solution, {'equal':{'nest':{'ST_GA_ampa':{'weight':0.236*1.1}}}})
    misc.dict_update(solution, {'equal':{'node':{'EA':{'rate':210.}}}})
    misc.dict_update(solution, {'equal':{'node':{'EI':{'rate':1200.0}}}})

    d={}
    misc.dict_update(d, solution['mul'])
    l+=[pl(d, '*', **{'name':''})]
      
    d={}
    misc.dict_update(d, solution['equal']) 
    l[-1]+=pl(d, '=', **{'name':'control_sim'}) 
    
    return l

l=get()
for i, p in enumerate(l):
    print i, p