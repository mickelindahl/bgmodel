'''
Created on Aug 12, 2013

@author: lindahlm
'''


from toolbox.network.default_params import Perturbation_list as pl
import numpy
import pprint
from core.toolbox import misc
import pylab
pp=pprint.pprint

from oscillation_perturbations8 import get_solution_slow_GP_striatum, update


def STN_ampa_gaba_input_magnitude():
    l=[
       [20, 188, 0.08], # 20 Hz 
       [25, 290, 0.119], #25 Hz
       [30, 430, 0.18],      #30 Hz
       [35, 540, 0.215],     #35 Hz,
       [40, 702, 0.28],     #40 Hz
       [45, 830., 0.336],   #45 Hz 
       [46, 876.7, 0.349],  #46 Hz
       [50, 1000.8, 0.3957],     # 50 Hz
       [55, 1159., 0.458],  #50 Hz
       [80, 2102, 0.794] # 80 Hz] 
       ]
    
    return l


def get():
    
    
    l=[]
    solution, s_mul, s_equal=get_solution_slow_GP_striatum()
    
    d0=0.8
    f_beta_rm=lambda f: (1-f)/(d0+f*(1-d0))
    
    v=STN_ampa_gaba_input_magnitude()
    
    for _, x, y in v:
#     x=2.5-1
        d={}
        for keys in s_mul: update(solution, d, keys)  
        l+=[pl(d, '*', **{'name':''})]
          
        d={}
        for keys in s_equal: update(solution, d, keys) 
        
    #     misc.dict_update(d,{'nest':{'ST':{'beta_I_AMPA_1': f_beta_rm(x)}}})
    #     misc.dict_update(d,{'nest':{'ST':{'beta_I_NMDA_1': f_beta_rm(x)}}})
    #     d['node']['EA']['rate']*=0.7
        
        misc.dict_update(d,{'node':{'CS':{'rate': x}}})
        misc.dict_update(d,{'nest':{'GI_ST_gaba':{'weight':y }}})
        
        l[-1]+=pl(d, '=', **{'name':'mod_ST_inp_'+str(x)+'_'+str(y)})    
    
    return l

# x,y0,y1=zip(*STN_ampa_gaba_input_magnitude())
# x=numpy.array(x)
# y0=numpy.array(y0)
# k0=numpy.mean(numpy.diff(y0)/numpy.diff(x))
# 
# y0/=y0[0]
# y1=numpy.array(y1)
# k1=numpy.mean(numpy.diff(y1)/numpy.diff(x))
# 
# y1/=y1[0]
# 
# print k0
# print k1
# 
# pylab.plot(x,y0)
# pylab.plot(x,y1)
# pylab.show()

get()