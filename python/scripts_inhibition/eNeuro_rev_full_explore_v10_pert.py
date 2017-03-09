'''
Created on Aug 12, 2013
 
@author: lindahlm
'''
 
from core.network.default_params import Perturbation_list as pl
from core import misc
from scripts_inhibition.eNeuro_fig_01_and_02_pert import get as _get
 
import numpy
import pprint
 
pp=pprint.pprint
 
d0=0.8
f_beta_rm=lambda f: (1-f)/(d0+f*(1-d0))
 
 
def get():
    l=[[],[]]
    
    labels=['beta', 'sw']
    
    for i, p in enumerate(_get('dictionary')):
                       
        for GPSTdelay, STGPdelay in [
                                     [5., 5.],
                                     [1., 5.],
                                     [1., 3.],
                                     [1., 2.]
                                     ]:

                d={}
                
                d=misc.dict_update(p['mul'], d) 
                 
                l[i]+=[pl(d, '*', **{'name':''})]
                          
                d={'nest':{'GI_ST_gaba':{'delay':GPSTdelay},
                           'GF_ST_gaba':{'delay':GPSTdelay},
                           
                           'ST_GI_ampa':{'delay':STGPdelay},
                           'ST_GF_ampa':{'delay':STGPdelay},
                           'ST_GA_ampa':{'delay':STGPdelay} }
                   }
                 
                d=misc.dict_update(p['equal'], d) 
                s='GPSTdelay_{0}_STGPdelay_{1}_{2}'
                s=s.format( GPSTdelay, STGPdelay, labels[i] )
                 
                l[ i ][ -1 ]+=pl(d, '=', **{'name':s})     

     
    beta, sw=l
    return {'beta':l[0], 'sw':l[1]}
     
 
ld=get()
pp(ld)
 
       
