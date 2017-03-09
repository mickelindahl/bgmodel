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
                       
        EAr=p['equal']['node']['EA']['rate']
                       
                       
        for STGPf, EAf in [
                     [.25, 1.],
                     [.35, 0.9],
                     [.45, 0.8],
                     [.55, 0.7],
                     
                     [.25, 1.],
                     [.35, 0.85],
                     [.45, 0.7],
                     [.55, 0.55]
                     ]:

                d={'nest':{'ST_GA_ampa':{'weight':STGPf}}}
                
                d=misc.dict_update(p['mul'], d) 
                 
#                 pp(d)
                 
                l[i]+=[pl(d, '*', **{'name':''})]
                          
                d={ 'node':{
                            'EA':{'rate':EAr*EAf}
                           }}
                 
                d=misc.dict_update(p['equal'], d) 
                s='STGPf_{0}_EAf_{1}_{2}'
                s=s.format( STGPf, EAf, labels[i] )
                 
                l[ i ][ -1 ]+=pl(d, '=', **{'name':s})     

     
    beta, sw=l
    return {'beta':l[0], 'sw':l[1]}
     
 
ld=get()
pp(ld)
 
       
