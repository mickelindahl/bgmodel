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
                       
        EAr0=p['equal']['node']['EA']['rate']
        EIr0=p['equal']['node']['EI']['rate']
                       
                       
        for EAr, EIr, in [
                     [50, -50.],
                     [100, -50.],
                     [150, -50],
                     
                     [50, -100.],
                     [100, -100.],
                     [150, -100],
                          
                     [50, -150.],
                     [100, -150.],
                     [150, -150],
                     ]:

                d={}
                
                d=misc.dict_update(p['mul'], d) 
                 
#                 pp(d)
                 
                l[i]+=[pl(d, '*', **{'name':''})]
                          
                d={ 'node':{
                            'EA':{'rate':EAr0+EAr},
                            'EI':{'rate':EIr0+EIr},
                            'EF':{'rate':EIr0+EIr}
                           }}
                 
                d=misc.dict_update(p['equal'], d) 
                s='EAr_{0}_EIr_{1}_{2}'
                s=s.format( EAr, EIr, labels[i] )
                 
                l[ i ][ -1 ]+=pl(d, '=', **{'name':s})     

     
    beta, sw=l
    return {'beta':l[0], 'sw':l[1]}
     
 
ld=get()
pp(ld)
 
       
