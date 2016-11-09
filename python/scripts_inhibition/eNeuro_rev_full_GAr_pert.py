'''
Created on Aug 12, 2013
 
@author: lindahlm
'''
 
from core.network.default_params import Perturbation_list as pl
from core import misc
from scripts_inhibition.base_perturbations import get_solution
from scripts_inhibition.fig_01_and_02_pert import get as _get
 
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
           
        for f_GA_r in [3.,2.5,2.,1.5,1]:    
            d={'nest':{
                       
                       # IF curve GA
                       'GA':{
                              'b':1.,
                              'C_m':1.,
                              'Delta_T':1.
                              }
                        }}
               
            d=misc.dict_update(p['mul'], d) 
             
            l[i]+=[pl(d, '*', **{'name':''})]
                           
            d={
               
               # Tuning rate GI/GF and GA
               'node':{
                        'EA':{'rate':f_GA_r*EAr}
                        }}
             
            d=misc.dict_update(p['equal'], d) 
            s='GAr_{0}-{1}'
            s=s.format( f_GA_r, labels[i] )
             
            l[i][-1]+=pl(d, '=', **{'name':s})     

     
    beta, sw=l
    return {'beta':l[0], 'sw':l[1]}
     
 
ld=get()
pp(ld)
 
       
