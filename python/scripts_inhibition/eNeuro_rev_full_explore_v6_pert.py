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
        EIr=p['equal']['node']['EI']['rate'] 
                       
        for CFf, EIf in [[1., 1.],
                         [.9, 1.],
                         [.8, 1.],
                         [1., 1.2],
                         [.9, 1.2],
                         [.8, 1.2],
                         [1., 1.4],
                         [.9, 1.4],
                         [.8, 1.4]]:    
            
            d={
                'node':{
                        'CF':{'rate':CFf},
                            }}
            
            d=misc.dict_update(p['mul'], d) 
             
            l[i]+=[pl(d, '*', **{'name':''})]
                      
            d={
                'node':{
                        'EI':{'rate':EIr*EIf},
                        'EF':{'rate':EIr*EIf},
                            }}
             
            d=misc.dict_update(p['equal'], d) 
            s='CFf_{0}_EIf_{1}_{2}'
            s=s.format( CFf, EIf, labels[i] )
             
            l[i][-1]+=pl(d, '=', **{'name':s})     

     
    beta, sw=l
    return {'beta':l[0], 'sw':l[1]}
     
 
ld=get()
pp(ld)
 
       
