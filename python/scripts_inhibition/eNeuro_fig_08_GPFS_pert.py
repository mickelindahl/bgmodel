'''
Created on Aug 12, 2013

@author: lindahlm
'''

from core.network.default_params import Perturbation_list as pl
from core import misc

import pprint

pp=pprint.pprint

d0=0.8
f_beta_rm=lambda f: (1-f)/(d0+f*(1-d0))

def get():
    
    l=[];
    p={'equal':{},
          'mul':{}}#

    
    for w1, w2 in [
                    [1., .0],      
                    [.8, .2], 
                    [.6, .4],    
                    [.5, .5],    
                    [.4, .6],       
                    [.2, .8],      
                    [.0, 1.],
                   ]:
                     
        d={}
        
        d=misc.dict_update(p['mul'], d) 
        
        l+=[pl(d, '*', **{'name':''})]
        
        d={'nest':{
                   'GA_FS_gaba':{'weight':w1},
                   'GF_FS_gaba':{'weight':w2}
                  }}

        d=misc.dict_update(p['equal'], d) 

        s='GA_{0}_GF_{1}'.format( w1, w2 )
        
        l[-1]+=pl(d, '=', **{'name':s})     
    
    return l
    

get()


