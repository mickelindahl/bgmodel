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
    
    
        # Test STN-STN
        
        for decay, decay2 in [
                             [1.,1.],
			      [1.,5.],
		              [5.,1.],
			      [5.,5.]
		              ]:
            for w1, w2, CFf in [
                                [1., .0, 1.4],      
                                [.8, .2, 1.4], 
                                [.7, .3, 1.4],    
                                [.6, .4, 1.4],    
                                [.5, .5, 1.4],    
                                [.4, .6, 1.4],    
                                [.3, .7, 1.4],    
                                [.2, .8, 1.4],    
				                [.1, .9, 1.4],    
                                [.0, 1., 1.4],

                                [.5, .5, 1.4],    
                                [.7, .7, 1.5],    
                                [.9, .9, 1.6],    
                                [1., 1., 1.7],    

                               ]:
                 
                
                d={  
                   'node':{'CF':{'rate':CFf}}}
                        
                d=misc.dict_update(p['mul'], d) 
                
                l[i]+=[pl(d, '*', **{'name':''})]
                
                d={'conn':{'GF_FS_gaba':{'lesion':False}},
                   'nest':{'FS_low':{
				     'GABAA_2_Tau_decay':12*decay,
                                     'GABAA_3_Tau_decay':12*decay2},
                           'GA_FS_gaba':{'weight':w1/decay},
                           'GF_FS_gaba':{'weight':w2/decay2}
                          }}

                d=misc.dict_update(p['equal'], d) 
        
                s='GA_{0}_GF_{1}_d_{2}_d2_{3}_CFf_{4}_{5}'.format( w1, w2, decay, decay2, CFf,labels[i] )
                
                l[i][-1]+=pl(d, '=', **{'name':s})     
        
    
    
    beta, sw=l
    return {'beta':l[0], 'sw':l[1]}
    

ld=get()
pp(ld)

