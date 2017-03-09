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
        
        for CFf in [1.0, 1.2, 1.4]:
            for f1, f2, decay, decay2 in [
                                [1., .0, 5., 1.],      
                                [.7, .4, 5., 1.], 
                                [.4, .6, 5., 1.],    
                                [.1, .9,5., 1.],  
                                [.01, .99, 5., 1.], 
                                [.001, .999, 5., 1.],
                                [.0, 1., 5., 1.],
                                
                                [1., .0, 5., 1.],      
                                [.7, .0, 5., 1.], 
                                [.4, .0, 5., 1.],    
                                [.1, .0, 5., 1.],  
                                [.01, .0, 5., 1.], 
                                [.001, .0, 5., 1.],
                                [.0, .0, 5., 1.],
                                
                                [.5, .5, 5., 1.],
                                [.5, .5, 4., 1.],
                                [.5, .5, 3., 1.],
                                [.5, .5, 2., 1.],
                                [.5, .5, 1., 1.],
                                
                                [.5, .5, 5., 1.],
                                [.5, .5, 5., 2.],
                                [.5, .5, 5., 3.],
                                [.5, .5, 5., 4.],
                                [.5, .5, 5., 5.],
                                
                               ]:
                 
                
                d={'nest':{'GA_FS_gaba':{'weight':5*f1/decay},
                           'GF_FS_gaba':{'weight':f2/decay2}},  
                   'node':{'CF':{'rate':CFf}}}
                        
                d=misc.dict_update(p['mul'], d) 
                
                l[i]+=[pl(d, '*', **{'name':''})]
                
                d={'conn':{'GF_FS_gaba':{'lesion':False}},
                   'nest':{'FS_low':{'GABAA_2_Tau_decay':12*decay,
                                     'GABAA_3_Tau_decay':17*decay2}}}
                d=misc.dict_update(p['equal'], d) 
        
                s='GA_{0}_GF_{1}_d_{2}_d2_{3}_CFf_{4}_{5}'.format( f1, f2, decay, decay2, CFf,labels[i] )
                
                l[i][-1]+=pl(d, '=', **{'name':s})     
        
    
    
    beta, sw=l
    return {'beta':l[0], 'sw':l[1]}
    

ld=get()
pp(ld)

