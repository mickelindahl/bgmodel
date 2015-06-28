'''
Created on Aug 12, 2013

@author: lindahlm
'''

from copy import deepcopy
from core.network.default_params import Perturbation_list as pl

import oscillation_perturbations4 as op
import pprint
pp=pprint.pprint


def get():
    
    ll=[op.get()[7]]
    
    for l in ll[:]:
        for beta in [0., -0.9375, -0.3125]:
            _lb=deepcopy(l) 
            per=pl({'nest':{'M1_low':{'beta_I_GABAA_2':beta },
                            'M2_low':{'beta_I_GABAA_2':beta }, 
                            }},
                      '=', 
                    **{'name':'beta-'+str(beta)})
            _lb+=per
            ll.append(_lb)
    
        
    return ll
 