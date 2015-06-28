'''
Created on Dec 16, 2014

@author: mikael
'''

import oscillation_perturbations4 as op
from core.network.default_params import Inhibition

per=op.get()[7]
par=Inhibition(perturbations=per)

for model in sorted(par['conn'].keys()):
    syn=par['conn'][model]['syn']
    print model, par['nest'][syn]['weight']
#     print model, par['conn'][model].keys()


for model in sorted(par['node'].keys()):

    print model, par['node'][model]['rate']