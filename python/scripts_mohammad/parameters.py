'''
Created on Aug 26, 2015

@author: mikael
'''
import fig_01_and_02_pert as op

from core.network import default_params
import pprint
pp=pprint.pprint


par=default_params.Inhibition(**{'perturbations':op.get()[0]})
pp(par.dic['nest'].keys())

#Look at MSN D1 (high or low gabaa reversal potential E_rev)
print('MSN D1')
pp(par.dic['nest']['M1_low']) #I am using low
# pp(par.dic['nest']['M1_high'])

print('MSN D2')
pp(par.dic['nest']['M2_low']) #I am using low


print('FSN')
pp(par.dic['nest']['FS'])
