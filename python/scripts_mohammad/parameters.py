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

import nest
# s='/home/mikael/opt/NEST/module/install-module-130701-nest-2.2.2/share/ml_module/sli'
# s='/home/mikael/opt/NEST/module/install-module-150605-2.6.0-nest-2.6.0/share/ml_module/sli'
# # s='/home/mikael/opt/NEST/module/install-module-150605-2.6.0-nest-2.6.0/share/ml_module/sli'
# p='/home/mikael/opt/NEST/module/install-module-150605-2.6.0-nest-2.6.0/lib/nest/ml_module'
# # p='/home/mikael/opt/NEST/module/install-module-150605-2.6.0-nest-2.6.0/lib/nest/ml_module'
# nest.sr('('+s+') addpath')
# nest.Install(p)

# import nest
# sli_path='/home/mikael/opt/NEST/module/install-module-150605-2.6.0-nest-2.6.0/share/ml_module/sli'
# nest.sr('('+sli_path+') addpath')
# path='/home/mikael/opt/NEST/module/install-module-150605-2.6.0-nest-2.6.0/lib/nest/ml_module'
# nest.Install(path)
# nest.sr('('+sli_path+') addpath')
# nest.Install(path)