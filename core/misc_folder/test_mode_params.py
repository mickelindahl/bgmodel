'''
Created on Sep 23, 2013

@author: lindahlm
'''
MODULE_PATH=  '/afs/nada.kth.se/home/w/u1yxbcfw/tools/NEST/dist/install-nest-2.2.2/lib/nest/ml_module'


from default_params import Par
from toolbox import data_handling
import sys

p =Par(MODULE_PATH)
save_at='/'.join(sys.argv[0].split('/')[:-1])+'/default_params2'
data_handling.dic_save(p.dic, save_at)