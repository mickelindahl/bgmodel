import numpy
import nest
import pprint
pp=pprint.pprint

import socket
from os.path import expanduser

import os
print os.environ['LD_LIBRARY_PATH']


#os.environ['LD_LIBRARY_PATH']='/afs/nada.kth.se/home/w/u1yxbcfw/opt/NEST/module/install-module-130701-nest-2.2.2/lib/nest'


print os.environ['LD_LIBRARY_PATH'], 'empty'
print socket.gethostname()

HOST=socket.gethostname().split('.')
if len(HOST)==1:
    os.environ['LD_LIBRARY_PATH']='/cfs/milner/scratch/l/lindahlm/opt/NEST/module/install-module-130701-nest-2.2.2-milner/lib/nest'

    HOME='/cfs/milner/scratch/l/lindahlm'
    ADD='-milner'
else:
    os.environ['LD_LIBRARY_PATH']='/afs/nada.kth.se/home/w/u1yxbcfw/opt/NEST/module/install-module-130701-nest-2.2.2/lib/nest'

    HOME = expanduser("~")
    ADD=''

print HOME

#MODULE_PATH= ('ml_module')
if nest.version()=='NEST 2.2.2':
    s='nest-2.2.2'
if nest.version()=='NEST 2.4.1':
    s='nest-2.4.1'    
if nest.version()=='NEST 2.4.2':
    s='nest-2.4.2'   
      
MODULE_PATH= (HOME+'/opt/NEST/module/'
              +'install-module-130701-'+s+ADD+'/lib/nest/ml_module')
MODULE_SLI_PATH= (HOME+'/opt/NEST/module/'
                  +'install-module-130701-'+s+ADD+'/share/ml_module/sli')

if (not 'bcpnn_dopamine_synapse' in nest.Models()):
    print 'hej'
#     nest.sr('(/cfs/milner/scratch/b/berthet/modules/bcpnndopa_module/share/ml_module/sli) addpath') #t/tully/sequences/share/nest/sli
    nest.sr('(/cfs/milner/scratch/l/lindahlm//opt/NEST/module/install-module-130701-nest-2.2.2-milner/share/ml_module/sli) addpath') #t/tully/sequences/share/nest/sli
    
#     nest.Install('/cfs/milner/scratch/b/berthet/modules/bcpnndopa_module/lib/nest/ml_module')
    nest.Install('/cfs/milner/scratch/l/lindahlm//opt/NEST/module/install-module-130701-nest-2.2.2-milner/lib/nest/ml_module')


models=nest.Models()
if not 'my_aeif_cond_exp' in nest.Models(): 
    nest.sr('('+MODULE_SLI_PATH+') addpath')
    print os.environ['LD_LIBRARY_PATH'], 'again'
    print MODULE_PATH
    print MODULE_SLI_PATH
    nest.Install(MODULE_PATH)
    #nest.Install('ml_module')

for model in models:
    if model in nest.Models():
        pass
    else:
        print model

import NeuroTools

