import nest
import pprint
pp=pprint.pprint

import socket
from os.path import expanduser

HOST=socket.gethostname().split('.')
if len(HOST)==1 and HOST[0]!='supermicro':
    HOME='/cfs/milner/scratch/l/lindahlm'
else: 
    HOME = expanduser("~")
 
print HOME

if nest.version()=='NEST 2.2.2':
    s='nest-2.2.2'
if nest.version()=='NEST 2.4.1':
    s='nest-2.4.1'    
if nest.version()=='NEST 2.4.2':
    s='nest-2.4.2'   
      
MODULE_PATH= (HOME+'/opt/NEST/module/'
              +'install-module-130701-'+s+'/lib/nest/ml_module')
MODULE_SLI_PATH= (HOME+'/opt/NEST/module/'
                  +'install-module-130701-'+s+'/share/ml_module/sli')

models=nest.get_models()
if not 'my_aeif_cond_exp' in nest.get_models(): 
    nest.sr('('+MODULE_SLI_PATH+') addpath')
    nest.Install(MODULE_PATH)

for model in nest.get_models():
    if model in models :
        pass
    else:
        print model

import NeuroTools

