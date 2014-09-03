import nest
import pprint
pp=pprint.pprint

from os.path import expanduser
HOME = expanduser("~")

#MODULE_PATH= ('ml_module')
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


pp(nest.Models())
if not 'my_aeif_cond_exp' in nest.Models(): 
    nest.sr('('+MODULE_SLI_PATH+') addpath')
    nest.Install(MODULE_PATH)
pp(nest.Models())
