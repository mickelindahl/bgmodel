'''
Created on Oct 2, 2014

@author: mikael
'''

'''
Created on Sep 11, 2014

@author: mikael
'''

import numpy
import nest
import pprint
pp=pprint.pprint


from os.path import expanduser

pp(nest.GetKernelStatus())
s='nest-2.2.2'
HOME = expanduser("~")
MODULE_PATH= (HOME+'/opt/NEST/module/'
              +'install-module-130701-'+s+'/lib/nest/ml_module')
MODULE_SLI_PATH= (HOME+'/opt/NEST/module/'
                  +'install-module-130701-'+s+'/share/ml_module/sli')
print MODULE_PATH
nest.sr('('+MODULE_SLI_PATH+') addpath')
nest.Install(MODULE_PATH)

import time

def set_random_params(ids,vals, keys):
    for val, p in zip(vals, keys):
        local_nodes=[]
        for _id in ids:
            ni=nest.GetStatus([_id])[0]
            if ni['local']:
                local_nodes.append((ni['global_id'], ni['vp']))
                
        for gid, vp in local_nodes:
            val_rand=1+0.1*(numpy.random.random()-0.5)
            val_rand*=val
            nest.SetStatus([gid],{p:val_rand})     

def get_pre_post(n, sources, targets):

    pre=sources*n
    
    post=[]
    for _ in sources:
        post+=list(numpy.random.permutation(targets)[0:n])
    
    return pre, post

def gen_network():
    d={}
    d[0]=nest.Create('izhik_cond_exp', 20000)
    set_random_params(d[0],[-70., 50., 100.], ['V_m', 'V_th', 'C_m'])
    
    d[1]=nest.Create('my_aeif_cond_exp',1000)
    
    df=nest.GetDefaults('my_aeif_cond_exp')['receptor_types']
    
#     syn_spec={'model':{'receptor_type':df['AMPA_1']}}

    params={'receptor_type':df['AMPA_1']}
    nest.CopyModel('static_synapse', 'static', params)
    nest.CopyModel('tsodyks_synapse', 'tsodyks', params)
    
#     nest.RandomConvergentConnect(d[0], d[0], 500, model='static')
#     nest.RandomConvergentConnect(d[0], d[1], 500, model='tsodyks')

    print 'Connecting'    
    args=get_pre_post(500,d[0], d[0] )
    nest.Connect(*args, model='static')
    
    args=get_pre_post(500,d[0], d[1] )
    nest.Connect(*args, model='tsodyks')
    nest.Simulate(10000)
    time.sleep(30)

if __name__=='__main__':
    
    gen_network()
    
    