'''
Created on Sep 11, 2014

@author: mikael
'''

import nest
import numpy
import pylab

from os.path import expanduser
import mpi4py

import pprint
pp=pprint.pprint

if nest.version()=='NEST 2.2.2':
    s='nest-2.2.2'
if nest.version()=='NEST 2.4.1':
    s='nest-2.4.1'    
if nest.version()=='NEST 2.4.2':
    s='nest-2.4.2'   

pp(nest.node_models+nest.synapse_models)

HOME = expanduser("~")
MODULE_PATH= (HOME+'/opt/NEST/module/'
              +'install-module-130701-'+s+'/lib/nest/ml_module')
MODULE_SLI_PATH= (HOME+'/opt/NEST/module/'
                  +'install-module-130701-'+s+'/share/ml_module/sli')

nest.sr('('+MODULE_SLI_PATH+') addpath')
nest.Install(MODULE_PATH)



def gen_spikes(n, duration, gap):
    
    
    spikes=[]
    for i in range(n):
        spikes+=range(1+i*(gap+duration),
                      1+duration+i*(gap+duration),10)    
    return numpy.array(spikes, dtype=numpy.float)
     
    

if __name__=='__main__':
    
    #ip=nest.Create('spike_generator',params={'spike_times':[10.,20.]})
    sg=nest.Create('spike_generator',1)
    st=gen_spikes(3, 100, 900)
    pp(nest.SetStatus(sg, {'spike_times':st}))
    # pp(nest.GetStatus(sg))
    n=nest.Create('my_aeif_cond_exp',1)
    
    df=nest.GetDefaults('my_aeif_cond_exp')['receptor_types']
    # df=nest.GetDefaults('my_aeif_cond_exp')['recordables']
    receptor='AMPA_1'
    recordables=['g_AMPA_1']
    print nest.version()
    nest.Connect(sg, n,model='tsodyks_synapse', params={
                                  'U':0.0192,
                                'tau_fac': 623., 
                                     'tau_rec':559. ,
                                     'receptor_type':df[receptor]})
    print nest.version()
    #         dic['nest']['M1_SN_gaba']['U']        = 0.0192
    #         dic['nest']['M1_SN_gaba']['tau_fac']  = 623. 
    #         dic['nest']['M1_SN_gaba']['tau_rec']  = 559. 
    #         dic['nest']['M1_SN_gaba']['tau_psc']  = 5.2 
    pp(df)
    p_mm={"withgid": True, 
          'to_file':False,  
          'to_memory':True,
          'record_from':recordables
          }
    mm=nest.Create('multimeter', params=p_mm)
    
    pp(nest.GetStatus(mm))
    nest.Connect(mm,n)
    
    
    nest.Simulate(st[-1])
    
    
    t=nest.GetStatus(mm)[0]['events']['times']
    y=nest.GetStatus(mm)[0]['events']['g_AMPA_1']
    
          
    pylab.plot(t, y)    
    # pylab.plot(d['n'])
    pylab.show()