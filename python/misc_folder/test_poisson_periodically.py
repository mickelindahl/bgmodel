'''
Created on Oct 3, 2014

@author: mikael
'''
import numpy
import nest
import pylab
import pprint
pp=pprint.pprint


from os.path import expanduser
from core.my_population import MySpikeList


s='nest-2.2.2'
HOME = expanduser("~")
MODULE_PATH= (HOME+'/opt/NEST/module/'
              +'install-module-130701-'+s+'/lib/nest/ml_module')
MODULE_SLI_PATH= (HOME+'/opt/NEST/module/'
                  +'install-module-130701-'+s+'/share/ml_module/sli')

nest.sr('('+MODULE_SLI_PATH+') addpath')
nest.Install(MODULE_PATH)

pp(nest.node_models+nest.synapse_models)


pp(nest.GetDefaults('poisson_generator_periodic'))
n=nest.Create('poisson_generator_periodic',  params={'rate_first':0.,
                                               'rate_second':1000.
                                               })
pn=nest.Create('parrot_neuron',2)
sd=nest.Create("spike_recorder",2)

nest.Connect(n+n,pn)

nest.Connect([pn[0]], [sd[0]])
nest.Connect([pn[1]], [sd[1]])

sim_time=5000.0
nest.Simulate(sim_time)


t=nest.GetStatus(sd)[0]['events']['times']
s=nest.GetStatus(sd)[0]['events']['senders']

signal  = zip( s, t )            
signal = MySpikeList( signal, [2], 1, sim_time)
fr=signal.get_firing_rate()

ax=pylab.subplot(111)
fr.plot(ax)


t=nest.GetStatus(sd)[1]['events']['times']
s=nest.GetStatus(sd)[1]['events']['senders']

signal  = zip( s, t )            
signal = MySpikeList( signal, [3], 1, sim_time)
fr=signal.get_firing_rate()

ax=pylab.subplot(111)
fr.plot(ax)

# ax.plot()
pylab.show()
