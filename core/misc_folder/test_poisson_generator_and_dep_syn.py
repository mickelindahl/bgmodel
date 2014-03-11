'''
Created on Oct 2, 2013

@author: lindahlm
'''
import nest
import pylab
t=1000.0
n=nest.Create('iaf_neuron')
mm=nest.Create('multimeter', params={'record_from':['V_m'], 'start':0.0})
pg=nest.Create('poisson_generator', params={'rate':10.0})
nest.Connect(pg,n,model='tsodyks_synapse')
#nest.Connect(mm,n)
nest.Connect(pg,n)
nest.Simulate(t)
smm=nest.GetStatus(mm)[0]
pylab.plot(smm['events']['V_m'])
pylab.show()


