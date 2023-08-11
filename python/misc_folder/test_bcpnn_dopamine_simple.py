import numpy
import pylab

from os.path import expanduser
from core import my_nest
import mpi4py

HOME = expanduser("~")
MODULE_PATH= (HOME+'/opt/NEST/dist/install-nest-2.2.2/lib/nest/ml_module')
my_nest.Install(MODULE_PATH)


#ip=my_nest.Create('spike_generator',params={'spike_times':[10.,20.]})
ip=my_nest.Create('poisson_generator',3, params={'rate':3500.})
pn=my_nest.Create('parrot_neuron',3)
sd=my_nest.Create("spike_recorder")
n=my_nest.Create('iaf_cond_exp',3, params={ 'C_m':200.0, 'V_th':-50.})
vt=my_nest.Create('volume_transmitter')
my_nest.Connect(ip,pn)
my_nest.Connect([pn[2]], [n[2]], params={'weight':10.})
my_nest.Connect([n[2]],sd)

my_nest.Connect([n[2]],vt)


my_nest.Connect(pn[0:2], n[0:2])
my_nest.SetDefaults('bcpnn_dopamine_synapse', params={'vt':vt[0]})
my_nest.Connect([n[0]],[n[1]], model='bcpnn_dopamine_synapse')

p_sd={"withgid": True, 'to_file':False,  'to_memory':True }
sd=my_nest.Create("spike_recorder",5, params=p_sd)
pd=my_nest.Create('poisson_generator',5, params={'rate':100.})
my_nest.Connect(pd, sd)

d={'n':[]}
T=[]
for t in range(200):
    my_nest.Simulate(1)
    d['n'].append(my_nest.GetConnProp([n[0]],[n[1]], 'n'))
    _t=my_nest.GetStatus(sd)[0]['events']['times']
    if list(_t[(_t>=t-1)*(_t<t)]):
        T.append(_t[(_t>=t-1)*(_t<t)])


T=reduce(lambda x,y:list(x)+list(y),T)
print _t-T
print my_nest.PrintNetwork()
t=[[tt,tt,tt] for tt in t]
y=[[0,0.01,0] for tt in t]

t=reduce(lambda x,y:x+y,t)
y=reduce(lambda x,y:x+y,y)

binned=numpy.zeros(len(d['n']))
      
pylab.plot(t, y)    
pylab.plot(d['n'])
pylab.show()
    


