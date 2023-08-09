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
from core.my_population import MySpikeList, MyNetworkNode
from core.network import default_params
from core.network.data_processing import Data_unit_vm    
# s='nest-2.2.2'
# HOME = expanduser("~")
# MODULE_PATH= (HOME+'/opt/NEST/module/'
#               +'install-module-130701-'+s+'/lib/nest/ml_module')
# MODULE_SLI_PATH= (HOME+'/opt/NEST/module/'
#                   +'install-module-130701-'+s+'/share/ml_module/sli')
# 
# nest.sr('('+MODULE_SLI_PATH+') addpath')
# nest.Install(MODULE_PATH)

pp(nest.node_models+nest.synapse_models)

d0=0.8
f_beta_rm=lambda f: (1-f)/(d0+f*(1-d0))

pgd0=nest.Create('noise_generator',  params={'mean':0., 'std':1.0})

net=default_params.Inhibition()  
df=net.rec['izh']
pp(df)
d_ST=net.get_nest()['GA']
d_ST['I_e']=12.0


# d_ST['b'] *=1.5
# d_ST['C_m'] *=1.5
# d_ST['Delta_T'] *=1.5

node=net.get_dic()['node']['GA']

for d in [d_ST]:
    del d['type_id']
pn=[]
# for i in range (1):    
#     pn.append(nest.Create('my_aeif_cond_exp',1, params=d_ST))


mm={'active':True,
    'params':{'to_memory':True,
              'to_file':False}}
st=MyNetworkNode('st',model='my_aeif_cond_exp', n=1, params=d_ST, mm=mm)

sd=nest.Create('spike_detector',1)
# sd=nest.Create('spike_de',1)


nest.DivergentConnect(st.ids, [sd[0]])
nest.Connect([pgd0[0]], st.ids, **{'params':{'receptor_type':6}})

sim_time=5000.
nest.Simulate(sim_time)

t=nest.GetStatus(sd)[0]['events']['times']
s=nest.GetStatus(sd)[0]['events']['senders']

signal  = zip( s, t )            
signal = MySpikeList( signal, st.ids, 100, sim_time)
fr=signal.get_firing_rate()

ah=signal.get_spike_stats()

print ah.rates['mean'],ah.rates['std']
ax=pylab.subplot(211)
fr.plot(ax, win=5)


dvm=Data_unit_vm('st',st.get_voltage_signal())
dvm.plot(pylab.subplot(212))

# t=nest.GetStatus(sd)[1]['events']['times']
# s=nest.GetStatus(sd)[1]['events']['senders']
# 
# signal  = zip( s, t )            
# signal = MySpikeList( signal, pn[1], 100, sim_time, )
# fr=signal.get_firing_rate()
# ah=signal.get_spike_stats()
# print ah.rates['mean'], ah.rates['std']
# 
# ax=pylab.subplot(111)
# fr.plot(ax,win=5)

# ax.plot()
pylab.show()