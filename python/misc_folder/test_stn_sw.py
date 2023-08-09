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
from core.network import default_params

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

# nest.GetDefaults(dn['type_id'])

nest.CopyModel('poisson_generator_dynamic', 'new')

pp(nest.GetDefaults('poisson_generator_dynamic'))


n=100
params={}
params['p_amplitude0']=1.
params['p_amplitude_upp']=0.2

params['p_amplitude_down']=-0.99#-1.0#-0.2#-1
params['period']='constant'
params['freq']=1
rate=200
start=0
stop=4000
typ='oscillation2'
scale=6
ss=default_params.calc_spike_setup(n, params, rate, start, stop, typ)
# ss=default_params.calc_spike_setup(n, params, rate*scale, start, stop, typ)

pgd0=nest.Create('poisson_generator_dynamic',  params={'rates':ss[0]['rates'], 
                                                      'timings':ss[0]['times']})

ss=default_params.calc_spike_setup(n, params, rate, start, stop, typ)

pgd1=nest.Create('poisson_generator_dynamic',  params={'rates':ss[0]['rates'], 
                                                      'timings':ss[0]['times']})


d0=0.8
f_beta_rm=lambda f: (1-f)/(d0+f*(1-d0))


net=default_params.Inhibition()  
df=net.rec['izh']
pp(df)
d_ST=net.get_nest()['ST']
d_CS_ST_ampa=net.get_nest()['CS_ST_ampa']
d_CS_ST_nmda=net.get_nest()['CS_ST_nmda']

for d in [d_ST, d_CS_ST_ampa, d_CS_ST_nmda]:
    del d['type_id']
pn=[]
for i in range (2):    
    if i==0:
        d_ST['tata_dop']=0.0                  
    if i==1:
        d_ST['beta_I_AMPA_1']= f_beta_rm(scale)
        d_ST['beta_I_NMDA_1']= f_beta_rm(scale)#     pn=nest.Create('parrot_neuron',2)
        d_ST['tata_dop']=-0.8
    pn.append(nest.Create('my_aeif_cond_exp',n, params=d_ST))


# nest.Connect(pgd1*n,pn[1], params=d_CS_ST_ampa)
# nest.Connect(pgd1*n,pn[1], params=d_CS_ST_nmda)
d_CS_ST_ampa['weight']*=scale*0.8
d_CS_ST_nmda['weight']*=scale*0.8
# nest.Connect(pgd0*n,pn[0], params=d_CS_ST_ampa)
# nest.Connect(pgd0*n,pn[0], params=d_CS_ST_nmda)


sd=nest.Create('spike_detector',2)
 

nest.DivergentConnect(pn[0], [sd[0]])
nest.DivergentConnect(pn[1], [sd[1]])

sim_time=stop
nest.Simulate(sim_time)
<

t=nest.GetStatus(sd)[0]['events']['times']
s=nest.GetStatus(sd)[0]['events']['senders']

signal  = zip( s, t )            
signal = MySpikeList( signal, pn[0], 100, sim_time)
fr=signal.get_firing_rate()

ah=signal.get_spike_stats()

print ah.rates['mean'],ah.rates['std']
ax=pylab.subplot(111)
fr.plot(ax, win=5)


t=nest.GetStatus(sd)[1]['events']['times']
s=nest.GetStatus(sd)[1]['events']['senders']

signal  = zip( s, t )            
signal = MySpikeList( signal, pn[1], 100, sim_time, )
fr=signal.get_firing_rate()
ah=signal.get_spike_stats()
print ah.rates['mean'],ah.rates['std']

ax=pylab.subplot(111)
fr.plot(ax,win=5)

# ax.plot()
pylab.show()