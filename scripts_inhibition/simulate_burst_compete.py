'''
Created on 25 mar 2014

@author: mikael
'''

from simulate_network import create_net, create_dic, do, iter_comb, perturbations
from toolbox.network.default_params import Perturbation_list as pl
from toolbox.network.construction import Network
from toolbox.network.default_params import Inhibition, Burst_compete 
import numpy
import pprint
pp=pprint.pprint



def perturbation_bursts(use):
    
    p1=numpy.linspace(1,2,5)
    p2=3-p1
    p3=2*numpy.ones(len(p1))
    p4=2*numpy.ones(len(p1))
    l=[]
    for a in zip(p1,p2,p3, p4):
        s=[]
        if 'C1' in use: s+=[['netw.input.C1.params.p_amplitude', 
                             numpy.array([a[0],a[1]]), '*']]
        if 'C2' in use: s+=[['netw.input.C2.params.p_amplitude', 
                             numpy.array([a[1],a[0]]), '*']]
        if 'CF' in use: s+=[['netw.input.CF.params.p_amplitude', 
                             numpy.array([a[2]]), '*']]
        if 'CS' in use: s+=[['netw.input.CF.params.p_amplitude', 
                             numpy.array([a[3]]), '*']]
        l+=[pl('', s)] 

        
    return l

def create_nets(**kwargs):
    
    l=[['low'],
       ['dop', 'no_dop'],
       ['general'],
       ['sub_sampling_MSN'],
       ['C1'],
#        ['C2', ''],
#        ['CF', ''],
#        ['CS', '']
] 
    l.append([perturbations()[0]])

    nets=[]
    for a in iter_comb(*l):
        
        name='_net_'+'_'.join(*([list(a[0:4])+[a[-1].name]]))
        net=create_net(name, a[0:4], a[-1], **kwargs)
        
        use=a[4:-1]
        net.set_replace_pertubation(perturbation_bursts(use))
        nets.append(net)
        
        
    return nets

def create_net(name, dic_calls, per, **kwargs):
    d = create_dic(dic_calls, **kwargs)
        
    par = Burst_compete(**{'dic_rep':d,
                           'other':Inhibition(), 
                           'pertubation':per})
    
    net = Network(name, **{'verbose':True, 
                           'par':par})
         
    return net

import pylab
if __name__ == '__main__':
    nets=create_nets(**{'sim_time':1500.0, 'sim_stop':1500.0*5, 'start_rec':0.0})
    pp(nets[0].par.dic_rep)
    pp(nets[0].replace_perturbation)
    
    duds=do('sim', nets, [0]*2)
    duds[0]['M1'][:,0].plot_firing_rate()
    duds[0]['M1'][:,1].plot_firing_rate()
    duds[0]['M1'][:,2].plot_firing_rate()
    pylab.show()
    
    pass