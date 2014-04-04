'''
Created on 25 mar 2014

@author: mikael
'''

from simulate_network import (create_net, create_dic, do, iter_comb, 
                              perturbations, save_dud)
from toolbox.network.default_params import Perturbation_list as pl
from toolbox.network.construction import Network
from toolbox.network.default_params import Inhibition, Burst_compete 
import toolbox.plot_settings as ps
import numpy
import pprint
pp=pprint.pprint



def perturbation_bursts(use):
    
    p1=numpy.linspace(1,3,5)
    p2=4-p1
    p3=2*numpy.ones(len(p1))
    p4=2*numpy.ones(len(p1))
    p5=[1000.+1500*i for i in range(len(p1))]
    l=[]
    for a in zip(p1,p2,p3, p4, p5):
        s=[]
        start=a[4]
        if 'C1' in use: 
            s+=[['netw.input.C1.params.p_amplitude', 
                     numpy.array([a[0],a[1]]), '*']]
            s+=[['netw.input.C1.params.start',start, '=']]
        if 'C2' in use: 
            s+=[['netw.input.C2.params.p_amplitude', 
                             numpy.array([a[1],a[0]]), '*']]
            s+=[['netw.input.C2.params.start',start, '=']]
 
        if 'CF' in use: 
            s+=[['netw.input.CF.params.p_amplitude', 
                             numpy.array([a[2]]), '*']]
            s+=[['netw.input.CF.params.start',start, '=']]
 
        if 'CS' in use: 
            s+=[['netw.input.CS.params.p_amplitude', 
                             numpy.array([a[3]]), '*']]
            s+=[['netw.input.CS.params.start',start, '=']]
 
        l+=[pl('', s)] 

        
    return l

def create_nets(**kwargs):
    
    l=[['low'],
       ['dop'],# 'no_dop'],
       ['general'],
       ['sub_sampling_MSN'],
       [['C1'], 
        ['C2'],
        ['C1', 'C2'],
        ['C1', 'CF'],
        ['C1', 'CS'],
        ['C1', 'C2', 'CF', 'CS']],
#        ['C2', ''],
#        ['CF', ''],
#        ['CS', '']
] 
    l.append([perturbations()[0]])

    nets=[]
    for a in iter_comb(*l):
        
        name='Burst_comptete_net_'+'_'.join(*([list(a[0:4])+[a[-1].name]]))
        net=create_net(name, a[0:4], a[-1], **kwargs)
        
        use=a[4]
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

def show(duds, nets):
  
    
    for id_dud in range(len(duds)):
        _, axs=ps.get_figure(n_rows=6, n_cols=1, w=1000.0, h=800.0, fontsize=16)   
        for i in range(3):
            axs[0].set_title(nets[id_dud].name)
            duds[id_dud]['M1'][:,i].plot_firing_rate(ax=axs[0])
            duds[id_dud]['M2'][:,i].plot_firing_rate(ax=axs[1])
            
            duds[id_dud]['GI'][:,i].plot_firing_rate(ax=axs[3])
            
            duds[id_dud]['SN'][:,i].plot_firing_rate(ax=axs[5])  
        duds[id_dud]['FS'].plot_firing_rate(ax=axs[2])
        duds[id_dud]['ST'].plot_firing_rate(ax=axs[4])
    pylab.show()

import pylab
if __name__ == '__main__':
    nets=create_nets(**{'sim_time':1500.0, 'sim_stop':1500.*5, 'start_rec':0.0,
                        'size':250.0, 'threads':4, 'print_time':False})
    pp(nets[0].par.dic_rep)
    pp(nets[0].replace_perturbation)
    
    duds=do('sim', nets, [0]*len(nets))
    show(duds, nets)
    save_dud(*duds)
    
    