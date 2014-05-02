'''
Created on Aug 12, 2013

@author: lindahlm
'''
from copy import deepcopy
import numpy
import pylab
import simulate_network as sni
from toolbox import plot_settings 
# from toolbox.default_params import Perturbation_list as pl
# from toolbox.default_params import Par_slow_wave

# from toolbox.network_handling import Network_model, Network_models_dic
from toolbox.my_axes import MyAxes 
import pylab

from simulate_network import (create_net, create_dic, do, iter_comb, 
                               save_dud)
from toolbox.network.default_params import Perturbation_list as pl
from toolbox.network.engine import Network
from toolbox.network.default_params import Inhibition, Beta
import toolbox.plot_settings as ps
import numpy
import pprint
pp=pprint.pprint

def create_nets(**kwargs):
    
    l=[['low'],
       ['dop', 'no_dop'],
       ['general']]
    l.append([perturbations()[0]])

    nets=[]
    for a in iter_comb(*l):
        name='Beta_net_'+'_'.join(*([list(a[0:3])+[a[-1].name]]))
        net=create_net(name, a[0:3], a[-1], **kwargs)
        nets.append(net)
        
        
    return nets


def create_net(name, dic_calls, per, **kwargs):
    d = create_dic(dic_calls, **kwargs)
        
    par = Beta(**{'dic_rep':d,
                           'other':Inhibition(), 
                           'pertubation':per})
    
    net = Network(name, **{'verbose':True, 
                           'par':par})
         
    return net

def perturbations():
    
    l=sni.perturbations()
    l+=[pl('p_mod-'+str(val), ['netw.input.oscillation.p_amplitude_mod',  val, '*']) for val in [0.8, 0.85, 0.9, 0.95]] 
    return l


def show(duds, nets):
  
    
    for id_dud in range(len(duds)):
        _, axs=ps.get_figure(n_rows=6, n_cols=1, w=1000.0, h=800.0, fontsize=16)   
        axs[0].set_title(nets[id_dud].name)
        duds[id_dud]['M1'].plot_firing_rate(ax=axs[0])
        duds[id_dud]['M2'].plot_firing_rate(ax=axs[1])
        duds[id_dud]['FS'].plot_firing_rate(ax=axs[2])
        duds[id_dud]['GA'].plot_firing_rate(ax=axs[3])
        duds[id_dud]['GI'].plot_firing_rate(ax=axs[3])
        duds[id_dud]['ST'].plot_firing_rate(ax=axs[4])
        duds[id_dud]['SN'].plot_firing_rate(ax=axs[5])  
            
    
    pylab.show()

import pylab
if __name__ == '__main__':
    nets=create_nets(**{'sim_time':3000.0, 'sim_stop':3000.0, 'start_rec':1000.0,
                        'size':10000.0, 'threads':4, 'print_time':True})
    pp(nets[0].par.dic_rep)
    pp(nets[0].replace_perturbation)
    
    duds=do('sim', nets, [0]*len(nets))
    show(duds, nets)
    save_dud(*duds)



# class Network_models_dic_slow_wave(Network_models_dic):
#     
#     def __init__(self,  threads, lesion_setup, setup_list_models, Network_model_class):
#         
#         super( Network_models_dic_slow_wave, self ).__init__( threads, lesion_setup, setup_list_models, Network_model_class) 
#     
#     
#     def show_phase_processing_example(self):
#         plot_settings.set_mode(pylab, mode='by_fontsize', w = 800.0, h = 800, fontsize=12)
#         fig = pylab.figure( facecolor = 'w' )
#         ax_list = []
#         n_rows=4
#         n_col=2
#         ypoints=numpy.linspace(0.1, 0.75, n_rows)
#         xpoints=numpy.linspace(0.1, 0.6, n_col)
#         for x in xpoints:
#             for y in ypoints:
#                 ax_list.append( MyAxes(fig, [ x,  y,  .8/(n_col+0.5), 0.8/(n_rows+1.5) ] ) )  
#         
#         xlim=[0,1000]
#         model='GPE_I'
#         ax=ax_list[0]
#         ax.plot(self.data[model].phase_spk[0:2].transpose())   
#         ax.set_xlim(xlim)                
#         ax.set_title('Raw spike trains')
#         ax.legend(['Neuron 1', 'Neuron 2'])
#                 
#         ax=ax_list[1]
#         ax.plot(self.data[model].get_setupphase_spk_conv[0:2].transpose())
#         ax.set_xlim(xlim)
#         ax.set_title('Convolved '+self.kernel_type+' '+str(self.kernel_extent)+' '+str(self.kernel_params))
#         ax.legend(['Neuron 1', 'Neuron 2'])        
#         
#         ax=ax_list[2]
#         ax.plot(self.data[model].phase_spk_conv[0:2].transpose())
#         ax.set_xlim(xlim)
#         ax.set_title('Bandpass low/high/order '+str(self.lowcut)+'/'+str(self.highcut)+'/'+str(self.order))
#         ax.legend(['Neuron 1', 'Neuron 2'])        
# 
# 
# def main():
# 
#     pds_setup=[1024*4, 10., 'gaussian',{'std_ms':5, 'fs':1000.0}]
#     cohere_setup=[1024*4, 10., 'gaussian',{'std_ms':5, 'fs':1000.0}, 40]
#     pds_models=['GP', 'GA', 'GI', 'ST', 'SN']
#     cohere_relations=['GA_GA', 'GI_GI', 'GA_GI','ST_GA', 'ST_GA']
#         
#     record_from_models=['M1', 'M2', 'FS', 'GA', 'GI', 'ST', 'SN']
#     plot_models=pds_models
#     plot_relations=cohere_relations
#     
#     stop=41000.0 
#     size=40000.0
#     sub_sampling=10.0
#     kwargs = {'class_network_construction':Slow_wave, 
#               'kwargs_network':{'save_conn':False, 'verbose':True}, 
#               'par_rep':{'simu':{'threads':2, 'sd_params':{'to_file':True, 'to_memory':False},
#                                  'print_time':True, 'start_rec':1000.0, 
#                                  'stop_rec':stop, 'sim_time':stop},
#                          'netw':{'size':size/sub_sampling, 'sub_sampling':{'M1':sub_sampling, 'M2':sub_sampling}}}}   
# 
# 
# 
#     pert0= sni.pert_MS_subsampling(sub_sampling)
#     setup_list=[]
#     #check_perturbations()
#     for s in perturbations():
#         s.append(pert0)
#         sni.check_perturbations([s], Par_slow_wave())
#         kwargs['perturbation']=s
#         kwargs['par_rep']['netw'].update({'tata_dop':0.8})      
#         setup_list.append([s.name+'-dop',   deepcopy(kwargs)])
#         kwargs['par_rep']['netw'].update({'tata_dop':0.0})
#         setup_list.append([s.name+'-no_dop', deepcopy(kwargs)])
#     
#     labels=[sl[0] for sl in setup_list]
# 
#     nms=Network_models_dic(setup_list, Network_model)
#     nms.simulate([0]*len(labels), labels, record_from_models)
#     nms.signal_pds([0]*len(labels), labels, pds_models, pds_setup)
#     nms.signal_coherence([0]*len(labels), labels, cohere_relations, cohere_setup)
#     #nms.signal_phase([0]*2, [labels[3]], plot_models[5:8], phase_setup)
#     
#     fig=nms.show_compact(labels, plot_models, plot_relations, band=[0.5,1.5])
#     nms.show_exclude_rasters(labels[0:4]+labels[16:18], plot_models, plot_relations, xlim=[5000.0,7000.0], xlim_pds=[0,5], xlim_coher=[0,5])
#     pylab.show()
#     
#         
# main()    





    