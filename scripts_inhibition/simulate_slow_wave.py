'''
Created on Aug 12, 2013

@author: lindahlm
'''
import numpy
import os
import pylab

from network import show_fr, show_hr, show_psd, show_coherence, show_phase_diff
from toolbox import misc
from toolbox.data_to_disk import Storage_dic
from toolbox.network import manager
from toolbox.network.data_processing import Data_units_relation
from toolbox.network.manager import compute, run, save, load
from toolbox.network.manager import Builder_slow_wave as Builder

import pprint
pp=pprint.pprint
    
DISPLAY=os.environ.get('DISPLAY')
THREADS=20
def get_kwargs_builder(**k_in):
    return {'print_time':True, 
            'threads':THREADS, 
            'save_conn':{'overwrite':False},
            'sim_time':k_in.get('sim_time'), 
            'sim_stop':k_in.get('sim_time'), 
            'size':k_in.get('size'), 
            'start_rec':0.0, 
            'sub_sampling':1}

def get_kwargs_engine():
    return {'verbose':True}

def get_networks(**k_in):
    return manager.get_networks(Builder, 
                                get_kwargs_builder(**k_in), 
                                get_kwargs_engine())



def compute_psd(d_pds, models, dd):
    for key1 in dd.keys():
        for model in models:
            psd=dd[key1][model]['firing_rate'].get_psd(**d_pds)
            dd[key1][model]['psd'] = psd
    


def create_relations(models_coher, dd):
    for key in dd.keys():
        for model in models_coher:
            k1, k2 = model.split('_')
            dd[key][model] = {}
            obj = Data_units_relation(model, dd[key][k1]['spike_signal'], 
                dd[key][k2]['spike_signal'])
            dd[key][model]['spike_signal'] = obj

def main(from_disk=2,
         script_name=__file__.split('/')[-1][0:-3],
         sim_time=40000.0,
         size=20000.0):
    
    k=get_kwargs_builder()
    from os.path import expanduser
    home = expanduser("~")

    d_pds={'NFFT':1024*4,
           'fs': 1000.,
           'noverlap':1024*2,
           'threads':THREADS}
    d_cohere={'fs':1000.0,
              'NFFT':1024*4,
              'noverlap':int(1024*2,),
              'sample':10.,
                      }
    d_phase_diff={'lowcut':0.,
                  'highcut':2.0,
                  'order':3,
                  'fs':1000.0,
                  'bin_extent':10.,
                  'kernel_type':'gaussian',
                  'params':{'std_ms':5.,
                            'fs': 1000.0}}
    attr=[ 'firing_rate', 
           'mean_rates', 
           'spike_statistic']  
    attr_coher=['mean_coherence',
                'phase_diff']
    
    kwargs_dic={'firing_rate':{'threads':THREADS},
                'mean_rates': {'t_start':k['start_rec']+1000.},
                'mean_coherence':d_cohere,
                'phase_diff':d_phase_diff,
                'spike_statistic': {'t_start':k['start_rec']+1000.},}
    file_name=(home+ '/results/papers/inhibition/network/'
               +script_name)
    
    models=['M1', 'M2', 'FS', 'GI', 'GA', 'ST', 'SN']
    models_coher=['GI_GI', 'GI_GA', 'GA_GA', 'GA_ST','GI_ST' ]
    
    info, nets, _ = get_networks(**{'size':size, 'sim_time':sim_time})

    sd=Storage_dic.load(file_name)
    sd.add_info(info)
    sd.garbage_collect()
    
    d={}
    from_disks=[from_disk]*2
    for net, fd in zip(nets, from_disks):
        if fd==0:
            dd = run(net)  
            dd = compute(dd, models,  attr, **kwargs_dic)      
            save(sd, dd)
            
        elif fd==1:
            filt=[net.get_name()]+models+['spike_signal']
            dd=load(sd, *filt)
            create_relations(models_coher, dd)       
            dd = compute(dd, models_coher, attr_coher, **kwargs_dic)  
            save(sd, dd)

        elif fd==2:
            filt=[net.get_name()]+models+models_coher+['spike_signal']+attr+attr_coher
            dd=load(sd, *filt)
            
            dd = compute(dd, models,  ['firing_rate'], **kwargs_dic)  
            compute_psd(d_pds, models, dd)
        d=misc.dict_update(d, dd)

    
    if numpy.all(numpy.array(from_disks)==2):                     
        figs=[]
    
        figs.append(show_fr(d, models))
        figs.append(show_hr(d, models))
        figs.append(show_psd(d, models=models))
        figs.append(show_coherence(d, models=models_coher))
        figs.append(show_phase_diff(d, models=models_coher))
        
        sd.save_figs(figs)
    
    if DISPLAY: pylab.show()  

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main()
    
# from copy import deepcopy
# import numpy
# import pylab
# import simulate_network as sni
# from toolbox import plot_settings 
# # from toolbox.default_params import Perturbation_list as pl
# # from toolbox.default_params import Par_slow_wave
# 
# # from toolbox.network_handling import Network_model, Network_models_dic
# from toolbox.my_axes import MyAxes 
# import pylab
# 
# from simulate_network import (create_net, create_dic, do, iter_comb, 
#                                save_dud)
# from toolbox.network.default_params import Perturbation_list as pl
# from toolbox.network.engine import Network
# from toolbox.network.default_params import Inhibition, Slow_wave 
# import toolbox.plot_settings as ps
# import numpy
# import pprint
# pp=pprint.pprint
# 
# def create_nets(**kwargs):
#     
#     l=[['low'],
#        ['dop', 'no_dop'],
#        ['general']]
#     l.append([perturbations()[0]])
# 
#     nets=[]
#     for a in iter_comb(*l):
#         name='_net_'+'_'.join(*([list(a[0:3])+[a[-1].name]]))
#         net=create_net(name, a[0:3], a[-1], **kwargs)
#         nets.append(net)
#         
#         
#     return nets
# 
# 
# def create_net(name, dic_calls, per, **kwargs):
#     d = create_dic(dic_calls, **kwargs)
#         
#     par = Slow_wave(**{'dic_rep':d,
#                            'other':Inhibition(), 
#                            'pertubation':per})
#     
#     net = Network(name, **{'verbose':True, 
#                            'par':par})
#          
#     return net
# 
# def perturbations():
#     
#     l=sni.perturbations()
#     l+=[pl('p_mod-'+str(val), ['netw.input.oscillation.p_amplitude_mod',  val, '*']) for val in [0.8, 0.85, 0.9, 0.95]] 
#     return l
# 
# 
# def show(duds, nets):
#   
#     
#     for id_dud in range(len(duds)):
#         _, axs=ps.get_figure(n_rows=6, n_cols=1, w=1000.0, h=800.0, fontsize=16)   
#         axs[0].set_title(nets[id_dud].name)
#         duds[id_dud]['M1'].plot_firing_rate(ax=axs[0])
#         duds[id_dud]['M2'].plot_firing_rate(ax=axs[1])
#         duds[id_dud]['FS'].plot_firing_rate(ax=axs[2])
#         duds[id_dud]['GA'].plot_firing_rate(ax=axs[3])
#         duds[id_dud]['GI'].plot_firing_rate(ax=axs[3])
#         duds[id_dud]['ST'].plot_firing_rate(ax=axs[4])
#         duds[id_dud]['SN'].plot_firing_rate(ax=axs[5])  
#             
#     
#     pylab.show()
# 
# import pylab
# if __name__ == '__main__':
#     nets=create_nets(**{'sim_time':10000.0, 'sim_stop':10000.0, 'start_rec':1000.0,
#                         'size':10000.0, 'threads':4, 'print_time':True})
#     pp(nets[0].par.dic_rep)
#     pp(nets[0].replace_perturbation)
#     
#     duds=do('sim', nets, [1]*len(nets))
#     show(duds, nets)
#     save_dud(*duds)



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





    