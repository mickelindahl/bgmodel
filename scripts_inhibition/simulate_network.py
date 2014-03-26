'''
Created on Jun 27, 2013

@author: lindahlm
'''
from copy import deepcopy
import itertools
import numpy
import pylab  
from toolbox.network.construction import Network
from toolbox.network.default_params import Perturbation_list as pl
from toolbox.network.default_params import Inhibition 
from toolbox import misc, data_to_disk
import toolbox.plot_settings as ps

import pprint
pp=pprint.pprint


def create_dic(dic_calls, **kwargs):
    d = {}
    for method in dic_calls:
        module = __import__(__name__)
        call = getattr(module, method)
        d = misc.dict_update(d, call(**kwargs))
    
    return d

def create_net(name, dic_calls, per, **kwargs):
    d = create_dic(dic_calls, kwargs)
        
    par = Inhibition(**{'dic_rep':d, 
                        'pertubation':per})
    net = Network(name, **{'verbose':True, 
                           'par':par})
         
    return net

def create_nets(**kwargs):
    
    c1=kwargs.get('call_rev',['low'])
    c2=kwargs.get('call_dop',['dop', 'no_dop'])    
    c3=kwargs.get('call_general',['general']) 
    c4=kwargs.get('call_sub_sampling',['sub_sampling_MSN']) 
    pl=kwargs.get('pl',[perturbations()[0]])
    
    nets=[]
    for c1, c2, c3, c4, p in iter_comb(c1,c2,c3,c4, pl):
        name='_net_'+'_'.join([c1, c2, p.name])
        net=create_net(name, [c1,c2, c3, c4], p)
        nets.append(net)
    return nets

def general(**kwargs):
    d={'simu':{
               'mm_params':{'to_file':False, 'to_memory':True},
               'print_time':True,
               'sd_params':{'to_file':False, 'to_memory':True},
               'sim_stop':kwargs.get('sim_stop', 2000.0),
               'sim_time':kwargs.get('sim_time', 2000.0),
               'start_rec':kwargs.get('start_rec', 1000.0),
               'stop_rec':kwargs.get('stop_rec',numpy.inf),
                
               'threads':kwargs.get('threads', 2),
               },
       'netw':{'size':500}}
    return d

def low(**kwargs):
    return {'node':{'M1':{'model':'M1_low'},
                    'M2':{'model':'M2_low'},
                    'FS':{'model':'FS_low'}}}
          
def high(**kwargs):
    return {'node':{'M1':{'model':'M1_high'},
                    'M2':{'model':'M2_high'},
                    'FS':{'model':'FS_high'}}}
def dop(**kwargs):
    return {'netw':{'tata_dop':0.8}}

def no_dop(**kwargs):
    return {'netw':{'tata_dop':0.0}}

def sub_sampling_MSN(**kwargs):
    d={'netw':{'sub_sampling':{'M1':100,'M2':100}}}
    return d

def perturbations():
    
    l=[]
    l+=[pl()]
    l+=[pl('Size-'+str(val), ['netw.size',  val, '*']) 
        for val in [0.5, 1.0, 1.5]] 
    l+=[pl('M2r-' +str(val), ['node.C2.rate', val, '*']) 
        for val in [1.3, 1.2, 1.1, 0.9, 0.8]] 
    l+=[pl('GAr-' +str(val), ['node.EA.rate', val, '*']) 
        for val in [0.8, 0.6, 0.4, 0.2]] 
    l+=[pl('GISTd-0.5', ['nest.GI_ST_gaba.delay', 0.5, '*'])]    # from GPe type I to STN  
    l+=[pl('STGId-0.5', ['nest.ST_GA_ampa.delay', 0.5, '*'])]     # from STN to GPe type I and A  
    l+=[pl('Bothd-0.5', [['nest.GI_ST_gaba.delay',0.5, '*'], 
                         ['nest.ST_GA_ampa.delay',0.5, '*']])]
    l+=[pl('V_th-0.5', ['netw.V_th_sigma',0.5, '*'])]
    l+=[pl('GAfan-'+str(val), ['netw.prop_fan_in_GPE_A', val, '*' ]) 
        for val in [2, 4, 6]]
    l+=[pl('GArV-0.5', [['netw.V_th_sigma',0.5, '*'], 
                        ['nest.GI_ST_gaba.delay', 0.5, '*']])]
    
    return l



def iter_comb(*args):
    l=list(itertools.product(*args))
    for a in l:
        yield a
    
def sim(obj, load, *args, **kwargs):
    fileName=obj.get_path_data()+obj.get_name()
    if load:
        return data_to_disk.pickle_load(fileName)
    else:
        dud=evaluate('simulation_loop', obj, *args, **kwargs)['spike_signal']
        dud.set_file_name(fileName)
        save_dud(dud)
        
    return dud

def do(method, *args, **kwargs):
    
    module= __import__(__name__)
    call=getattr(module, method)
       
    l=[]
    for a in zip(*args):
        l.append(call(*a, **kwargs))
    return l

def evaluate(method, obj, **k):
    call=getattr(obj, method)
    duds=call(**k)
    return duds

def save_dud(*args):
    for a in args:
        data_to_disk.pickle_save(a, a.get_file_name())

def plot_firing_rate(net, dud, ax=None, **kwargs):
    node=kwargs.get('node', 'FS')
    dud[node].plot_firing_rate(ax=ax, t_start=net.get_start_rec(),
                              t_stop=net.get_sim_stop(), 
                              **{'label':node})    
    
def plot_firing_rates(*args, **kwargs):
    nodes=kwargs['nodes']
    for name in nodes:
        kwargs['node']=name
        plot_firing_rate(*args, **kwargs)

def show_firing_rates(axs, nets, duds):
    
    for i, nodes in enumerate([['M1','M2'],['FS','ST'],['GA','GI'],['SN']]):
        do('plot_firing_rates', nets, duds, **{'ax':axs[i],'nodes':nodes})

def main():
    
    p=perturbations()
    nets=create_nets()
    pp(nets[0].par.dic_rep)
    print nets[0].par
    _, axs=ps.get_figure(n_rows=2, n_cols=2, w=1000.0, h=800.0, fontsize=16)
    
    
    duds=do('sim', nets, [0]*2)
    show_firing_rates(axs, nets, duds)
    save_dud(*duds)
    
    pylab.show()
    
#     stop=11000.0
#     sub_sampling=10.0
#     kwargs = {'class_network_construction':Inhibition_base, 
#               'kwargs_network':{'save_conn':False, 'verbose':True}, 
#               'par_rep':{'simu':{'threads':4, 'sd_params':{'to_file':True, 'to_memory':False},
#                                  'print_time':True, 'start_rec':1000.0, 
#                                  'stop_rec':stop, 'sim_time':stop},
#                              'netw':{'size':10000.0/sub_sampling, 'sub_sampling':{'M1':sub_sampling, 
#                                                                                   'M2':sub_sampling}}}}          
#     
#     pert=pl('MS-sub-samp', [['nest.M1_GI_gaba.weight',  sub_sampling, '*'],
#                             ['nest.M2_SN_gaba.weight',  sub_sampling, '*'],
#                             ['nest.M1_M1_gaba.weight',  sub_sampling, '*'],
#                             ['nest.M1_M2_gaba.weight',  sub_sampling, '*'],
#                             ['nest.M2_M1_gaba.weight',  sub_sampling, '*'],
#                             ['nest.M2_M2_gaba.weight',  sub_sampling, '*']])
#     
#     record_from_models=['M1', 'M2', 'FS', 'GA', 'GI', 'ST', 'SN']
#     labels=['Control', 'No_dopamine']
#     dopamine=[0.8, 0.0]
#     
#     setup_list=[]
#     for l, d in zip(*[labels, dopamine]): 
#         kwargs['par_rep']['netw'].update({'tata_dop':d})      
#         kwargs['perturbations']=pert
#         setup_list.append([l, deepcopy(kwargs)])
#     
#     
#     pds_setup    =[256, 10., 'gaussian',{'std_ms':5, 'fs':1000.0}]
#     cohere_setup =[256, 40., 'gaussian',{'std_ms':20,'fs':1000.0}, 20]
#     pds_models=record_from_models+['GP']
#     cohere_relations=['GP_GP', 'GA_GA', 'GA_GI','GI_GI','ST_GP',
#                       'ST_GA', 'ST_GI']
#     plot_models=pds_models[0:5]
#     plot_relations=cohere_relations[0:5]
#     
#     nms=Network_models_dic(setup_list, Network_model)
#     nms.simulate([1]*2, labels, record_from_models)
#     nms.signal_pds([0]*2, labels, pds_models, pds_setup)
#     nms.signal_coherence([0]*2, labels, cohere_relations, cohere_setup)
#     
#     #fig=nms.show_signal_processing_example( labels[0], 'GPE_I')
#     fig=nms.show(labels, plot_models, plot_relations)
#     fig=nms.show_compact(labels, plot_models, plot_relations)
#     #fig.savefig( nms.path_pictures +'example_sp'+'.svg', format = 'svg') 
#     pylab.show()
#     #fig.savefig( nms.path_pictures +'.svg', format = 'svg') 
    

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main()

   


    

    
