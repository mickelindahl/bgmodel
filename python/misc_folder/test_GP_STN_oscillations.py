'''
Created on Sep 11, 2014

@author: mikael
'''


import pylab
import random
import numpy
from scripts_inhibition.simulate_beta import Setup
from core.network import default_params
from core import my_nest
from core import misc
from core.my_population import MyNetworkNode
import pprint
pp=pprint.pprint

from core.network.manager import get_storage_list, save, load
from core import directories as dir
from core import data_to_disk
import os

path=dir.HOME_DATA+'/'+__file__.split('/')[-1][0:-3]    
if not os.path.isdir(path):
    data_to_disk.mkdir(path)
par=default_params.Inhibition()
setup=Setup(50,20) 

def gs_builder(*args, **kwargs):
    import matplotlib.gridspec as gridspec
    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',1)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0.1 ), 
              hspace=kwargs.get('hspace', 0.1 ))

    iterator = [[slice(0,1),slice(0,1)],
                [slice(0,1),slice(1,2)],
                [slice(1,2),slice(0,1)],
                [slice(1,2),slice(1,2)],
                [slice(2,3),slice(0,1)],
                [slice(2,3),slice(1,2)],
                [slice(3,4),slice(0,2)],
                ]
    
    return iterator, gs,     


def get_fig_axs():
    scale=4
    kw={'n_rows':4, 
        'n_cols':2, 
        'w':72/2.54*18*scale, 
        'h':300*scale, 
        'fontsize':7*scale,
        'frame_hight_y':0.5,
        'frame_hight_x':0.7,
        'title_fontsize':7*scale,
        'font_size':7*scale,
        'text_fontsize':7*scale,
        'linewidth':1.*scale,
        'gs_builder':gs_builder}
#     kwargs_fig=kwargs.get('kwargs_fig', kw)
    from core import plot_settings as ps
    fig, axs=ps.get_figure2(**kw) 
    return fig, axs


def simulate_network(**kw):
    my_nest.ResetKernel()
    my_nest.SetKernelStatus({'local_num_threads':4,
#                              'print_time':True
                             })
    
    p_gi=par.dic['nest']['GI']
    p_st=par.dic['nest']['ST']
    
    p_st.update(kw.get('p_st', {}))
    
    
    for d in [p_gi, p_st]:
        if 'type_id' in d: del d['type_id']
    
    sd={'active':True,
        'params':{'to_memory':True,
                  'to_file':False,
                  'start':500.0
                  }}
    gi=MyNetworkNode('gi',model='my_aeif_cond_exp', n=kw.get('gi_n'), params=p_gi, sd=sd)
#     st=MyNetworkNode('gi',model='my_aeif_cond_exp', n=kw.get('st_n'), params=p_gi, sd=sd)
    st=MyNetworkNode('st',model='my_aeif_cond_exp', n=kw.get('st_n'), params=p_st, sd=sd)
    

    for key in ['GI_ST_gaba', 'GI_GI_gaba', 'ST_GI_ampa','EI_GI_ampa', 'CS_ST_ampa']:
        d=par.dic['nest'][key]
        if 'type_id' in d: del d['type_id']
        my_nest.CopyModel('static_synapse', key, d)

    
    df=my_nest.GetDefaults('my_aeif_cond_exp')['receptor_types']
    for post in st:
        idx=random.sample(range(kw.get('gi_n')),30)
#         params={'receptor_type':df['GABAA_1']}
        delay=kw['gi_st_delay']
        model='GI_ST_gaba'
        my_nest.Connect([pre for i, pre in enumerate(gi) if i in idx ], [post]*30, [0.08]*30, [delay]*30, model)
 
    for pre in gi:
        idx=random.sample(range(kw.get('gi_n')),30)
#         params={'receptor_type':df['GABAA_1']}
        delay=kw['gi_gi_delay']
        model='GI_GI_gaba'
        my_nest.Connect([pre]*30, [post for i, post in enumerate(gi) if i in idx ], [1.3]*30, [delay]*30, model)
    
    for pre in gi:
        idx=random.sample(range(kw.get('st_n')),30)
#         params={'receptor_type':df['AMPA_1']}
        delay=kw['st_gi_delay']
        model='ST_GI_ampa'
        my_nest.Connect([pre for i, pre in enumerate(st) if i in idx ], [post]*30, [0.35]*30, [delay]*30, model)
       
     
    d={'rate_first':1400.*(1+kw.get('gi_amp',0)),'rate_second':1400.**(1-kw.get('gi_amp',0)),
       'period_first':25.0,  'period_second':25.}
    inp_gi=my_nest.Create('poisson_generator_periodic',n=kw.get('gi_n'),  params=d)    
    
    d={'rate_first':400.*(1+kw.get('st_amp',0)),'rate_second':400.**(1-kw.get('st_amp',0)),
       'period_first':25.0,  'period_second':25.}
    inp_st=my_nest.Create('poisson_generator_periodic',n=kw.get('st_n'),  params=d)
        
    my_nest.Connect(inp_gi,gi.ids, 0.25, 1., model='EI_GI_ampa')
    my_nest.Connect(inp_st,st.ids, 0.25, 1., model='CS_ST_ampa')
    
    my_nest.Simulate(kw.get('sim_time'), chunksize=1000.0)
    
    d={}

    from core.network.data_processing import Data_unit_spk, Data_units_relation    
    gi_dus=Data_unit_spk('gi',gi.get_spike_signal())
    d['gi']={'spike_signal':gi_dus}
    st_dus=Data_unit_spk('st',st.get_spike_signal())
    d['st']={'spike_signal':st_dus}
    
    d['gi_st']={'spike_signal':Data_units_relation('gi_st', gi_dus, st_dus)}
       
    return d
    
def compute_attrs(d, net):
    pp(d)
    module=misc.import_module('core.network.manager')
    for key in ['st','gi']:
        for attr, kw in [['firing_rate',setup.firing_rate()]]:
            call=getattr(module, attr)
         
            obj=call(d[net][key]['spike_signal'], **kw)    
            d[net][key][attr]=obj
#     key='gi_st'
#     for attr, kw in [['phases_diff_with_cohere', setup.phases_diff_with_cohere()]]:
#         kw
#         call=getattr(module, attr)
#     
#         obj=call(d[net][key]['spike_signal'], **kw)    
#         d[net][key][attr]=obj
    
    
    return d


if __name__=='__main__':
    
    p_st_list=[
#                 {},
              {'a_1':.3,
              'a_2':0.,
              'b':0.05,
              'Delta_T':16.,
              'E_L':-80.2,
              'V_a':-70.,
              'V_th':-64.0,
              'g_L':10.,
              'V_reset_max_slope1':-50. },
#                              {'a_1':1.,
#               'a_2':0.,
#               'b':0.25,
#               'Delta_T':16.4,
#               'E_L':-53.,
#               'V_a':-53.,
#               'V_th':-50.0,
#               'g_L':5.,
#               'V_reset_max_slope1':-50. },
#                
#               {'a_1':1.,
#               'a_2':0.,
#               'b':0.25,
#               'Delta_T':8.4,
#               'E_L':-58.4,
#               'V_a':-58.4,
#               'V_th':-50.0,
#               'g_L':5.,
#               'V_reset_max_slope1':-50. },
                             
#               {'a_1':.7,
#               'a_2':0.,
#               'b':0.15,
#               'Delta_T':5.6,
#               'E_L':-55.6,
#               'V_a':-55.6,
#               'V_th':-50.0,
#               'g_L':5.,
#               'V_reset_max_slope1':-50. },
#                
#              {'a_1':1.,
#               'a_2':0.,
#               'b':0.25,
#               'Delta_T':2.8,
#               'E_L':-52.8,
#               'V_a':-52.8,
#               'V_th':-50.0,
#               'g_L':5.,
#               'V_reset_max_slope1':-50. },
             ]


    from_disk=0
    kw_list=[]

    for p_st in p_st_list:
        for delay in [3.,]:
            kw={
                
                'gi_amp':1,
                'gi_n':600,
                
                'gi_st_delay':delay,
                'gi_gi_delay':1.,
                'st_gi_delay':delay,
                
                'p_st':p_st,
                
                'sim_time':3500.0,
                
                'st_amp':0,
                'st_n':200}
            kw_list.append(kw)
    
    
    nets=['Net_{0:0>2}'.format(i) for i in range(len(kw_list))]
    sd_list=get_storage_list(nets, path, '')
    
    d={}
    for net, sd, kw in zip(nets, sd_list, kw_list):
    
        if from_disk==0:# and net=='Net_05':
            print net
            dd={net:simulate_network(**kw)}
            dd=compute_attrs(dd, net)
            save(sd, dd)
        
        if from_disk==1:
            filt= [net]+['gi','st', 'gi_st']+['spike_signal']
            dd = load(sd, *filt)
            pp(dd)
            dd=compute_attrs(dd, net)
            save(sd, dd)
            
        elif from_disk==2:
            filt = [net]+['gi','st', 'gi_st']+['firing_rate', 'phases_diff_with_cohere'] 
            dd = load(sd, *filt)
        
        d = misc.dict_update(d, dd)
    
    pp(d)
    fig, axs=get_fig_axs()
    
    
    for i in range(len(nets)):
        d['Net_0{0}'.format(i)]['gi']['firing_rate'].plot(axs[i], **{'win':1.})
        d['Net_0{0}'.format(i)]['st']['firing_rate'].plot(axs[i], **{'win':1.})
        axs[i].set_xlim(0.0,1500.0)
 
    colors=misc.make_N_colors('copper', len(kw_list))
    
    
#     for net, c in zip(nets, colors):
#         kw={ 'all':True,
#            'color':c,
#            'p_95':False,}
#         d[net]['gi_st']['phases_diff_with_cohere'].hist(axs[-1], **kw)
#     axs[-1].set_xlim(-numpy.pi, numpy.pi)
 
    pylab.show() 
    
    
    
    
    # pylab.plot(d['n'])
    