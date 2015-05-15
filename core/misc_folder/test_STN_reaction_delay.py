'''
Created on Sep 11, 2014

@author: mikael
'''


import pylab
import random
import numpy
from simulate_beta import Setup
from toolbox.network import default_params
from toolbox import my_nest
from toolbox import misc
from toolbox.my_population import MyNetworkNode
import pprint
pp=pprint.pprint

from toolbox.network.manager import get_storage_list, save, load
from toolbox import directories as dir
from toolbox import data_to_disk
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
                [slice(1,2),slice(0,1)],
                ]
    
    return iterator, gs,     


def get_fig_axs():
    scale=4
    kw={'n_rows':2, 
        'n_cols':1, 
        'w':72/2.54*18*scale, 
        'h':225*scale, 
        'fontsize':7*scale,
        'frame_hight_y':0.5,
        'frame_hight_x':0.7,
        'title_fontsize':7*scale,
        'font_size':7*scale,
        'text_fontsize':7*scale,
        'linewidth':1.*scale,
        'gs_builder':gs_builder}
#     kwargs_fig=kwargs.get('kwargs_fig', kw)
    from toolbox import plot_settings as ps
    fig, axs=ps.get_figure2(**kw) 
    return fig, axs


def simulate_network(**kw):
    my_nest.ResetKernel()
    my_nest.SetKernelStatus({'local_num_threads':1,
#                              'print_time':True
                             })
    
    p_gi=par.dic['nest']['GI']
    p_st=par.dic['nest']['ST']
    for d in [p_gi, p_st]:
        if 'type_id' in d: del d['type_id']
    
    mm={'active':True,
        'params':{'to_memory':True,
                  'to_file':False}}
    st=MyNetworkNode('st',model='my_aeif_cond_exp', n=1, params=p_st, mm=mm)
    

    for key in [ 'GI_ST_gaba','CS_ST_ampa']:
        d=par.dic['nest'][key]
        if 'type_id' in d: del d['type_id']
        my_nest.CopyModel('static_synapse', key, d)

 
    d={'rate_first':30**2.*(1+kw.get('gi_amp',0)),'rate_second':30**2.**(1-kw.get('gi_amp',0)),
       'period_first':25.0,  'period_second':25.}
    inp_st_gaba=my_nest.Create('poisson_generator_periodic',n=1,  params=d)    
        
    d={'rate_first':400.*(1+kw.get('st_amp',0)),'rate_second':400.**(1-kw.get('st_amp',0)),
       'period_first':25.0,  'period_second':25.}
    inp_st_ampa=my_nest.Create('poisson_generator_periodic',n=1,  params=d)
     
        
    my_nest.Connect(inp_st_gaba, st.ids, 0.08, 1., model='GI_ST_gaba')
    my_nest.Connect(inp_st_ampa, st.ids, 0.25, 1., model='CS_ST_ampa')
    
    my_nest.Simulate(kw.get('sim_time'), chunksize=1000.0)
    
    d={}

    from toolbox.network.data_processing import Data_unit_vm    
    st_dus=Data_unit_vm('st',st.get_voltage_signal())
    d['st']={'vm_signal':st_dus}
        
    return d
    
# def compute_attrs(d, net):
#     pp(d)
#     module=misc.import_module('toolbox.network.manager')
#     for key in ['st','gi']:
#         for attr, kw in [['firing_rate',setup.firing_rate()]]:
#             call=getattr(module, attr)
#          
#             obj=call(d[net][key]['spike_signal'], **kw)    
#             d[net][key][attr]=obj
#     key='gi_st'
#     for attr, kw in [['phases_diff_with_cohere',setup.phases_diff_with_cohere()]]:
#         call=getattr(module, attr)
#     
#         obj=call(d[net][key]['spike_signal'], **kw)    
#         d[net][key][attr]=obj
#     
#     
#     return d


if __name__=='__main__':
    from_disk=0
    kw_list=[]
    sim_time=1000.
    for gi_amp in [1.]:
        for delay in range(1,10,10):
            kw={
                
                'gi_amp':gi_amp,
#                 'gi_n':600,
                
                'gi_st_delay':1.,
                'gi_gi_delay':1.,
                'st_gi_delay':1.,
                
                'sim_time':sim_time,
                
                'st_amp':0,
#                 'st_n':200
                }
            kw_list.append(kw)
    
    
    nets=['Net_{0:0>2}'.format(i) for i in range(len(kw_list))]
    sd_list=get_storage_list(nets, path, '')
    
    d={}
    for net, sd, kw in zip(nets, sd_list, kw_list):
    
        if from_disk==0:
            print net
            dd={net:simulate_network(**kw)}
#             dd=compute_attrs(dd, net)
            save(sd, dd)
        
#         if from_disk==1:
#             filt= [net]+['gi','st', 'gi_st']+['spike_signal']
#             dd = load(sd, *filt)
#             pp(dd)
#             dd=compute_attrs(dd, net)
#             save(sd, dd)
            
        elif from_disk==2:
            filt = [net]+['gi','st', 'gi_st']+['vm_signal'] 
            dd = load(sd, *filt)
        d = misc.dict_update(d, dd)
    
    pp(d)
    fig, axs=get_fig_axs()
    
    d['Net_00']['st']['vm_signal'].plot(axs[0])
    x=numpy.arange(sim_time)
    pylab.plot(x, -60+2*numpy.sin((20*x*numpy.pi*2)/1000))
#     d['Net_00']['st']['firing_rate'].plot(axs[0], **{'win':1.})
 
#     colors=misc.make_N_colors('copper', len(kw_list))
#     
#     
#     for net, c in zip(nets, colors):
#         kw={ 'all':True,
#            'color':c,
#            'p_95':False,}
#         d[net]['gi_st']['phases_diff_with_cohere'].hist(axs[1], **kw)
#     axs[1].set_xlim(-numpy.pi, numpy.pi)
#  
    pylab.show() 
    
    
    
    
    # pylab.plot(d['n'])
    