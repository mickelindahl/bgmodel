'''
Created on Sep 11, 2014

@author: mikael
'''

import numpy
import pylab
import random
from core.network import default_params
from core import my_nest
from core import misc
from core.my_population import MyNetworkNode
import pprint
pp=pprint.pprint

from core import directories as dr
from core.network.manager import get_storage_list, save, load
from core.data_to_disk import Storage_dic


par=default_params.Inhibition()


class Setup(object):

    def __init__(self, period, local_num_threads, **k):
        self.fs=256.
        self.local_num_threads=local_num_threads
        self.period=period
       
    def phases_diff_with_cohere(self):
        d={
            'fs':self.fs, 
            'NFFT':128, 
            'noverlap':int(128/2), 
            'sample':30.**2,  
            
            'lowcut':15, 
            'highcut':25., 
            'order':3, 
           
            'min_signal_mean':0.0, #minimum mean a signal is alowed to have, toavoid calculating coherence for 0 signals
            'exclude_equal_signals':True, #do not compute for equal signals

            'local_num_threads':self.local_num_threads}
        return d

    def firing_rate(self):
        d={'average':False, 
           'local_num_threads':self.local_num_threads,
#            'win':100.0,
           'time_bin':1000.0/self.fs}
        
        return d


def gs_builder(*args, **kwargs):
    import matplotlib.gridspec as gridspec
    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',1)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0.1 ), 
              hspace=kwargs.get('hspace', 0.4 ))

    iterator = [[slice(0,1),slice(0,1)],
                [slice(1,2),slice(0,1)],
                ]
    
    return iterator, gs,     


def get_fig_axs(scale=4):
    kw={'n_rows':2, 
        'n_cols':1, 
        'w':72/2.54*11.6*scale, 
        'h':150*scale, 
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


def simulate_network(*args, **kw):
    my_nest.ResetKernel()
    my_nest.SetKernelStatus({'local_num_threads':kw.get('local_num_threads'),
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

    for post in st:
        idx=random.sample(range(kw.get('gi_n')),30)
        delay=kw['gi_st_delay']
        model='GI_ST_gaba'
        my_nest.Connect([pre for i, pre in enumerate(gi) if i in idx ], [post]*30, [0.08]*30, [delay]*30, model)
 
    for pre in gi:
        idx=random.sample(range(kw.get('gi_n')),30)
        delay=kw['gi_gi_delay']
        model='GI_GI_gaba'
        my_nest.Connect([pre]*30, [post for i, post in enumerate(gi) if i in idx ], [1.3]*30, [delay]*30, model)
    
    for pre in gi:
        idx=random.sample(range(kw.get('st_n')),30)
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
    
def compute_attrs(d, net, setup):
    
    pp(d)
    module=misc.import_module('core.network.manager')
    for key in ['st','gi']:
        for attr, kw in [['firing_rate',setup.firing_rate()]]:
            call=getattr(module, attr)
         
            obj=call(d[net][key]['spike_signal'], **kw)    
            d[net][key][attr]=obj
            
    key='gi_st'
    for attr, kw in [['phases_diff_with_cohere', setup.phases_diff_with_cohere()]]:
        kw
        call=getattr(module, attr)
     
        obj=call(d[net][key]['spike_signal'], **kw)    
        d[net][key][attr]=obj
    
    
    return d

def simulate(from_disk=0,
             kw={},
             net='Net_0',
             script_name=__file__.split('/')[-1][0:-3],
             setup=Setup(50,20) ):
    
    setup=Setup(50,20) 
    file_name = kw.get('file_name', dr.HOME_DATA+'/'+script_name)
    file_name_figs = kw.get('file_name_figs', dr.HOME_DATA+'/fig/'+script_name)
    
    sd=get_storage_list([net], file_name, '')[0]
    
    d={}
    if from_disk==0:
        print net
        dd={net:simulate_network(**kw)}
        dd=compute_attrs(dd, net, setup)
        save(sd, dd)
    
    if from_disk==1:
        filt= [net]+['gi','st', 'gi_st']+['spike_signal']
        dd = load(sd, *filt)
        pp(dd)
        dd=compute_attrs(dd, net, setup)
        save(sd, dd)
        
    elif from_disk==2:
        filt = [net]+['gi','st', 'gi_st']+['firing_rate', 'phases_diff_with_cohere'] 
        dd = load(sd, *filt)
    
    d = misc.dict_update(d, dd)

    return d, file_name_figs
    
def create_figs(d, file_name_figs, **kw):
    
    net=kw['net']
    fig, axs=get_fig_axs(scale=kw.get('scale',3))
    
    figs=[fig]

    ax=axs[0]
    d[net]['gi']['firing_rate'].plot(ax, **{'win':1.})
    d[net]['st']['firing_rate'].plot(ax, **{'win':1.})
    ax.set_xlim(1000.0,1500.0)
     
    kw={ 'all':True,
         'color':'b',
         'p_95':False,}
    
    d[net]['gi_st']['phases_diff_with_cohere'].hist(axs[-1], **kw)
    axs[-1].set_xlim(-numpy.pi, numpy.pi)
 
    sd_figs = Storage_dic.load(file_name_figs)
    sd_figs.save_figs(figs, format='png', dpi=200)
    sd_figs.save_figs(figs[1:], format='svg', in_folder='svg')
 
 
    for ax in axs:
        ax.my_set_no_ticks(yticks=4)
#     pylab.show() 
    # pylab.plot(d['n'])

def main(*args, **kwargs):
    
    args=simulate(*args, **kwargs)
    create_figs(*args,**kwargs)



def run_simulation(from_disk=0, local_num_threads=10):
    
    
    p_st={'a_1':.7,
          'a_2':0.,
          'b':0.15,
          'Delta_T':5.6,
          'E_L':-55.6,
          'V_a':-55.6,
          'V_th':-50.0,
          'g_L':5.,
          'V_reset_max_slope1':-50. }

    delay=3.
    kw={    
        'gi_amp':1,
        'gi_n':300,
        
        'gi_st_delay':delay,
        'gi_gi_delay':1.,
        
        'local_num_threads':local_num_threads,
        
        'p_st':p_st,
        
        'sim_time':3500.0, 
        'st_gi_delay':delay,
        'st_amp':0,
        'st_n':100,
        }
    
    args=simulate(from_disk=from_disk,
                   kw=kw,
                   net='Net_0',
                   script_name=__file__.split('/')[-1][0:-3],
                   setup=Setup(50,20) )
    
    return args
  
import unittest
class TestGP_STN_0csillation(unittest.TestCase):     
    def setUp(self):
        
        v=run_simulation(from_disk=0, local_num_threads=10)
        d, file_name_figs=v
                
        self.d=d
        self.file_name_figs=file_name_figs
        self.net=net
        
    def test_create_figs(self):
        create_figs(self.d,
                    self.file_name_figs, 
                    self.net,
                    **{'scale':4} )
        pylab.show()
     
    
if __name__ == '__main__':
    d={
       TestGP_STN_0csillation:[
                        'test_create_figs',
                        ],}

    test_classes_to_run=d
    suite = unittest.TestSuite()
    for test_class, val in  test_classes_to_run.items():
        for test in val:
            suite.addTest(test_class(test))

    unittest.TextTestRunner(verbosity=2).run(suite)    
    
    
    