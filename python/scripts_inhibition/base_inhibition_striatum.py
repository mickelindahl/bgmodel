'''
Created on Jun 27, 2013

@author: lindahlm
'''
import pythonpath #must be imported first
import numpy
import os

from os.path import expanduser
from scripts_inhibition.base_simulate import (main_loop, show_fr, show_mr, 
                      get_file_name, get_file_name_figs,
                      get_path_nest)
from core import pylab
from core.data_to_disk import Storage_dic
from core.network import manager
from core.network.manager import (add_perturbations,
                                      get_storage_list)
from core.network.manager import Builder_inhibition_striatum as Builder
import core.plot_settings as ps
import pprint
pp=pprint.pprint
    
DISPLAY=os.environ.get('DISPLAY')

def get_kwargs_builder(**k_in):
    return {'print_time':False, 
            'local_num_threads':8, 
            'save_conn':{'overwrite':False},
            'sim_time':8000.0, 
            'sim_stop':8000.0, 
            'size':3000.0, 
            'start_rec':0.0, 
            'sub_sampling':1,
            
            'repetition':k_in.get('repetition',1.),            
            'resolution':k_in.get('resolution',1.),
            'lower':k_in.get('lower',1.),
            'upper':k_in.get('upper',2.),
            
            
            }

def get_kwargs_engine():
    return {'verbose':True}

def get_networks(builder, k_builder, k_director, k_default_params):
    info, nets, builder=manager.get_networks(builder,
                                             get_kwargs_builder(**k_builder),
                                             k_director, #director
                                             get_kwargs_engine(),
                                             k_default_params)
    
    intervals=builder.dic['intervals']
    rates=builder.dic['amplitudes']
    rep=builder.dic['repetitions']
    
    return info, nets, intervals, rates, rep

def gs_builder(*args, **kwargs):
    import matplotlib.gridspec as gridspec
    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',3)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0. ), 
              hspace=kwargs.get('hspace', 0.5 ))

    iterator = [[slice(5,10),slice(1,7)],
                [slice(10,15),slice(1,7)],
                [slice(5,10),slice(10,16)],
                [slice(10,15),slice(10,16)],
               ]
    
    return iterator, gs, 

class Setup(object):
    
    def __init__(self, **k):
        self.local_num_threads=k.get('local_num_threads',1)
        self.nets_to_run=k.get('nets_to_run',[])
        self.res=k.get('resolution',3)
        self.rep=k.get('repetition',1)
        self.low=k.get('lower',1.)
        self.upp=k.get('upper',2.)
   

    def builder(self):
        d= {'repetition':self.rep,
            'resolution':self.res,
            'lower':self.low, 
            'upper':self.upp}
        return d
    
    def default_params(self):
        return {}
    
    def director(self):
        return {'nets_to_run':self.nets_to_run}
    
    def firing_rate(self):
        d={'average':False, 
           'local_num_threads':self.local_num_threads, 
           'win':100.0}
        return d
    
    def plot_fr(self):
        d={'labels':['All', 
                     'Only MSN-MSN',
                     'Only FSN-MSN',
                     'Only FSN-MSN-static',
                     'Only GPe TA-MSN',
                     'No inhibition'],
           'win':100.,
           't_start':0.0,
           't_stop':20000.0,           
           'fig_and_axes':{'n_rows':2, 
                            'n_cols':1, 
                            'w':600.0, 
                            'h':500.0, 
                            'fontsize':16},
                      'x_lim':[1,1.5], 
                      'x_lim':[1,30], 
           }
        return d


    def plot_mr_general(self):
        d={'fig_and_axes':{'n_rows':17, 
                        'n_cols':16, 
                        'w':72*8.5/2.54, 
                        'h':150, 
                        'fontsize':7,
                        'frame_hight_y':0.5,
                        'frame_hight_x':0.7,
                        'title_fontsize':7,
                        'font_size':7,
                        'text_fontsize':7,
                        'linewidth':1.,
                        'gs_builder':gs_builder},
           }
        return d
        

    def plot_mr(self):
        d={'labels':['All', 
                     r'Only MSN$\to$MSN',
                     r'Only FSN$\to$MSN',
                     r'Only FSN$\to$$MSN^{static}',
                     r'Only $GPe_{TA}$\to$MSN',
                     r'No inhibition'],
                      'x_lim':[0.8,1.5],
                      'y_lim':[0,30]}
        return d
        
    def plot_mr2(self):
        d={'labels':['All', 
                     r'Only MSN$\to$MSN',
                     r'Only FSN$\to$MSN',
                     r'Only FSN$\to$$MSN^{static}$',
                     r'Only $GPe_{TA}$$\to$MSN',
                     r'No inhibition'],
           'fontsize':24,
           'relative':True,
           'relative_to':[5,0],
           'x_lim':[0.8,1.5],
           'y_lim':[0,1.1],
           'delete':[0,5],
           }
        return d
    


def simulate(builder, 
             from_disk, 
             perturbation_list, 
             script_name, 
             setup):
    

#     file_name = get_file_name(script_name, home)
#     file_name_figs = get_file_name_figs(script_name, home)
    
    attr = ['firing_rate', 'mean_rate_slices']
    models = ['M1', 'M2', 'FS', 'GI', 'GA', 'ST', 'SN']
    sets = []
    
    info, nets, intervals, amplitudes, rep = get_networks(builder,
                                                          setup.builder(),
                                                          setup.director(),
                                                          setup.default_params())
    key=nets.keys()[0]
    file_name = get_file_name(script_name, nets[key].par)
    file_name_figs = get_file_name_figs(script_name,  nets[key].par)   
    
    path_nest=get_path_nest(script_name, nets.keys(), nets[key].par)
    for net in nets.values():
        net.set_path_nest(path_nest)
    
    d_firing_rate = setup.firing_rate()
    
    kwargs_dic = {'firing_rate':d_firing_rate, 
                  'mean_rate_slices':{'intervals':intervals[1], 
                                      'repetition':rep, 
                                      'x':amplitudes}}
    
    add_perturbations(perturbation_list, nets)
#     sd = get_storage(file_name, info)

    # Adding nets no file name
    sd_list=get_storage_list(nets, file_name, info)

    from_disks, d = main_loop(from_disk, attr, models, 
                              sets, nets, kwargs_dic, sd_list) 
#     from_disks, d = main_loop(from_disk, attr, models, 
#                               sets, nets, kwargs_dic, sd)
    
    return file_name_figs, from_disks, d, models


def create_figs(file_name_figs, from_disks, d, models, setup):
    sd_figs = Storage_dic.load(file_name_figs)

    d_plot_fr = setup.plot_fr()
    d_plot_mr = setup.plot_mr()
    d_plot_mr2 = setup.plot_mr2()
    figs = []
    
    pp(setup.plot_mr_general().get('fig_and_axes'))
    fig, axs=ps.get_figure2(**setup.plot_mr_general().get('fig_and_axes'))
    figs.append(fig)
    for ax in axs:
        ax.tick_params(direction='out',
                       length=2,
                       width=0.5,
                       pad=0.01,
                        top=False, right=False
                        )
    
    show_mr(d, ['M1', 'M2'], axs[2:4], **d_plot_mr2)


    for i, s, c0, c1, rotation in [[2, 'Rel. inh. effect', -0.31, 0., 90],
#                                    [2, 'decrease (Hz)', -0.39, 0., 90],
                                   [0, 'Firing rate (Hz)', -0.45, 0., 90],
                                   [0, r'$MSN_{D1}$', -0.27, 0.5, 90],
                                   [1, r'$MSN_{D2}$', -0.27, 0.5, 90],
#                                    [2, r'$MSN_{D1}$', 1.1, 0.5, 270],
#                                    [3, r'$MSN_{D2}$', 1.1, 0.5, 270],
                                   ]:
                                   
        axs[i].text(c0, c1, s, 
                    fontsize=7,
                    transform=axs[i].transAxes,
                    verticalalignment='center', 
                    horizontalalignment='center', 
                    rotation=rotation) 
   
#     ps.shift('left', axs, 0.5, n_rows=len(axs), n_cols=1)
    for ax in axs[2:4]:
        if not ax.legend():
            continue
#         ax.legend(bbox_to_anchor=(2.2, 1))
        ax.legend().set_visible(False)
#         ax.set_ylabel('')
#         ax.set_xlim([1,2])
#     figs.append(show_fr(d, models, **d_plot_fr))
#     figs.append(show_mr(d, models, **d_plot_mr))
#     figs.append(show_mr(d, models, **d_plot_mr2))
    show_mr(d, ['M1', 'M2'],  axs[0:2], **d_plot_mr)
#     axs=figs[-1].get_axes()
#     ps.shift('left', axs, 0.5, n_rows=len(axs), n_cols=1)
     
    axs[0].legend(axs[0].lines[0:6], 
                  d_plot_mr2['labels'][0:6], 
                  bbox_to_anchor=(2.4, 2.4), ncol=2,
                  handletextpad=0.1,
                  frameon=False,
                  columnspacing=0.3,
                  labelspacing=0.2) 
    
    for i, ax in enumerate(axs[1:2]):
        if ax.legend():
            ax.legend().set_visible(False)
#         ax.legend(bbox_to_anchor=(2.2, 1))
#         ax.set_xlim([1,1.5])
        if i==1:
            ax.set_ylim([0,20])
    axs[0].my_remove_axis(xaxis=True, yaxis=False,
                          keep_ticks=False) 
    axs[2].my_remove_axis(xaxis=True, yaxis=False,
                          keep_ticks=False)
    for i, ax in enumerate(axs):
        ax.set_ylabel('')
        ax.my_set_no_ticks(yticks=3, xticks=4)
#         ax.set_xticks([1.1,1.3,1.5])
        if i==0:
            ax.set_yticks([0,10,20])
        if i==1:
            ax.set_yticks([0,7,14])
            ax.set_ylim([0,20])
    
    if len(axs[2].lines)>2:
        axs[2].lines.remove(axs[2].lines[0])
        axs[2].lines.remove(axs[2].lines[-1])
            
    sd_figs.save_figs(figs, format='png', dpi=100)
    sd_figs.save_figs(figs, format='svg', in_folder='svg')

class Main():    
    def __init__(self, **kwargs):
        self.kwargs=kwargs
    
    def __repr__(self):
        return self.kwargs['script_name']

    def do(self):
        main(**self.kwargs)    
    
    def get_nets(self):
        return self.kwargs['setup'].nets_to_run

    def get_script_name(self):
        return self.kwargs['script_name']

    def get_name(self):
        nets='_'.join(self.get_nets()) 
        script_name=self.kwargs['script_name']
        script_name=script_name.split('/')[1].split('_')[0:2]
        script_name='_'.join(script_name)+'_'+nets
        return script_name+'_'+str(self.kwargs['from_disk'])
        

def main(builder=Builder,
         from_disk=2,
         perturbation_list=None,
         script_name=__file__.split('/')[-1][0:-3],
         setup=Setup(**{'local_num_threads':8,
                        'resolution':2,
                        'repetition':7,
                        'lower':1,
                        'upper':3})):
    
    

    v=simulate(builder, from_disk, 
               perturbation_list, script_name,  setup)
    file_name_figs, from_disks, d, models = v
    
    if numpy.all(numpy.array(from_disks) > 0):
        create_figs(file_name_figs, from_disks, d, models, setup)


#   da  if DISPLAY: pylab.show() 
    
    

import unittest
class TestInhibitionStriatum(unittest.TestCase):     
    def setUp(self):
        from core.network.default_params import Perturbation_list as pl
        from_disk=2
        
        import oscillation_perturbations4 as op
        
        rep, res, low, upp=1, 3, 1, 1.5
        
        sim_time=rep*res*1000.0
        size=3000.0
        local_num_threads=16
        
        l=op.get()
        
        p=pl({'simu':{'sim_time':sim_time,
                      'sim_stop':sim_time,
                      'local_num_threads':local_num_threads},
                  'netw':{'size':size}},
                  '=')
        p+=l[7]
        self.setup=Setup(**{'local_num_threads':local_num_threads,
                            'resolution':res,
                            'repetition':rep,
                            'lower':low,
                            'upper':upp})
        
        v=simulate(builder=Builder,
                            from_disk=from_disk,
                            perturbation_list=p,
                            script_name=(__file__.split('/')[-1][0:-3]
                                         +'/data'),
                            setup=self.setup)
        
        file_name_figs, from_disks, d, models= v
        
        
        self.file_name_figs=file_name_figs
        self.from_disks=from_disks
        self.d=d
        self.models=models
        

    def test_create_figs(self):
        create_figs(self.file_name_figs, 
                    self.from_disks, 
                    self.d, 
                    self.models, 
                    self.setup)
        pylab.show()
    
#     def test_show_fr(self):
#         show_fr(self.d, self.models, **{'win':20.,
#                                         't_start':4000.0,
#                                         't_stop':5000.0})
#         pylab.show()
 

if __name__ == '__main__':
    test_classes_to_run=[
                         TestInhibitionStriatum
                         ]
    suites_list = []
    for test_class in test_classes_to_run:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)
    unittest.TextTestRunner(verbosity=2).run(big_suite)
    
    
    
    
