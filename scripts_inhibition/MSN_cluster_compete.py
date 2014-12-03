'''
Created on May 14, 2014

@author: mikael
'''

import numpy
import os
from os.path import expanduser
from simulate import (main_loop,main_loop_conn, show_fr, show_mr, show_mr_diff,
                      get_file_name, get_file_name_figs)

from toolbox import misc, pylab
from toolbox.data_to_disk import Storage_dic
from toolbox.my_signals import Data_generic
from toolbox.network import manager
from toolbox.network.manager import (add_perturbations,
                                    get_storage_list)
import toolbox.plot_settings as ps
from toolbox.network.manager import Builder_MSN_cluster_compete as Builder



import pprint
pp=pprint.pprint
    
DISPLAY=os.environ.get('DISPLAY')



def cmp_mean_rate_diff(d, models, parings, x, _set='set_0'):
    
    for model in models:
        y=[]
        y_std=[]
        for net0, net1 in parings:
            d0=d[net0][_set]
            d1=d[net1][_set]
   
            v0=d0[model]['mean_rate_slices']
            v1=d1[model]['mean_rate_slices']
            
            y.append(numpy.mean(numpy.abs(v0.y_raw_data-v1.y_raw_data)))
            y_std.append(numpy.std(numpy.abs(v0.y_raw_data-v1.y_raw_data)))
        
        dd={'y':numpy.array(y),
            'y_std':numpy.array(y_std),
            'x':numpy.array(x)}
        
        obj=Data_generic(**dd)
        
        d=misc.dict_recursive_add(d, ['Difference',
                                  model,
                                 'mean_rate_diff'], obj)
    pp(d)
    return d    
        
      
        
def get_kwargs_builder(**k_in):
    rep=k_in.get('repetition', 5)
    return {'print_time':False, 
            'local_num_threads':10, 
            'save_conn':{'active':False,
                         'overwrite':False},
            'sim_time':rep*1000.0, 
            'sim_stop':rep*1000.0, 
            'size':3000.0, 
            'start_rec':0.0, 
            'sub_sampling':1, 
            'repetition': rep }

def get_kwargs_engine():
    return {'verbose':True}

def get_networks(builder, **k_in):
    info, nets, builder=manager.get_networks(builder,
                                             get_kwargs_builder(**k_in),
                                             get_kwargs_engine())
    
    intervals=builder.dic['intervals']
    rep=builder.dic['repetition']
    x=builder.dic['percents']
    
    return info, nets, intervals, rep, x
    
def gs_builder(*args, **kwargs):
    import matplotlib.gridspec as gridspec
    n_rows=kwargs.get('n_rows',2)
    n_cols=kwargs.get('n_cols',3)
    order=kwargs.get('order', 'col')
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=kwargs.get('wspace', 0. ), 
              hspace=kwargs.get('hspace', 0.2 ))

    iterator = [[slice(5,10),slice(1,5)],
                [slice(10,15),slice(1,5)],
                [slice(5,10),slice(6,10)],
                [slice(10,15),slice(6,10)],
               ]
    return iterator, gs
    
class Setup(object):

    def __init__(self, **k):
        self.local_num_threads=k.get('local_num_threads',1)
        self.nets_to_run=k.get('nets_to_run',[])
        self.rep=k.get('repetition',2)
        
                
    def builder(self):
        d= {'repetition':self.rep}
        return d

    def firing_rate(self):
        d={'average':False, 
           'sets':[0,1],
           'time_bin':5,
           'local_num_threads':self.local_num_threads}
        return d

    def plot_fr(self):
        labels=['Unspec active {}%'.format(int(100/v)) 
                for v in [5, 10, 20, 40, 80]]
        labels+=['Spec cluster size {}%'.format(int(100/v)) 
                 for v in [5, 10, 20, 40, 80]]
 
        
        d={'win':10.,
           'by_sets':False,
           'labels':labels}
        return d



    def plot_fig_axs(self):
        d={'fig_and_axes':{'n_rows':17, 
                        'n_cols':10, 
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

    def plot_fr2(self):
        labels=['Unspecific {}%'.format(int(100/v)) 
                for v in [10]]
        labels+=['Specific {}%'.format(int(100/v)) 
                 for v in [10]]
        
        d={'win':10.,
           't_stop':1500.0,
           'by_sets':False,
           'labels':labels,

           'nets':['Net_1_set_0', 'Net_6_set_0'],
#            'fig_and_axes':{'n_rows':2, 
#                                         'n_cols':1, 
#                                         'w':350.0, 
#                                         'h':500.0, 
#                                         'fontsize':16}
              'fig_and_axes':{'n_rows':2, 
                                'n_cols':1, 
                        'w':72*8.5/2.54, 
                        'h':150,  
                                'fontsize':7,
                                'legend_fontsize':7,
                                'frame_hight_x':0.6,
                                'linewidth':1.}}
        return d

    def plot_mr_diff(self):
     
        d={
#            'fig_and_axes':{'n_rows':2, 
#                                         'n_cols':1, 
#                                         'w':350.0, 
#                                         'h':500.0, 
#                                         'fontsize':16}
           'colors':['k'],
#                                  'fig_and_axes':{'n_rows':2, 
#                                         'n_cols':1, 
#                                         'w':350.0*0.65*2, 
#                                         'h':500.0*0.65*2, 
#                                         'fontsize':12*2,
#                                         'legend_fontsize':9*2,
#                                         'frame_hight_x':0.6,
#                                         'linewidth':4.}
           }
        return d
    
    def plot_mr(self):
        labels=['Unspec {}%'.format(int(100/v)) 
                for v in [5, 10, 20, 40, 80]]
        labels+=['Spec {}%'.format(int(100/v)) 
                 for v in [5, 10, 20, 40, 80]]
        
        d={'win':10.,
           'by_sets':True,
           'labels':labels}
        return d


def simulate(builder, from_disk, perturbation_list, script_name, setup):
    home = expanduser("~")
    
#     file_name = get_file_name(script_name, home)
#     file_name_figs = get_file_name_figs(script_name, home)
#     
    d_firing_rate = setup.firing_rate()
    
    attr = ['firing_rate', 'mean_rate_slices']
    attr_conn=['conn_matrix']
    models=['M1', 'M2']
    models_conn=['M1_M1_gaba', 'M1_M2_gaba',
                 'M2_M1_gaba', 'M2_M2_gaba']
    sets = ['set_0']
    
    info, nets, intervals, rep, x = get_networks(builder, 
                                                  **setup.builder())
    
    key=nets.keys()[0]
    file_name = get_file_name(script_name, nets[key].par)
    file_name_figs = get_file_name_figs(script_name,  nets[key].par)   
    
    kwargs_dic = {'firing_rate':d_firing_rate, 
                  'mean_rate_slices': {'intervals':intervals[1], 
                                       'repetition':rep, 
                                       'set_0':{'x':x}, 
                                       'sets':[0]}}

#     add_perturbations(perturbation_list, nets)    
#     sd = get_storage(file_name, info)
# 
# #     _, d1 = main_loop_conn(from_disk, attr_conn, models_conn, 
# #                               sets, nets, kwargs_dic, sd)     
    d1={}
#     from_disks, d2 = main_loop(from_disk, attr, models, 
#                               sets, nets, kwargs_dic, sd)
#     
    add_perturbations(perturbation_list, nets)

    # Adding nets no file name
    sd_list=get_storage_list(nets, file_name, info)

    from_disks, d2 = main_loop(from_disk, attr, models, 
                              sets, nets, kwargs_dic, sd_list) 
    
    d=misc.dict_update(d1,d2)
    if from_disk==2:
        d=cmp_mean_rate_diff(d, models, [['Net_0', 'Net_5'],
                                         ['Net_1', 'Net_6'],
                                         ['Net_2', 'Net_7'],
                                         ['Net_3', 'Net_8'],
                                         ['Net_4', 'Net_9']], x)

    return file_name, file_name_figs, from_disks, d, models


def create_figs(setup, file_name_figs, d, models):
    
    sd_figs = Storage_dic.load(file_name_figs)
    figs = []
    
#     d_plot_fr = setup.plot_fr()
    d_plot_fr2=setup.plot_fr2()
#     d_plot_mr=setup.plot_mr()
    d_plot_mr_diff=setup.plot_mr_diff()


    fig, axs=ps.get_figure2(**setup.plot_fig_axs().get('fig_and_axes'))
    figs.append(fig)
    for ax in axs:
        ax.tick_params(direction='out',
                       length=2,
                       width=0.5,
                       pad=0.01,
                        top=False, right=False
                        )
#     for name in sorted(d.keys()):
#         if name=='Difference':
#             continue        
#         for model in d[name]['set_0'].keys():
#             v={'firing_rate':d[name]['set_0'][model]['firing_rate']}
#             d['Net_0'][model]=v
    
#     figs.append(show_fr(d, models, **d_plot_fr))
    show_fr(d, models, axs[0:2], **d_plot_fr2)
    axs=figs[-1].get_axes()
    for i, ax in enumerate(axs[2:4]):
        if i==0:
            ax.set_ylim([0,70])
        else: 
            ax.set_ylim([0,50])
        ax.my_set_no_ticks(xticks=4)
    
    show_mr_diff(d, models, axs[2:4], **d_plot_mr_diff)
#     axs=figs[-1].get_axes()
    for i, ax in enumerate(axs[2:4]):
        handles, labels=ax.get_legend_handles_labels() 
        ax.legend(handles, labels,loc='upper left')

    for i, s, c0, c1, rotation in [
#                                    [2, 'Ralative', -0.58, 0., 90],
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
        
    for i, ax in enumerate(axs[1:4]):
        ax.legend().set_visible(False)
        
    axs[0].legend(axs[0].lines[:], 
                  d_plot_fr2['labels'][:], 
                  bbox_to_anchor=(1.2, 1.8), ncol=1,
                  handletextpad=0.1,
                  frameon=False,
                  columnspacing=0.3,
                  labelspacing=0.2) 

    axs[0].my_remove_axis(xaxis=True, yaxis=False,
                          keep_ticks=False) 
    axs[2].my_remove_axis(xaxis=True, yaxis=False,
                          keep_ticks=False)
    
    for i, ax in enumerate(axs):
        ax.set_ylabel('')
        ax.my_set_no_ticks(yticks=3, xticks=3)    
        if i==1:
            ax.set_xticks([0, 500, 1000])
            ax.set_ylim([0,40])
        if i==2:
            ax.set_title('Difference')
#             ax.set_yticks([0, 1000.0])
            ax.set_ylim([0,25])
        if i==3:
#             ax.set_yticks([0, 1000.0])
            ax.set_ylim([0,20])
                        
    sd_figs.save_figs(figs, format='png', dpi=400)
    sd_figs.save_figs(figs, format='svg')


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
         setup=Setup(**{'local_num_threads':4,
                        'repetition':5})):
    
    
    v=simulate(builder, from_disk, perturbation_list, script_name, setup)
    file_name, file_name_figs, from_disks, d, models = v
    
    if numpy.all(numpy.array(from_disks) > 0):
        create_figs(setup, file_name_figs, d, models)
    
    
#     pylab.show()
 
import unittest
class TestMethods(unittest.TestCase):     
    def setUp(self):
        from toolbox.network.default_params import Perturbation_list as pl
        from_disk=2
        
        import oscillation_perturbations4 as op
        
        rep=2
        
        sim_time=rep*1000.0
      
        local_num_threads=20
        
        l=op.get()
        
        p=pl({'simu':{'sim_time':sim_time,
                      'sim_stop':sim_time,
                      'local_num_threads':local_num_threads}},
                  '=')
        p+=l[7]
        self.setup=Setup(**{'local_num_threads':local_num_threads,
                        'repetition':rep})
        v=simulate(builder=Builder,
                            from_disk=from_disk,
                            perturbation_list=p,
                            script_name=(__file__.split('/')[-1][0:-3]
                                         +'/data'),
                            setup=self.setup)
        
        file_name, file_name_figs, from_disks, d, models= v
        

        self.file_name_figs=file_name_figs
        self.from_disks=from_disks
        self.d=d
        self.models=models
        

    def test_create_figs(self):
        create_figs(
                    self.setup,
                    self.file_name_figs, 
                    self.d, 
                    self.models)
        pylab.show()
    
#     def test_show_fr(self):
#         show_fr(self.d, self.models, **{'win':20.,
#                                         't_start':4000.0,
#                                         't_stop':5000.0})
#         pylab.show()
 

if __name__ == '__main__':
    test_classes_to_run=[
                         TestMethods
                         ]
    suites_list = []
    for test_class in test_classes_to_run:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)
    unittest.TextTestRunner(verbosity=2).run(big_suite)

