'''
Created on Jun 27, 2013

@author: lindahlm
'''
import numpy
import os

from os.path import expanduser
from simulate import (main_loop, show_fr, show_mr, 
                      get_file_name, get_file_name_figs)
from toolbox import pylab
from toolbox.data_to_disk import Storage_dic
from toolbox.network import manager
from toolbox.network.manager import (add_perturbations,
                                      get_storage)
from toolbox.network.manager import Builder_inhibition_striatum as Builder
import toolbox.plot_settings as ps
import pprint
pp=pprint.pprint
    
DISPLAY=os.environ.get('DISPLAY')

def get_kwargs_builder(**k_in):
    return {'print_time':False, 
            'threads':8, 
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

def get_networks(builder, **k_in):
    info, nets, builder=manager.get_networks(builder,
                                             get_kwargs_builder(**k_in),
                                             get_kwargs_engine())
    
    intervals=builder.dic['intervals']
    rates=builder.dic['amplitudes']
    rep=builder.dic['repetitions']
    
    return info, nets, intervals, rates, rep

class Setup(object):
    
    def __init__(self, **k):
        self.threads=k.get('threads',1)
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
    
    def firing_rate(self):
        d={'average':False, 
           'threads':self.threads, 
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

    def plot_mr(self):
        d={'labels':['All', 
                     'Only MSN-MSN',
                     'Only FSN-MSN',
                     'Only FSN-MSN-static',
                     'Only GPe TA-MSN',
                     'No inhibition'],
                      'fig_and_axes':{'n_rows':2, 
                                        'n_cols':1, 
                                        'w':600.0*0.65*2, 
                                        'h':500.0*0.65*2, 
                                        'fontsize':12*2,
                                        'legend_fontsize':9*2,
                                        'linewidth':4.} ,
                      'x_lim':[1,1.5],
                      'y_lim':[0,30]}
        return d
        
    def plot_mr2(self):
        d={'labels':['All', 
                     'Only MSN-MSN',
                     'Only FSN-MSN',
                     'Only FSN-MSN-static',
                     'Only GPe TA-MSN',
                     'No inhibition'],
           'fontsize':16,
           'relative':True,
           'relative_to':[5,0],
           'x_lim':[1,1.5],
           'y_lim':[0,1],
           'delete':[0,5],
           'fig_and_axes':{'n_rows':2, 
                                        'n_cols':1, 
                                        'w':600.0*0.65*2, 
                                        'h':500.0*0.65*2, 
                                        'fontsize':12*2,
                                        'legend_fontsize':9*2,
                                        'linewidth':4.} }
        return d
    


def simulate(builder, 
             from_disk, 
             perturbation_list, 
             script_name, 
             setup):
    home = expanduser("~")
    file_name = get_file_name(script_name, home)
    file_name_figs = get_file_name_figs(script_name, home)
    
    attr = ['firing_rate', 'mean_rate_slices']
    models = ['M1', 'M2', 'FS', 'GI', 'GA', 'ST', 'SN']
    sets = []
    
    info, nets, intervals, amplitudes, rep = get_networks(builder,
                                                          **setup.builder())
    
    d_firing_rate = setup.firing_rate()
    
    kwargs_dic = {'firing_rate':d_firing_rate, 
                  'mean_rate_slices':{'intervals':intervals[1], 
                                      'repetition':rep, 
                                      'x':amplitudes}}
    
    add_perturbations(perturbation_list, nets)
    sd = get_storage(file_name, info)
    
    from_disks, d = main_loop(from_disk, attr, models, 
                              sets, nets, kwargs_dic, sd)
    
    return file_name_figs, from_disks, d, models


def create_figs(file_name_figs, from_disks, d, models, setup):
    sd_figs = Storage_dic.load(file_name_figs)

    d_plot_fr = setup.plot_fr()
    d_plot_mr = setup.plot_mr()
    d_plot_mr2 = setup.plot_mr2()
    figs = []
    figs.append(show_mr(d, ['M1', 'M2'], **d_plot_mr2))
    axs=figs[-1].get_axes()
    ps.shift('left', axs, 0.5, n_rows=len(axs), n_cols=1)
    for ax in axs:
        ax.legend(bbox_to_anchor=(2.2, 1))
        ax.set_ylabel('Rel decrease (Hz)')
#         ax.set_xlim([1,2])
#     figs.append(show_fr(d, models, **d_plot_fr))
#     figs.append(show_mr(d, models, **d_plot_mr))
#     figs.append(show_mr(d, models, **d_plot_mr2))
    figs.append(show_mr(d, ['M1', 'M2'], **d_plot_mr))
    axs=figs[-1].get_axes()
    ps.shift('left', axs, 0.5, n_rows=len(axs), n_cols=1)
    for i, ax in enumerate(axs):
        ax.legend(bbox_to_anchor=(2.2, 1))
#         ax.set_xlim([1,1.5])
        if i==1:
            ax.set_ylim([0,20])
    sd_figs.save_figs(figs, format='png')
    sd_figs.save_figs(figs, format='svg')
    
def main(builder=Builder,
         from_disk=2,
         perturbation_list=None,
         script_name=__file__.split('/')[-1][0:-3],
         setup=Setup(**{'threads':8,
                        'resolution':2,
                        'repetition':7,
                        'lower':1,
                        'upper':3})):
    
    

    v=simulate(builder, from_disk, 
               perturbation_list, script_name,  setup)
    file_name_figs, from_disks, d, models = v
    
    if numpy.all(numpy.array(from_disks) > 0):
        create_figs(file_name_figs, from_disks, d, models, setup)


#     if DISPLAY: pylab.show() 
    
    

import unittest
class TestInhibitionStriatum(unittest.TestCase):     
    def setUp(self):
        from toolbox.network.default_params import Perturbation_list as pl
        from_disk=1
        
        import oscillation_perturbations4 as op
        
        rep, res, low, upp=1, 3, 1, 1.5
        
        sim_time=rep*res*1000.0
        size=3000.0
        threads=16
        
        l=op.get()
        
        p=pl({'simu':{'sim_time':sim_time,
                      'sim_stop':sim_time,
                      'threads':threads},
                  'netw':{'size':size}},
                  '=')
        p+=l[7]
        self.setup=Setup(**{'threads':threads,
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
    
    
    
    
