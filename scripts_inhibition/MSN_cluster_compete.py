'''
Created on May 14, 2014

@author: mikael
'''

import numpy
import os
from os.path import expanduser
from simulate import (main_loop, show_fr, show_mr, show_mr_diff,
                      get_file_name, get_file_name_figs)

from toolbox import misc, pylab
from toolbox.data_to_disk import Storage_dic
from toolbox.my_signals import Data_generic
from toolbox.network import manager
from toolbox.network.manager import (add_perturbations,
                                    get_storage)

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
            'threads':10, 
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
    
    
class Setup(object):

    def __init__(self, **k):
        self.threads=k.get('threads',1)
        self.rep=k.get('repetition',2)
                
    def builder(self):
        d= {'repetition':self.rep}
        return d

    def firing_rate(self):
        d={'average':False, 
           'sets':[0,1],
           'time_bin':5,
           'threads':self.threads}
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

    def plot_mr(self):
        labels=['Unspec active {}%'.format(int(100/v)) 
                for v in [5, 10, 20, 40, 80]]
        labels+=['Spec cluster size {}%'.format(int(100/v)) 
                 for v in [5, 10, 20, 40, 80]]
        
        d={'win':10.,
           'by_sets':True,
           'labels':labels}
        return d

def simulate(builder, from_disk, perturbation_list, script_name, setup):
    home = expanduser("~")
    
    file_name = get_file_name(script_name, home)
    file_name_figs = get_file_name_figs(script_name, home)
    
    d_firing_rate = setup.firing_rate()
    
    attr = ['firing_rate', 'mean_rate_slices']
    models=['M1', 'M2']
    sets = ['set_0']
    
    info, nets, intervals, rep, x = get_networks(builder, 
                                                  **setup.builder())
       
    kwargs_dic = {'firing_rate':d_firing_rate, 
                  'mean_rate_slices': {'intervals':intervals[1], 
                                       'repetition':rep, 
                                       'set_0':{'x':x}, 
                                       'sets':[0]}}

    add_perturbations(perturbation_list, nets)    
    sd = get_storage(file_name, info)
     
    from_disks, d = main_loop(from_disk, attr, models, 
                              sets, nets, kwargs_dic, sd)
       
    d=cmp_mean_rate_diff(d, models, [['Net_0', 'Net_5'],
                                     ['Net_1', 'Net_6'],
                                     ['Net_2', 'Net_7'],
                                     ['Net_3', 'Net_8'],
                                     ['Net_4', 'Net_9']], x)

    return file_name, file_name_figs, from_disks, d, models


def create_figs(setup, file_name_figs, d, models):
    
    sd_figs = Storage_dic.load(file_name_figs)
    figs = []
    
    d_plot_fr = setup.plot_fr()
    d_plot_mr=setup.plot_mr()

#     for name in sorted(d.keys()):
#         if name=='Difference':
#             continue        
#         for model in d[name]['set_0'].keys():
#             v={'firing_rate':d[name]['set_0'][model]['firing_rate']}
#             d['Net_0'][model]=v
    
    figs.append(show_fr(d, models, **d_plot_fr))
    figs.append(show_mr_diff(d, models))
    figs.append(show_mr(d, models, **d_plot_mr))
    
    sd_figs.save_figs(figs, format='png')

def main(builder=Builder,
         from_disk=2,
         perturbation_list=None,
         script_name=__file__.split('/')[-1][0:-3],
         setup=Setup(**{'threads':4,
                        'resolution':5,
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
        
        import oscillation_perturbations as op
        
        rep=2
        
        sim_time=rep*1000.0
      
        threads=4
        
        l=op.get()
        
        p=pl({'simu':{'sim_time':sim_time,
                      'sim_stop':sim_time,
                      'threads':threads}},
                  '=')
        p+=l[1]
        self.setup=Setup(**{'threads':threads,
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

