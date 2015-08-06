'''
Created on Jun 4, 2015

@author: mikael
'''

import oscillation_perturbations_new_beginning_slow_fitting_GI as op
from core.network.manager import (add_perturbations,
                                     get_storage_list, 
                                     get_networks,
                                     save, load,)
from scripts_inhibition.base_simulate import (get_file_name,
                      get_path_nest)

from scripts_inhibition.base_oscillation import mallet2008
from core.network.manager import Builder_single as builder

from core.network.default_params import Perturbation_list as pl
from core.network.engine import Network_list
from core.network.optimization import Fmin
import pprint

pp=pprint.pprint


class Setup(object):

    def __init__(self,*a, **k):
        self.nets_to_run=k.get('nets_to_run')
        if self.nets_to_run[0]=='Net_0':
            s='control'
        else:
            s='lesion'
        
        self.tp_name=k.get('tp_name')+'_'+s
        
        self.opt=k.get('opt')
        self.single_unit=k.get('single_unit')

    def builder(self):
        return {}

    def default_params(self):
        d={}
        return d

    def director(self):
        return {'nets_to_run':self.nets_to_run}  
           
    def engine(self):
        d= {'record':['spike_signal', 'voltage_signal'],
            'save_conn':False, 
            'sub_folder':'conn', 
            'verbose':False,
            'display_opt':True}
        return d

    
    def par(self):
        d={'simu':{'print_time':False,

    #                         'start_rev'
             'mm_params':{'to_file':False, 'to_memory':True},
             'sd_params':{'to_file':False, 'to_memory':True}},
        'netw':{'rand_nodes':{'C_m':False, 
                              'V_th':False, 
                              'V_m':False},
                'size':1.,
                'single_unit':self.single_unit,
#                 'optimization':self.opt[net],
                },
        'node':{self.single_unit:{'mm':{'active':True}},
    #                             'M2p':{'lesion':True}
                }} 
        return d 
def target_perturbations():
    d=mallet2008()
    out={}
    ll=[d['all']['activation']['control']['rate'], 
        d['STN']['activation']['control']['rate'],
        d['TI']['activation']['lesioned']['rate'],
        d['TA']['activation']['lesioned']['rate'], 
        d['STN']['activation']['lesioned']['rate'],
        d['all']['slow_wave']['control']['rate'], 
        d['STN']['slow_wave']['control']['rate'],
        d['TI']['slow_wave']['lesioned']['rate'],
        d['TA']['slow_wave']['lesioned']['rate'], 
        d['STN']['slow_wave']['lesioned']['rate']]
    
    d={'node':{'GI':{'rate':ll[0]},
               'GA':{'rate':ll[3]},
               'ST':{'rate':ll[1]}}}
    out['beta_control']=d
    
    d={'node':{'GI':{'rate':ll[2]},
               'GA':{'rate':ll[3]},
               'ST':{'rate':ll[4]}}}
    out['beta_lesion']=d
    
    d={'node':{'GI':{'rate':ll[5]},
               'ST':{'rate':ll[6]}}}
    out['sw_control']=d
    
    d={'node':{'GI':{'rate':ll[7]},
               'GA':{'rate':ll[8]},
               'ST':{'rate':ll[9]}}}
    out['sw_lesion']=d    
    
    for d in out.values():
        for model in d['node'].keys():
            d['node'][model+'p']=d['node'][model].copy()
    
    return out

def opt_setups():
    out={}
    d={'f':['GI', 'ST'],
       'x':['node.EI.rate', 'node.CS.rate'],
       'x0':[1700, 230]}
    out['beta_control']=d
    
    d={'f':['GI', 'ST'],
       'x':['node.EI.rate','node.CS.rate'],
       'x0':[1700, 230]}
    out['beta_lesion']=d
    
    d={'f':['GI', 'ST'],
       'x':['node.EI.rate', 'node.CS.rate'],
       'x0':[1000, 180]}
    out['sw_control']=d
    
    d={'f':['GI', 'GA','ST'],
       'x':['node.EI.rate','node.EA.rate', 'node.CS.rate'],
       'x0':[800, 100, 180]}
    out['sw_lesion']=d
    
    return out  

def main(*args, **kwargs):
    
    setup=kwargs.get('setup')#, Setup())
    script_name=kwargs.get('script_name')#,__file__.split('/')[-1][0:-3])
    builder=kwargs.get('builder')
    
    info, nets, _ = get_networks(builder, 
                                 setup.builder(), 
                                 setup.director(),
                                 setup.engine(),
                                 setup.default_params()) 
    perturbation_list=kwargs.get('perturbation_list')
    
    perturbation_list+=pl(target_perturbations()[setup.tp_name], '=', **{'name':setup.tp_name})
    perturbation_list+=pl(setup.par(), '=', **{'name':setup.tp_name})
    add_perturbations(perturbation_list, nets)

    key=nets.keys()[0]
    file_name = get_file_name(script_name, nets[key].par)
    path_nest=get_path_nest(script_name, nets.keys(), nets[key].par)
    
    for val in nets.values():
        for key in sorted(val.par['nest'].keys()):
            val2=val.par['nest'][key]
            if 'weight' in val2.keys():
                print key, val2['weight']
        for key in sorted(val.par['node'].keys()):
            val2=val.par['node'][key]
            if 'spike_setup' in val2.keys():
                print key, val2['spike_setup']
            if 'rate' in val2.keys():
                print key, val2['rate']
                
    print setup.tp_name, nets.keys()
    pp(nets['Net_1'].par['nest']['GA_GA_gaba']['weight'])
    for net in nets.values():
        net.set_path_nest(path_nest) 
        pp(nets['Net_1'].par['nest']['GA_GA_gaba']['weight'])
        net.set_opt(setup.opt[net.name])
    pp(nets['Net_1'].par['nest']['GA_GA_gaba']['weight'])
    
    # vt=net.voltage_trace()
    # import pylab
    # ax=pylab.subplot(111)
    # vt.plot_voltage_trace(ax)
    # pylab.show()

    sd_list=get_storage_list(nets.keys(), file_name, info)
    
    for net, sd in zip(nets.values(), sd_list):    
        nl=Network_list([net])
#         pp(net.par.dic['node']['STp'])
        kwargs={'model':nl,
                'call_get_x0':'get_x0',
                'call_get_error':'sim_optimization', 
                'verbose':True,}
        
        kwargs_fmin={'maxiter':100, 
                     'maxfun':100,
                     'full_output':1,
                     'retall':1,
                     'xtol':.0001,
                     'ftol':.0001,
                     'disp':0}
        kwargs_fmin={}
        kwargs['kwargs_fmin']=kwargs_fmin
        f=Fmin('GA_opt', **kwargs)
        h=f.fmin() 
        dd={setup.tp_name.split('_')[0]:{net.name:{'opt':h}}}
        
        save(sd, dd)
        
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

import unittest
class TestMain(unittest.TestCase):
    
    def setUp(self):
        pass
    def test_do(self):
        
        d={'simu':{'sim_stop':20000.0,
                   'sim_time':20000.0,
                   'stop_rec':20000.0,
                   'start_rec':1000.0}}
        per=pl(d, '=')
        per+=op.get()[0]
        kw_setup={'nets_to_run':['Net_1'],
#                   'single_unit':'GA',
#                   'opt':{'Net_1':{'f':['GA'],
#                                   'x':['nest.ST_GA_ampa.weight'],
#                                   'x0':[0.45]}},
                  'single_unit':'GI',
                  'opt':{'Net_1':{'f':['GI'],
                                  'x':['node.EIp.rate'],
                                  'x0':[1000.0]}},
                  'tp_name':'beta'}
        kwargs={'builder':builder,
                'perturbation_list':per,
                'script_name':__file__.split('/')[-1][0:-3],
                'setup':Setup(**kw_setup)
                }
        obj=Main(**kwargs)
        obj.do()
    
        

if __name__ == '__main__':
    d={
       TestMain:[
                        'test_do',
                           ]}


    test_classes_to_run=d
    suite = unittest.TestSuite()
    for test_class, val in  test_classes_to_run.items():
        for test in val:
            suite.addTest(test_class(test))

    unittest.TextTestRunner(verbosity=2).run(suite)
        
        