'''
Created on Apr 18, 2014

@author: lindahlm
'''
import itertools
import numpy
import os
import unittest

from toolbox.data_to_disk import Storage_dic
from toolbox.network.data_processing import (dummy_data_du, Data_unit_spk,
                                             Data_unit_base, Data_units_relation)
from toolbox.network.default_params import Perturbation_list as pl
from toolbox.network.default_params import (Beta,
                                            Go_NoGo_compete,
                                            Inhibition, 
                                            Inhibition_striatum,
                                            MSN_cluster_compete,
                                            Par_base, 
                                            Single_unit, 
                                            Slow_wave)

from toolbox.network.engine import Network, Network_base
from toolbox import misc

import pprint
pp=pprint.pprint

from os.path import expanduser
HOME = expanduser("~")


class Director(object):
    
    __builder=None
    def set_builder(self, builder):
        self.builder=builder
        
    def get_networks(self, **kwargs):
        
        per=self.builder.get_perturbations()
        
        nets={}
        info=''
        for i, p in enumerate(per):
            name='Net_'+str(i)
            info+='Net_'+str(i)+':'+ p.name+'\n'
            par=self.builder.get_parameters(p)
            net=self.builder.get_network(name, par, **kwargs)
            nets[net.get_name()]=net
        return info, nets

def set_networks_names(names, nets):
    for name, net in zip(names, nets):
        net.set_name(name)

class Mixin_dopamine(object):
    def _dop(self):
        d={'netw':{'tata_dop':0.8}}
        return pl(d, '=', **{'name':'dop'})
    
    def _no_dop(self):
        d={'netw':{'tata_dop':0.0}}
        return pl(d, '=', **{'name':'no_dop'})
    

class Mixin_reversal_potential_striatum(object):
    def _low(self):
        d={'node':{'M1':{'model':'M1_low'},
                   'M2':{'model':'M2_low'},
                   'FS':{'model':'FS_low'}}}
        return  pl(d, '=', **{'name':'low'})
        
              
    def _high(self):
        d={'node':{'M1':{'model':'M1_high'},
                   'M2':{'model':'M2_high'},
                   'FS':{'model':'FS_high'}}}
        return pl(d,'=', **{'name':'high'})

class Mixin_reversal_potential_FS(object):
    def _low(self):
        d={'node':{'FS':{'model':'FS_low'}}}
        return  pl(d, '=', **{'name':'low'})
                  
    def _high(self):
        d={'node':{'FS':{'model':'FS_high'}}}
        return pl(d,'=', **{'name':'high'})

class Mixin_reversal_potential_M1(object):
    def _low(self):
        d={'node':{'M1':{'model':'M1_low'}}}
        return  pl(d, '=', **{'name':'low'})
                  
    def _high(self):
        d={'node':{'M1':{'model':'M1_high'}}}
        return pl(d,'=', **{'name':'high'})

 
class Mixin_reversal_potential_M2(object):
    def _low(self):
        d={'node':{'M2':{'model':'M2_low'}}}
        return  pl(d, '=', **{'name':'low'})
                  
    def _high(self):
        d={'node':{'M2':{'model':'M2_high'}}}
        return pl(d,'=', **{'name':'high'})


class Mixin_general_network(object):
    def _general(self):
        k=self.kwargs
        d={'simu':get_simu(k),
           'netw':{
                   'size':k.get('size',500.0),
                   'sub_sampling':{'M1':k.get('sub_sampling',1),
                                   'M2':k.get('sub_sampling',1)},
                   'rand_nodes':k.get('rand_nodes',{'C_m':True, 
                                                    'V_th':True,
                                                    'V_m':True})}}
        return pl(d,'=')
    
class Mixin_general_single(object):
    def _general(self):
        k=self.kwargs
        su=self.kwargs.get('single_unit', 'FS')
        sui=self.kwargs.get('single_unit_input', 'CFp')
        
        d={'simu':get_simu(k),
           'netw':{'size':k.get('size',9.0),
                   'rand_nodes':k.get('rand_nodes',{'C_m':True, 
                                                    'V_th':True,
                                                    'V_m':True}),
                   'single_unit':su,
                   'single_unit_input':sui},
           'node':{su:{'mm':{'active':self.kwargs.get('mm', False)}, 
                       'sd':{'active':True}, 
                       'n_sets':1}}}
        
        for inp in self.kwargs.get('inputs'):
            d['node'][inp] = {'lesion':self.kwargs.get('lesion', False)}     
        
        return pl(d,'=')            

class Builder_abstract(object):
    def __init__(self, **kwargs):
        self.kwargs=kwargs
        self.dic={}

    def _variable(self):
        
        l=[]
        l+=[pl(**{'name':'no_pert'})]
#         l+=[pl({'netw':{'size':val}}, '*', **{'name':'Size-'+str(val)}) 
#             for val in [0.5, 1.0, 1.5]] 
#         l+=[pl({'node':{'C2':{'rate':val}}}, '*', **{'name':'M2r-'+str(val)}) 
#             for val in [1.3, 1.2, 1.1, 0.9, 0.8]] 
#         l+=[pl({'node':{'EA':{'rate':val}}}, '*', **{'name':'GAr-'+str(val)}) 
#             for val in [0.8, 0.6, 0.4, 0.2]] 
#         l+=[pl({'nest':{'GI_ST_gaba':{'delay':0.5}}}, '*', 
#                **{'name':'GISTd-0.5'})]    # from GPe type I to STN  
#         l+=[pl({'nest':{'ST_GA_ampa':{'delay':0.5}}}, '*',
#                **{'name':'STGId-0.5'})]     # from STN to GPe type I and A  
#         l+=[pl({'nest':{'GI_ST_gaba':{'delay':0.5}, 
#                         'ST_GA_ampa':{'delay':0.5}}}, '*', 
#                **{'name':'Bothd-0.5'})]
#         l+=[pl({'netw':{'V_th_sigma':0.5}}, '*', **{'name':'V_th-0.5'})]
#         l+=[pl({'netw':{'prop_fan_in_GPE_A':val}}, '*', 
#                **{'name':'GAfan-'+str(val)}) 
#             for val in [2, 4, 6]]
#         l+=[pl('GArV-0.5', [['netw.V_th_sigma',0.5, '*'], 
#                             ['nest.GI_ST_gaba.delay', 0.5, '*']])]
        
        return l

    def _get_general(self):
        return [self._general()]
    
    def _get_striatal_reversal_potentials(self):
        return [self._low(), self._high()]

    def _get_dopamine_levels(self):
        return [self._dop(), self._no_dop()]
    
    def _get_variable(self):
        return self._variable()
     
    def get_perturbations(self):
        l=[]
        l.append(self._get_general())        
        if hasattr(self, '_low') and hasattr(self, '_high'):     
            l.append(self._get_striatal_reversal_potentials())
        l.append(self._get_dopamine_levels())
        l.append(self._get_variable())

        # Create combinations
        comb=list(itertools.product(*l))
        comb=[sum(c) for c in comb]
        return comb        
        
    def get_parameters(self, per):
        raise NotImplementedError     
    
    def get_network(self, name, par, **kwargs):
        return Network(name, **{'verbose':kwargs.get('verbose',True), 
                                'par':par})       
          
class Builder_network_base(Builder_abstract):    
    
    def _variable(self):
        
        l=[]
        l+=[pl(**{'name':'no_pert'})]
        return l
    
    def _get_striatal_reversal_potentials(self):
        return [self._low()]    

    def get_parameters(self, per):
        return Inhibition(**{'perturbations':per})
     
class Builder_network(Builder_network_base, 
                      Mixin_dopamine, 
                      Mixin_general_network, 
                      Mixin_reversal_potential_striatum):
    pass


class Builder_MSN_cluster_compete_base(Builder_abstract):    
    
    def _variable(self):
        
        n_sets0=[5, 10, 20, 40, 80]
        n_sets=n_sets0*2
        rules=['all-all']*5+['set-not_set']*5
        
        durations=[500., 500.]
        amplitudes=[1.,1.75]
        rep=5
        
        l=[]
        for n,rule in zip(n_sets, rules):
            d={}
            for node in ['M1', 'M2']:
                d=misc.dict_update(d, {'node':{node:{'n_sets':n}}})
                
            for name in ['M1_M1_gaba', 'M1_M2_gaba', 'M2_M1_gaba', 'M2_M2_gaba']: 
                d=misc.dict_update(d, {'conn':{name:{'rule':rule}}})
           
            for inp in ['C1', 'C2']:
                d=misc.dict_update(d, {'node':{inp:{'n_sets':n}}})        
                
                params={'type':'burst3',
                        'params':{'n_set_pre':n,
                                  'repetitions':rep},
                }
                
                d_sets={}

                d_sets.update({str(0):{'active':True,
                                          'amplitudes':amplitudes,
                                  'durations':durations,
                                  'proportion_connected':1,
                                  }})
                params['params'].update({'params_sets':d_sets})
   
                d=misc.dict_update(d, {'netw':{'input':{inp:params}}})
#             pp(d)
            l+=[pl(d, '=', **{'name':'n_sets_'+str(n)+'_prop_'+rule})] 
            
        intervals=[d*i for i,d in enumerate(durations*rep)]
        intervals.append(intervals[-1]+500.0)
        intervals=[[d,d+500] for d in intervals[1::2]]
        
        self.dic['intervals']=intervals  
        self.dic['repetitions']=rep      
        self.dic['percents']=[100/v for v in n_sets0]
        
        for i in range(len(l)):
            l[i]+=pl({'conn':{'FS_M1_gaba':{'lesion': True },
                                'FS_M2_gaba':{'lesion': True  },
                                'GA_M1_gaba':{'lesion': True },
                                'GA_M2_gaba':{'lesion': True }}},
                               '=',
                               **{'name':'only MSN-MSN'})
        
        return l
 
    def _get_dopamine_levels(self):
        return [self._dop()]  
    
    def _get_striatal_reversal_potentials(self):
        return [self._low()]    

    def get_parameters(self, per):
        
#         per.get_
        return MSN_cluster_compete(**{'other':Inhibition(),
                                      'perturbations':per})   

     
class Builder_MSN_cluster_compete(Builder_MSN_cluster_compete_base, 
                          Mixin_dopamine, 
                          Mixin_general_network, 
                          Mixin_reversal_potential_striatum):
    pass

class Builder_inhibition_striatum_base(Builder_network_base):
    def _variable(self):
        
        l=[]
        l+=[pl(**{'name':'all'})]
        l+=[pl({'conn':{'FS_M1_gaba':{'lesion': True },
                        'FS_M2_gaba':{'lesion': True  },
                        'GA_M1_gaba':{'lesion': True },
                        'GA_M2_gaba':{'lesion': True }}},
                       '=',
                       **{'name':'only MSN-MSN'})]
        l+=[pl({'conn':{'M1_M1_gaba':{'lesion': True },
                        'M1_M2_gaba':{'lesion': True },                     
                        'M2_M1_gaba':{'lesion': True },
                        'M2_M2_gaba':{'lesion': True },
                        'GA_M1_gaba':{'lesion': True },
                        'GA_M2_gaba':{'lesion': True }}},
                       '=',
               **{'name':'only FS-MSN'})]
        
        l+=[pl({'conn':{'M1_M1_gaba':{'lesion': True },
                        'M1_M2_gaba':{'lesion': True },                     
                        'M2_M1_gaba':{'lesion': True },
                        'M2_M2_gaba':{'lesion': True },
                        'GA_M1_gaba':{'lesion': True },
                        'GA_M2_gaba':{'lesion': True },
                        'FS_M1_gaba':{'syn':'FS_M1_gaba_s'},
                        'FS_M2_gaba':{'syn':'FS_M2_gaba_s'}}},
                       '=',
               **{'name':'only FS-MSN-static'})]
        
        l+=[pl({'conn':{'M1_M1_gaba':{'lesion': True },
                        'M1_M2_gaba':{'lesion': True },                     
                        'M2_M1_gaba':{'lesion': True },
                        'M2_M2_gaba':{'lesion': True },
                        'FS_M1_gaba':{'lesion': True },
                        'FS_M2_gaba':{'lesion': True }}},
                       '=',
               **{'name':'only GA-MSN'})]
        l+=[pl({'conn':{'M1_M1_gaba':{'lesion': True },
                        'M1_M2_gaba':{'lesion': True },                     
                        'M2_M1_gaba':{'lesion': True },
                        'M2_M2_gaba':{'lesion': True },
                        'FS_M1_gaba':{'lesion': True },
                        'FS_M2_gaba':{'lesion': True },
                        'GA_M1_gaba':{'lesion': True },
                        'GA_M2_gaba':{'lesion': True}}},
                       '=',
               **{'name':'no inhibition'})]
   
        duration=[500.,500.]*14
        amps0=numpy.linspace(1, 2, len(duration)/2)
        amps=[[1, amp] for amp in amps0]
        amps=numpy.array(reduce(lambda x,y:x+y, amps))
        rep=1
        d={'C1':{'type':'burst2',
                 'params':{'amplitudes':amps,
                           'duration':duration,
                           'repetitions':rep}},
        'C2':{'type':'burst2',
              'params':{'amplitudes':amps,
                        'duration':duration,
                        'repetitions':rep}},
        'CF':{'type':'burst2',
              'params':{'amplitudes':amps, 
                        'duration':duration,
                        'repetitions':rep}},         
        'CS':{'type':'burst2',
              'params':{'amplitudes':amps,
                        'duration':duration,
                        'repetitions':rep}}}
        d={'netw':{'input': d}}   
        
        
        for i in range(len(l)):
            l[i]=l[i]+pl(d, '=')
        
        
        intervals=[d*i for i,d in enumerate(duration)]
        intervals.append(intervals[-1]+500.0)
        intervals=[[d,d+500] for d in intervals[1::2]]
        
        self.dic['intervals']=intervals  
        self.dic['repetitions']=rep 
        self.dic['amplitudes']=amps0   
          
        return l
    
   
    def get_parameters(self, per):
        return Inhibition_striatum(**{'other':Inhibition(),
                                      'perturbations':per})   

    def _get_dopamine_levels(self):
        return [self._dop()]    
    
class Builder_inhibition_striatum(Builder_inhibition_striatum_base, 
                      Mixin_dopamine, 
                      Mixin_general_network, 
                      Mixin_reversal_potential_striatum):
    pass
    
class Builder_slow_wave_base(Builder_abstract):    
      
    def _get_striatal_reversal_potentials(self):
        return [self._low()]    

    def get_parameters(self, per):
        return Slow_wave(**{'other':Inhibition(),
                            'perturbations':per})
     
class Builder_slow_wave(Builder_slow_wave_base, 
                      Mixin_dopamine, 
                      Mixin_general_network, 
                      Mixin_reversal_potential_striatum):
    pass

 
 
class Builder_beta_base(Builder_abstract):    
      
    def _get_striatal_reversal_potentials(self):
        return [self._low()]    

    def get_parameters(self, per):
        return Beta(**{'other':Inhibition(),
                       'perturbations':per})
     
class Builder_beta(Builder_beta_base, 
                      Mixin_dopamine, 
                      Mixin_general_network, 
                      Mixin_reversal_potential_striatum):
    pass
 
 
def get_intervals(rep, durations, d, sequence):
    accum = 0
    intervals = []
    out=[]
    for d in durations * rep:
        intervals.append(accum)
        accum += d
    
    intervals.append(accum)
    for i in range(sequence):
        out.append([[d1, d2] for d1, d2 in zip(intervals[i::sequence], intervals[i+1::sequence])])
    return out
   
class Builder_Go_NoGo_compete_base(Builder_network):    

    def get_parameters(self, per):
        return Go_NoGo_compete(**{'other':Inhibition(),
                       'perturbations':per})




    def _variable(self):
        
        kwargs=self.kwargs
        res=kwargs.get('resolution',5)
        rep=kwargs.get('repetition',5)
        v=numpy.linspace(2.3, 3, res)

        x, y=numpy.meshgrid(v,v)
        x, y=x.ravel(), y.ravel()                
              
        p0=reduce(lambda x,y:x+y, [[1,p] for p in y])
        p1=reduce(lambda x,y:x+y, [[1,p] for p in x])    
        
        durations=[900., 100.]*res*res
        
        l=[]
        for inp_list in [['C1'],
                         ['C1', 'C2']]:
            d={}
      
            for inp in inp_list:
                d=misc.dict_update(d, {'node':{inp:{'n_sets':2}}})        
                
                params={'type':'burst3',
                        'params':{'n_set_pre':2,
                                  'repetitions':rep}}
                
                d_sets={}
                d_sets.update({str(0):{'active':True,
                                       'amplitudes':p0,
                                       'durations':durations,
                                       'proportion_connected':1,
                                  },
                               str(1):{'active':True,
                                       'amplitudes':p1,
                                       'durations':durations,
                                       'proportion_connected':1,
                                  }})
                params['params'].update({'params_sets':d_sets})
   
                d=misc.dict_update(d, {'netw':{'input':{inp:params}}})
            
        l+=[pl(d, '=', **{'name':'_'.join(inp_list)})]        
        
        sequence=2
        intervals = get_intervals(rep, durations, d, sequence)

        self.dic['intervals']=intervals  
        self.dic['repetitions']=rep      

        return l    




class Builder_Go_NoGo_compete(Builder_Go_NoGo_compete_base, 
                      Mixin_dopamine, 
                      Mixin_general_network, 
                      Mixin_reversal_potential_striatum):
    pass




class Builder_single_base(Builder_abstract):   
    
    def get_parameters(self, per):
        return Single_unit(**{'other':Inhibition(**{'perturbations':per}),
                              'perturbations':per})

class Builder_single_FS(Builder_single_base, 
                      Mixin_dopamine, 
                      Mixin_general_single, 
                      Mixin_reversal_potential_FS):
    pass

class Builder_single_M1(Builder_single_base, 
                      Mixin_dopamine, 
                      Mixin_general_single, 
                      Mixin_reversal_potential_M1):
    pass
    
class Builder_single_M2(Builder_single_base, 
                      Mixin_dopamine, 
                      Mixin_general_single, 
                      Mixin_reversal_potential_M2):
    pass
    
class Builder_single_rest(Builder_single_base, 
                      Mixin_dopamine, 
                      Mixin_general_single):
    pass


class Builder_single_GA_GI_base(Builder_single_base): 
      
    def _get_dopamine_levels(self):
        return [self._dop()]    
    
    def _variable(self):
   
        l=[]
        
        for rate in self.kwargs.get('TA_rates'):
            d={'node':{'GA':{'rate':rate}}}
            args=[d, '=']
            l+=[pl(*args)]
        return l


class Builder_single_GA_GI(Builder_single_GA_GI_base, 
                          Mixin_dopamine, 
                          Mixin_general_single):
    pass


class Builder_single_MS_weights_base(Builder_single_base): 
 
    def _get_striatal_reversal_potentials(self):
        return [self._low()]    
      
    def _get_dopamine_levels(self):
        return [self._dop()]    
    
    def _variable(self):
   
        l=[]
        
        for v in self.kwargs.get('conductance_scale'):
            d={'nest':{'GA_M1_gaba':{'weight':v},
                       'GA_M2_gaba':{'weight':v},
                       'M1_M1_gaba':{'weight':v},
                       'M1_M2_gaba':{'weight':v},
                       'M2_M1_gaba':{'weight':v},
                       'M2_M2_gaba':{'weight':v},
                    }}
            l+=[pl(d, '*', **{'name':'scale*'+str(v)})]  
        return l


class Builder_single_M1_weights(Builder_single_MS_weights_base, 
                          Mixin_dopamine, 
                          Mixin_general_single,
                          Mixin_reversal_potential_M1):
    pass

class Builder_single_M2_weights(Builder_single_MS_weights_base, 
                          Mixin_dopamine, 
                          Mixin_general_single,
                          Mixin_reversal_potential_M2):
    pass

def add_perturbations(perturbations, nets):
    if not perturbations:
        return
    for key in sorted(nets.keys()):
#         print nets[key].par
        nets[key].par.update_perturbations(perturbations)


def compute(d, models, attr, **kwargs_dic):
    dout={}
    for keys, val in misc.dict_iter(d):
        if not  isinstance(val, Data_unit_base) and not isinstance(val, Data_units_relation):
            continue
        if keys[1] not in models:
            continue
        for a in attr:
            module=misc.import_module('toolbox.network.manager')
            
            call=getattr(module, a)
            k=kwargs_dic.get(a,{})
            u=call(val, **k)
            dout=misc.dict_recursive_add(dout, keys[0:2]+[a], u)
    
    dout=misc.dict_update(d,dout)
    return d

def dummy_data_dic():
    attr1='spike_signal'
    attr2='voltage_signal'

    d1, d2= {}, {}
    for name in ['dummy1', 'dummy2']:
        d1[name]={}
        d2[name]={}
        
        d1[name][attr1], d2[name][attr2]=dummy_data_du(**{'name':name})
    
    d={'net1':d1}
    return d

def FF_curve(data, **kwargs):
    return data.get_mean_rate_parts()

def firing_rate(data, **kwargs):
       
    if 'set' in kwargs.keys():
        _set=kwargs.get('set', 0)
        del kwargs['set']
        return data[:,_set].get_firing_rate(**kwargs)
    else: 
        return data.get_firing_rate(**kwargs)

def firing_rate_set(data, **kwargs):
    sets=kwargs.get('sets', [0])
    if 'sets' in kwargs.keys():
        del kwargs['sets']

    d={}
    for _set in sets:
        d[_set]=data[:,_set].get_firing_rate(**kwargs)
    
    return d
    

def get_simu(k):
    d={
       'mm_params':{'to_file':False, 
                    'to_memory':True},
       'print_time':k.get('print_time', True),
       'save_conn':k.get('save_conn',{'overwrite':False}),
       'start_rec':k.get('start_rec', 1000.0),
       'sd_params':{'to_file':False, 
                    'to_memory':True},
       'sim_stop':k.get('sim_stop', 2000.0),
       'sim_time':k.get('sim_time', 2000.0),
       'stop_rec':k.get('stop_rec', numpy.inf),
       'threads':k.get('threads', 2),
       }
    return d    

def get_networks(Builder, kwargs_builder={}, kwargs_engine={}):
    
    builder = Builder(**kwargs_builder)
    director = Director()
    director.set_builder(builder)
    info, nets = director.get_networks(**kwargs_engine)
    return info, nets, builder

def get_storage(file_name, info, nets=None):
    sd = Storage_dic.load(file_name, nets)
    sd.add_info(info)
    sd.garbage_collect()
    return sd

def IV_curve(data):
    return data.get_IV_curve()

def IF_curve(data):
    return data.get_IF_curve()

def load(storage_dic, keys, *args):
    return storage_dic.load_dic(keys, *args)

def mean_coherence(data, **k):
    return data.get_mean_coherence(**k)

def mean_rates(data, **k):
    return data.get_mean_rates(**k)

def phase_diff(data, **k):
    return data.get_phase_diff(**k)

def phases_diff_with_cohere(data, **k):
    return data.get_phases_diff_with_cohere(**k)

def psd(data, **k):
    return data.get_psd(**k)

def run(net):
    d=net.simulation_loop()
    return {net.get_name():d}

def save(storage_dic, d):         
        storage_dic.save_dic(d)
                    
def spike_statistic(data,**k):
    return data.get_spike_stats(**k)

       
class TestModuleFunctions(unittest.TestCase):
    
    def setUp(self):
        self.d=dummy_data_dic()
        self.file_name=HOME+'/tmp/manager_unittest'
        self.main_path=HOME+'/tmp/manager_unittest/'
       
    def test_data_functions(self):
        
        for method in [
                     'firing_rate',
                     'mean_rates',
                     'spike_statistic',
                     ]:
            mod = __import__(__name__)
            call=getattr(mod, method)
            for _, val in misc.dict_iter(self.d):
                if not isinstance(val, Data_unit_spk):
                    continue
                _=call(val)  

    def test_get_metwork(self):
        
        info, nets, builder=get_networks(Builder_network, 
                                kwargs_builder={'netw':{'size':10}}, 
                                kwargs_engine={'verbose':False})
        self.assertTrue(type(info)==str)
        self.assertTrue(type(nets)==list)
        self.assertTrue(isinstance(builder, Builder_abstract))
        
    def test_load(self):
        self.test_save()
        d1=self.s.load_dic(self.file_name)    
        self.assertTrue(isinstance(d1,dict))
    
    def test_compute(self):
        class Network_mockup():
            @classmethod
            def simulation_loop(cls):
                return dummy_data_dic()['net1']
            
            @classmethod
            def get_name(self):
                return 'net1'
            
        
        attr=['firing_rate',
              'mean_rates',
              'spike_statistic']
        models=['dummy1','dummy2']
        d = run(Network_mockup)
        d = compute(d, models, attr)
        attr.append('spike_signal')
        self.assertListEqual(d.keys(), ['net1'])
        self.assertListEqual(sorted(d['net1'].keys()), models)
        self.assertListEqual(sorted(d['net1']['dummy1'].keys()), sorted(attr))
        self.assertListEqual(sorted(d['net1']['dummy2'].keys()), sorted(attr))                
        
     
    def test_save(self):
        self.s=Storage_dic(self.file_name)
        save(self.s, self.d)
        self.assertTrue(os.path.isdir(self.file_name+'/'))    
        self.assertTrue(os.path.isfile(self.file_name+'.pkl'))    

             
    def tearDown(self):
        if os.path.isfile(self.file_name+'.pkl'):
            os.remove(self.file_name+'.pkl')        
        
        path=self.file_name+'/'
        if os.path.isdir(path):
            l=os.listdir(path)
            l=[path+ll for ll in l]
            for p in l:
                os.remove(p)
            os.rmdir(path)
            
        
class TestBuilderMixin(object):
    
    def test_1_perturbations_functions(self):
        v=1
        for method in ['_get_general',
                       '_get_striatal_reversal_potentials',
                       '_get_dopamine_levels',
                       '_get_variable',
                       'get_perturbations'
                       ]:
            call=getattr(self.builder,method)
            l=call()
            
            if method=='get_perturbations':
                self.assertEqual(v, len(l))
            else:
                v*=len(l) 
                
            self.assertTrue(type(l)==list)
            for e in l:
                self.assertTrue(isinstance(e, pl))

    def test_2_get_parameters(self):
        
        
        per=self.builder.get_perturbations()
        par=self.builder.get_parameters(per[0])
        self.assertTrue(isinstance(par, Par_base))
        for pl in per:
            for p in pl:
#                 print p
                self.assertTrue(misc.dict_haskey(par.dic, p.keys))
        

    def test_3_get_network(self):
        per=self.builder.get_perturbations()
        par=self.builder.get_parameters(per[0])
        name='dummy'
        net=self.builder.get_network(name, par)
        self.assertTrue(isinstance(net, Network_base))
        

class TestDirectorMixin(object):
    def test_4_director(self):
        director=Director()
        director.set_builder(self.builder)
        _, nets=director.get_networks()
        for net in nets:
            self.assertTrue(isinstance(net, Network_base))
        
class TestBuilder_network_base(unittest.TestCase):
    def setUp(self):
        self.builder=Builder_network()


class TestBuilder_network(TestBuilder_network_base, TestBuilderMixin, 
                          TestDirectorMixin):
    pass

class TestBuilder_single_base(unittest.TestCase):
    def setUp(self):
        self.builder=Builder_network()


class TestBuilder_single(TestBuilder_single_base, TestBuilderMixin, 
                          TestDirectorMixin):
    pass


class TestBuilder_inhibition_striatum_base(unittest.TestCase):
    def setUp(self):
        self.builder=Builder_inhibition_striatum()


class TestBuilder_inhibition_striatum(TestBuilder_inhibition_striatum_base, TestBuilderMixin, 
                          TestDirectorMixin):
    pass


class TestBuilder_MSN_cluster_compete_base(unittest.TestCase):
    def setUp(self):
        self.builder=Builder_MSN_cluster_compete()


class TestBuilder_MSN_cluster_compete(TestBuilder_MSN_cluster_compete_base, 
                                      TestBuilderMixin, 
                                      TestDirectorMixin):
    pass

if __name__ == '__main__':
    
    test_classes_to_run=[
                         TestModuleFunctions,
                        TestBuilder_network,
#                          TestBuilder_single,
#                         TestBuilder_inhibition_striatum,
                        TestBuilder_MSN_cluster_compete
                        ]
    suites_list = []
    for test_class in test_classes_to_run:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)
    unittest.TextTestRunner(verbosity=2).run(big_suite)  

