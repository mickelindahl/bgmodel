'''
Created on Apr 18, 2014

@author: lindahlm

Experiments are produced by builders. Each builder
returns a list of perturbation_lists (from modul default_params). 
Each perturbation list represents an experiment.

The director uses the builder to produce networks. First it 
retrieves the perturbations. It then produce a parameter object from
a parameter class in module default_params with the perturbations as
input. It finally takes the parameter object as input to a network
class from engine module to create a network object.
The network object can then be used to run the model. Thus 
to produce a network the director "directs" all this using the 
builder.
 
'''

import itertools
import numpy
import os
import unittest

from copy import deepcopy        
from core.data_to_disk import Storage_dic
from core.network.data_processing import (dummy_data_du, Data_unit_spk,
                                             Data_unit_base, Data_units_relation)
from core.network.default_params import Perturbation_list as pl,\
    Slow_wave2_EI_EA, Beta_EI_EA, Beta_striatum
from core.network.default_params import (Beta,
                                            Go_NoGo_compete,
                                            Compete_with_oscillations,
                                            Inhibition, 
                                            Inhibition_striatum,
                                            MSN_cluster_compete,
                                            Par_base, 
                                            Single_unit, 
                                            Slow_wave,
                                            Slow_wave2)

from core.network.engine import Network, Network_base, Single_units_activity
from core import misc

import pprint
pp=pprint.pprint

from os.path import expanduser
HOME = expanduser("~")


class Director(object):
    
    __builder=None
    def set_builder(self, builder):
        self.builder=builder
        
    def get_networks(self, kwargs_director, kwargs_engine, kwargs_default_params):
        
        per=self.builder.get_perturbations()
        
        get_these=kwargs_director.get('nets_to_run',[])
                
        nets={}
        info=''
        for i, p in enumerate(per):
            name='Net_'+str(i)
            if get_these and not (name in get_these):
                continue
            info+='Net_'+str(i)+':'+ p.name+'\n'
            par=self.builder.get_parameters(p, **kwargs_default_params)
#             perturbation_consistency(p, par)
            net=self.builder.get_network(name, par, **kwargs_engine)
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
        
    def get_parameters(self, per, **kwargs):
        raise NotImplementedError     
    
    def get_network(self, name, par, **kwargs):
        kwargs['par']=par
        return Network(name, **kwargs)       
          
class Builder_network_base(Builder_abstract):    
    
    def _variable(self):
        
        l=[]
        l+=[pl(**{'name':'no_pert'})]
        return l
    
    def _get_striatal_reversal_potentials(self):
        return [self._low()]    

    def get_parameters(self, per, **kwargs):
        return Inhibition(**{'home':kwargs.get('home'),
                             'home_data':kwargs.get('home_data'),
                             'home_module':kwargs.get('home_module'),
                             'perturbations':per})
     
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
        rep=self.kwargs.get('repetition',5)
        
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

            l+=[pl(d, '=', **{'name':'n_sets_'+str(n)+'_prop_'+rule})] 
        
        sequence = 2
        intervals = get_intervals(rep, durations, d, sequence)

        
        self.dic['intervals']=intervals  
        self.dic['repetition']=rep      
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

    def get_parameters(self, per, **kwargs):
        return MSN_cluster_compete(**{'other':Inhibition(),
                                      'perturbations':per})   

     
class Builder_MSN_cluster_compete(Builder_MSN_cluster_compete_base, 
                          Mixin_dopamine, 
                          Mixin_general_network, 
                          Mixin_reversal_potential_striatum):
    pass

def get_striatum_inhibition_input(l, **k):
    
    
    resolution=k.get('resolution',14)
    repetition=k.get('repetition',1)
    lower=k.get('lower', 1)
    upper=k.get('upper',2)
    
    duration = [500., 500.] * resolution
    amps0 = numpy.linspace(lower, upper, len(duration) / 2)
    amps = [[1, amp] for amp in amps0]
    amps = numpy.array(reduce(lambda x, y:x + y, amps))
    rep =repetition
    d = {'C1':{'type':'burst2', 'params':{'amplitudes':amps, 
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
    d = {'netw':{'input':d}}
    for i in range(len(l)):
        l[i] = l[i] + pl(d, '=')
    
#         intervals=[d*i for i,d in enumerate(duration)]
#         intervals.append(intervals[-1]+500.0)
#         intervals=[[d,d+500] for d in intervals[1::2]]
#
    sequence = 2
    intervals = get_intervals(rep, duration, d, sequence)
    return intervals, rep, amps0

class Builder_single_base(Builder_network_base):

    def _variable(self):
        
        l=[]
        l+=[pl({}, '=', **{'name':'single_net'})]      
        return l
    
   
    def get_parameters(self, per, **kwargs):
        
        return Single_unit(**{'other':Inhibition(),
                              'perturbations':per})  

    def _get_dopamine_levels(self):
        return [self._dop(), self._no_dop()]    

    def get_network(self, name, par, **kwargs):
        kwargs['par']=par
        return Single_units_activity(name, **kwargs) 
    
class Builder_single(Builder_single_base, 
                      Mixin_dopamine, 
                      Mixin_general_network, 
                      Mixin_reversal_potential_striatum):
    pass

class Builder_beta_GA_GI_ST_base(Builder_network_base):

    def _variable(self):
        
        l=[]
        l+=[pl({'node':{'C1':{'lesion': True },
                        'C2':{'lesion': True },
                        'CF':{'lesion': True },
                        'M1':{'lesion': True },
                        'M2':{'lesion': True  },
                        'FS':{'lesion': True },
                        'SN':{'lesion': True }}},
                       '=',
                       **{'name':'GA_GI_ST_net'})]      
        return l
    
   
    def get_parameters(self, per, **kwargs):
    
        return Beta(**{'other':Inhibition(),
                       'perturbations':per})  

    def _get_dopamine_levels(self):
        return [self._dop(), self._no_dop()]    
    
class Builder_beta_GA_GI_ST(Builder_beta_GA_GI_ST_base, 
                      Mixin_dopamine, 
                      Mixin_general_network, 
                      Mixin_reversal_potential_striatum):
    pass


class Builder_slow_wave2_GA_GI_ST_base(Builder_network_base):

    def _variable(self):
        
        l=[]
        l+=[pl({'node':{'C1':{'lesion': True },
                        'C2':{'lesion': True },
                        'CF':{'lesion': True },
                        'M1':{'lesion': True },
                        'M2':{'lesion': True  },
                        'FS':{'lesion': True },
                        'SN':{'lesion': True }}},
                       '=',
                       **{'name':'GA_GI_ST_net'})]      
        return l
    
   
    def get_parameters(self, per, **kwargs):
    
        return Slow_wave2(**{'other':Inhibition(),
                       'perturbations':per})  

    def _get_dopamine_levels(self):
        return [self._dop(), self._no_dop()]    
    
class Builder_slow_wave2_GA_GI_ST(Builder_slow_wave2_GA_GI_ST_base, 
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
   
#         res=self.kwargs.get('resolution',5)
#         rep=self.kwargs.get('repetition',5)
   
        intervals, rep, amps0 = get_striatum_inhibition_input(l, 
                                                              **self.kwargs)
        
        self.dic['intervals']=intervals  
        self.dic['repetitions']=rep 
        self.dic['amplitudes']=amps0   
          
        return l
    
   
    def get_parameters(self, per, **kwargs):
        return Inhibition_striatum(**{'other':Inhibition(),
                                      'perturbations':per})    

    def _get_dopamine_levels(self):
        return [self._dop()]    
    
class Builder_inhibition_striatum(Builder_inhibition_striatum_base, 
                      Mixin_dopamine, 
                      Mixin_general_network, 
                      Mixin_reversal_potential_striatum):
    pass


class Builder_striatum_base(Builder_network_base):

    def _variable(self):
        
        l=[]
        l+=[pl(**{'name':'all'})]
   
#         res=self.kwargs.get('resolution',5)
#         rep=self.kwargs.get('repetition',5)
   
        intervals, rep, amps0 = get_striatum_inhibition_input(l, **self.kwargs)
        
        self.dic['intervals']=intervals  
        self.dic['repetitions']=rep 
        self.dic['amplitudes']=amps0   
          
        return l
    
   
    def get_parameters(self, per, **kwargs):
        return Inhibition_striatum(**{'other':Inhibition(),
                                      'perturbations':per})   

    def _get_dopamine_levels(self):
        return [self._dop()]    
    
class Builder_striatum(Builder_striatum_base, 
                      Mixin_dopamine, 
                      Mixin_general_network, 
                      Mixin_reversal_potential_striatum):
    pass
    
class Builder_slow_wave_base(Builder_abstract):    
      
    def _get_striatal_reversal_potentials(self):
        return [self._low()]    

    def get_parameters(self, per, **kwargs):
        d={'home':kwargs.get('home'),
           'home_data':kwargs.get('home_data'),
           'home_module':kwargs.get('home_module')}
        return Slow_wave(**{'other':Inhibition(**d),
                            'perturbations':per})
     
class Builder_slow_wave(Builder_slow_wave_base, 
                        Mixin_dopamine, 
                        Mixin_general_network, 
                        Mixin_reversal_potential_striatum):
    pass

class Builder_slow_wave2_base(Builder_abstract):    
      
    def _get_striatal_reversal_potentials(self):
        return [self._low()]    

    def get_parameters(self, per, **kwargs):
        d={'home':kwargs.get('home'),
           'home_data':kwargs.get('home_data'),
           'home_module':kwargs.get('home_module')}
        return Slow_wave2(**{'other':Inhibition(**d),
                            'perturbations':per})
     
class Builder_slow_wave2(Builder_slow_wave2_base, 
                      Mixin_dopamine, 
                      Mixin_general_network, 
                      Mixin_reversal_potential_striatum):
    pass

class Builder_slow_wave2_EI_EA_base(Builder_abstract):    
      
    def _get_striatal_reversal_potentials(self):
        return [self._low()]    

    def get_parameters(self, per, **kwargs):
        d={'home':kwargs.get('home'),
           'home_data':kwargs.get('home_data'),
           'home_module':kwargs.get('home_module')}
        return Slow_wave2_EI_EA(**{'other':Inhibition(**d),
                                   'perturbations':per})
     
class Builder_slow_wave2_EI_EA(Builder_slow_wave2_EI_EA_base, 
                      Mixin_dopamine, 
                      Mixin_general_network, 
                      Mixin_reversal_potential_striatum):
    pass

class Builder_slow_wave2_perturb_base(Builder_abstract):    
    
    def _variable(self):
        return get_model_conn_perturbation()  
      
    def _get_striatal_reversal_potentials(self):
        return [self._low()]    

    def _get_dopamine_levels(self):
        return [self._dop(), self._no_dop()]    

    def get_parameters(self, per, **kwargs):
        d={'home':kwargs.get('home'),
           'home_data':kwargs.get('home_data'),
           'home_module':kwargs.get('home_module')}
        return Slow_wave2(**{'other':Inhibition(**d),
                            'perturbations':per})
     
class Builder_slow_wave2_perturb(Builder_slow_wave2_perturb_base, 
                      Mixin_dopamine, 
                      Mixin_general_network, 
                      Mixin_reversal_potential_striatum):
    pass

 
class Builder_beta_base(Builder_abstract):    
      
    def _get_striatal_reversal_potentials(self):
        return [self._low()]    

    def get_parameters(self, per, **kwargs):

        return Beta(**{'other':Inhibition(),
                       'perturbations':per})
     
class Builder_beta(Builder_beta_base, 
                      Mixin_dopamine, 
                      Mixin_general_network, 
                      Mixin_reversal_potential_striatum):
    pass
 
class Builder_beta_EI_EA_base(Builder_abstract):    
      
    def _get_striatal_reversal_potentials(self):
        return [self._low()]    

    def get_parameters(self, per, **kwargs):
        d={'home':kwargs.get('home'),
           'home_data':kwargs.get('home_data'),
           'home_module':kwargs.get('home_module')}
        return Beta_EI_EA(**{'other':Inhibition(**d),
                             'perturbations':per})
     
class Builder_beta_EI_EA(Builder_beta_EI_EA_base, 
                      Mixin_dopamine, 
                      Mixin_general_network, 
                      Mixin_reversal_potential_striatum):
    pass 

class Builder_beta_striatum_base(Builder_abstract):    
      
    def _get_striatal_reversal_potentials(self):
        return [self._low()]    

    def get_parameters(self, per, **kwargs):
        d={'home':kwargs.get('home'),
           'home_data':kwargs.get('home_data'),
           'home_module':kwargs.get('home_module')}
        return Beta_striatum(**{'other':Inhibition(**d),
                                'perturbations':per})
     
class Builder_beta_striatum(Builder_beta_striatum_base, 
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


def _get_input_go_nogo_p0_and_p1(res, dur):
    v = numpy.linspace(1, 3, res)
    x, y = numpy.meshgrid(v, v)
    x, y = x.ravel(), y.ravel()
    p0 = reduce(lambda x, y:x + y, [[1., p] for p in y])
    p1 = reduce(lambda x, y:x + y, [[1., p] for p in x])
    durations = dur * res * res
    return durations, p0,  p1, x, y


    
def get_input_Go_NoGo(kwargs):
    n_sets=kwargs.get('n_sets', 2)
    
    input_type=kwargs.get('input_type')
    
    p_pulse=kwargs.get('p_pulse')
    res = kwargs.get('resolution', 2)
    rep = kwargs.get('repetition', 2)
    dur = kwargs.get('duration', [1000., 500.])
    prop_conn=kwargs.get('proportion_connected', 1)
    
    act=kwargs.get('act',['M1', 'M2', 'GI', 'SN'])
    act_input=kwargs.get('act_input', ['C1', 'C2'])
    no_act=kwargs.get('no_act',['FS', 'GA', 'ST'])
    other_scenario=kwargs.get('other_scenario',False)
    
    if other_scenario:
        conn_setup=[['M1_SN_gaba', 'M2_GI_gaba', 'GI_SN_gaba'], 
                     ['set-set', 'set-set',  'set-set']] 
    else:
        conn_setup=[['M1_SN_gaba', 'M2_GI_gaba', 'GI_SN_gaba'], 
                    ['set-set', 'set-set', 'all-all']]
      
    input_lists = kwargs.get('input_lists', [['C1'], ['C1', 'C2']])
    
    durations, p0,  p1, x, y = _get_input_go_nogo_p0_and_p1(res, dur)

 
    l = []
    for inp_list in input_lists:
        
        inp_no_ch=[name for name in ['CF', 'CS', 'C2'] 
                   if name not in inp_list]
        
        d = {}
        for node in act:
            d = misc.dict_update(d, {'node':{node:{'n_sets':2}}})
        for node in no_act:
            d = misc.dict_update(d, {'node':{node:{'n_sets':1}}})                
        for conn, rule in zip(*conn_setup):
            d = misc.dict_update(d, {'conn':{conn:{'rule':rule}}})

        for inp in inp_list :
            if inp not in act_input:
                continue
            d = misc.dict_update(d, {'node':{inp:{'n_sets':n_sets}}})
            params = {'type':input_type, 
                'params':{'n_set_pre':n_sets, 
                          'repetitions':rep}}
            d_sets = {}
            
            if inp=='C2' and other_scenario:
                p_tmp=p0
                p0=p1
                p1=p_tmp
                
            d_sets.update({str(0):{'active':True, 
                                   'amplitudes':p0, 
                                   'durations':durations, 
                                   'proportion_connected':prop_conn}, 
                           
                           str(1):{'active':True, 
                                   'amplitudes':p1, 
                                   'durations':durations, 
                                   'proportion_connected':prop_conn}})
            
            params['params'].update({'params_sets':d_sets})
            d = misc.dict_update(d, {'netw':{'input':{inp:params}}})

        for inp in inp_list :
            if inp=='CS_pulse':
                continue
            
            if inp in act_input:
                continue
            
            d = misc.dict_update(d, {'node':{inp:{'n_sets':1}}})
            params = {'type':input_type, 
                      'params':{'n_set_pre':1, 
                                'repetitions':rep}}
            d_sets = {}
            d_sets.update({str(0):{'active':True, 
                                   'amplitudes':list((numpy.array(p0) 
                                                      +numpy.array(p1))/2), 
                                   'durations':durations, 
                                   'proportion_connected':1}})
            
            params['params'].update({'params_sets':d_sets})
            d = misc.dict_update(d, {'netw':{'input':{inp:params}}})            
        
        for inp in inp_no_ch:
            d = misc.dict_update(d, {'node':{inp:{'n_sets':1}}})
            params = {'type':input_type, 
                      'params':{'n_set_pre':1, 
                                'repetitions':1}}
            d_sets = {}
            
            d_sets.update({str(0):{'active':True, 
                                   'amplitudes':[1], 
                                   'durations':[sum(durations)], 
                                   'proportion_connected':1}})
            
            params['params'].update({'params_sets':d_sets})
            d = misc.dict_update(d, {'netw':{'input':{inp:params}}})            
   
            
        if 'CS_pulse' in inp_list:
            inp='CS'
            pamp=p0[:]
            pamp[1::2]=[p_pulse]*len(pamp[1::2])
            d = misc.dict_update(d, {'node':{inp:{'n_sets':1}}})
            params = {'type':input_type, 
                      'params':{'n_set_pre':1, 
                                'repetitions':rep}}
            d_sets = {}
            d_sets.update({str(0):{'active':True, 
                                   'amplitudes':pamp, 
                                   'durations':durations, 
                                   'proportion_connected':1}})
            
            params['params'].update({'params_sets':d_sets})
            d = misc.dict_update(d, {'netw':{'input':{inp:params}}})  
        
        if input_type in ['burst3_oscillations']:
                
            for inp in ['C1','C2','CF','CS']:
                
                if kwargs.get('STN_amp_mod') and inp=='CS':
                    f=kwargs.get('STN_amp_mod')
                else:
                    f=1
                
#                 if inp in kwargs.get('amp_base_skip'):
                amp_base=kwargs.get('amp_base')
#                 else:
#                     amp_base=0
                
                dtmp={'p_amplitude_upp':kwargs.get('freqs')*f,
                      'p_amplitude_down':-kwargs.get('freqs')*f,
                      'p_amplitude0':amp_base,
                      'freq': kwargs.get('freq_oscillations'),
                      'period':'constant'}
            
                params={'params':dtmp}
                d = misc.dict_update(d, {'netw':{'input':{inp:params}}})

        l += [pl(d, '=', **{'name':'_'.join(inp_list)})]
            
    sequence = 2
    intervals = get_intervals(rep, durations, d, sequence)
    dic={}
    dic['intervals'] = intervals
    dic['repetitions'] = rep
    dic['x'] = x
    dic['y'] = y
    return l, dic


class Builder_Go_NoGo_compete_base(Builder_network):    

    def get_parameters(self, per, **kwargs):
        return Go_NoGo_compete(**{'other':Inhibition(),
                                  'perturbations':per})


    def _get_dopamine_levels(self):
        return [self._dop()]    
    

    def _variable(self):
        
        l, self.dic = get_input_Go_NoGo(self.kwargs)      
        
        return l    




class Builder_Go_NoGo_compete(Builder_Go_NoGo_compete_base, 
                      Mixin_dopamine, 
                      Mixin_general_network, 
                      Mixin_reversal_potential_striatum):
    pass



class Builder_Go_NoGo_compete_oscillations_base(Builder_network):    

    def get_parameters(self, per, **kwargs):
        d={'home':kwargs.get('home'),
           'home_data':kwargs.get('home_data'),
           'home_module':kwargs.get('home_module')}
        return Compete_with_oscillations(**{'other':Inhibition(**d),
                                            'perturbations':per})


    def _get_dopamine_levels(self):
        return [self._dop()]    
    

    def _variable(self):
        
        l, self.dic = get_input_Go_NoGo(self.kwargs)      
        
        return l    




class Builder_Go_NoGo_compete_oscillations(Builder_Go_NoGo_compete_oscillations_base, 
                      Mixin_dopamine, 
                      Mixin_general_network, 
                      Mixin_reversal_potential_striatum):
    pass


def add_lesions_Go_NoGo(l):
    lesions = [{'conn':{'M1_M1_gaba':{'lesion':True}, 
                        'M1_M2_gaba':{'lesion':True}, 
                        'M2_M1_gaba':{'lesion':True}, 
                        'M2_M2_gaba':{'lesion':True}}}, 
                {'conn':{'FS_M1_gaba':{'lesion':True}, 
                         'FS_M2_gaba':{'lesion':True}}}, 
                {'conn':{'GA_M1_gaba':{'lesion':True}, 
                         'GA_M2_gaba':{'lesion':True}}}]
    
    names = ['no-MS-MS', 
             'no-FS', 
             'no_GP']
    for lesion, name in zip(lesions, names):
        l += [deepcopy(l[1])]
        l[-1] += pl(lesion, 
            '=', **
            {'name':name})
    
    return l




def set_lesions_scenarios_GPe_Go_NoGo(l):


    lesions = [{'conn':{'GA_FS_gaba':{'lesion':True}}},
               {'conn':{'GA_M1_gaba':{'lesion':True}}},
               {'conn':{'GA_M2_gaba':{'lesion':True}}},
               
               {'conn':{'GA_FS_gaba':{'lesion':True},
                        'GA_M1_gaba':{'lesion':True}}},
               
               {'conn':{'GA_FS_gaba':{'lesion':True},
                        'GA_M2_gaba':{'lesion':True}}},
               
               {'conn':{'GA_M1_gaba':{'lesion':True},
                        'GA_M2_gaba':{'lesion':True}}}]
    
    names = ['no-GP_FS', 
             'no-GP_M1', 
             'no_GP_M2',
             'no_GP_FS-M1',
             'no_GP_FS_M2',
             'no_GP_M1_M2']
    
    for lesion, name in zip(lesions, names):
        l += [deepcopy(l[0])]
        l[-1] += pl(lesion, 
            '=', **
            {'name':name})
    
    return l[-6:]

class Builder_Go_NoGo_with_lesion_base(Builder_network):    

    def get_parameters(self, per, **kwargs):
        return Go_NoGo_compete(**{'other':Inhibition(),
                                  'perturbations':per})


    def _get_dopamine_levels(self):
        return [self._dop()]    
    
    def _variable(self):
        
        l, self.dic = get_input_Go_NoGo(self.kwargs)    
        

          
        l = add_lesions_Go_NoGo(l)             

        del l[0]
        del l[1]
        
        return l    

class Builder_Go_NoGo_with_lesion(Builder_Go_NoGo_with_lesion_base, 
                      Mixin_dopamine, 
                      Mixin_general_network, 
                      Mixin_reversal_potential_striatum):
    pass     

class Builder_Go_NoGo_with_nodop_base(Builder_network):    

    def get_parameters(self, per, **kwargs):
        return Go_NoGo_compete(**{'other':Inhibition(),
                                  'perturbations':per})


    def _get_dopamine_levels(self):
        return [self._dop(), self._no_dop()]    
    
    def _variable(self):
        
        

        l, self.dic = get_input_Go_NoGo(self.kwargs)    
        
        return l    

class Builder_Go_NoGo_with_nodop(Builder_Go_NoGo_with_nodop_base, 
                      Mixin_dopamine, 
                      Mixin_general_network, 
                      Mixin_reversal_potential_striatum):
    pass     

class Builder_Go_NoGo_with_nodop_FS_base(Builder_network):    

    def get_parameters(self, per, **kwargs):
        return Go_NoGo_compete(**{'other':Inhibition(),
                                  'perturbations':per})


    def _get_dopamine_levels(self):
        return [self._dop(), self._no_dop()]    
    
    def _variable(self):
        
        self.kwargs['input_lists']= [
#                                      ['C1','CF'], 
                                     ['C1', 'C2', 'CF']
                                     ]        

        l, self.dic = get_input_Go_NoGo(self.kwargs)    
        
        return l    

class Builder_Go_NoGo_with_nodop_FS(Builder_Go_NoGo_with_nodop_FS_base, 
                      Mixin_dopamine, 
                      Mixin_general_network, 
                      Mixin_reversal_potential_striatum):
    pass    

class Builder_Go_NoGo_with_nodop_FS_oscillation_base(Builder_network):    

    def get_parameters(self, per, **kwargs):
        return Compete_with_oscillations(**{'other':Inhibition(),
                                            'perturbations':per})


    def _get_dopamine_levels(self):
        return [self._dop(), self._no_dop()]    
    
    def _variable(self):
        
        self.kwargs['input_lists']= [
#                                      ['C1','CF'], 
                                     ['C1', 'C2', 'CF']
                                     ]        

        l, self.dic = get_input_Go_NoGo(self.kwargs)    
        
        return l    

class Builder_Go_NoGo_with_nodop_FS_oscillation(Builder_Go_NoGo_with_nodop_FS_oscillation_base, 
                      Mixin_dopamine, 
                      Mixin_general_network, 
                      Mixin_reversal_potential_striatum):
    pass 
     
class Builder_Go_NoGo_with_lesion_FS_base(Builder_network):    

    def get_parameters(self, per, **kwargs):
        return Go_NoGo_compete(**{'other':Inhibition(),
                                  'perturbations':per})


    def _get_dopamine_levels(self):
        return [self._dop()]    
    
    def _variable(self):
        
        self.kwargs['input_lists']= [['C1','CF'], 
                                     ['C1', 'C2', 'CF']]

        
        l, self.dic = get_input_Go_NoGo(self.kwargs)      
        

        l = add_lesions_Go_NoGo(l)    
        

        
        return l    
    
class Builder_Go_NoGo_with_lesion_FS(Builder_Go_NoGo_with_lesion_FS_base, 
                                     Mixin_dopamine, 
                                     Mixin_general_network, 
                                     Mixin_reversal_potential_striatum):
    pass


class Builder_Go_NoGo_with_lesion_FS_base_oscillation(Builder_network):    

    def get_parameters(self, per, **kwargs):
        return Compete_with_oscillations(**{'other':Inhibition(),
                                            'perturbations':per})


    def _get_dopamine_levels(self):
        return [self._dop()]    
    
    def _variable(self):
        
        self.kwargs['input_lists']= [['C1','CF'], 
                                     ['C1', 'C2', 'CF']]

        
        l, self.dic = get_input_Go_NoGo(self.kwargs)      
        

        l = add_lesions_Go_NoGo(l)    
        

        
        return l    
    
class Builder_Go_NoGo_with_lesion_FS_oscillation(Builder_Go_NoGo_with_lesion_FS_base_oscillation, 
                                     Mixin_dopamine, 
                                     Mixin_general_network, 
                                     Mixin_reversal_potential_striatum):
    pass


class Builder_Go_NoGo_only_D1D2_FS_base(Builder_network):    

    def get_parameters(self, per, **kwargs):
        return Go_NoGo_compete(**{'other':Inhibition(),
                                  'perturbations':per})


    def _get_dopamine_levels(self):
        return [self._dop()]    
    
    def _variable(self):
        
        self.kwargs['input_lists']= [
                                     ['C1', 'C2', 'CF']]

        
        l, self.dic = get_input_Go_NoGo(self.kwargs)      
            
        return l    
    
class Builder_Go_NoGo_only_D1D2_FS(Builder_Go_NoGo_only_D1D2_FS_base, 
                                     Mixin_dopamine, 
                                     Mixin_general_network, 
                                     Mixin_reversal_potential_striatum):
    pass



class Builder_Go_NoGo_only_D1D2_nodop_FS_base(Builder_network):    

    def get_parameters(self, per, **kwargs):
        d={'home':kwargs.get('home'),
           'home_data':kwargs.get('home_data'),
           'home_module':kwargs.get('home_module')}
        return Go_NoGo_compete(**{'other':Inhibition(**d),
                                  'perturbations':per})


    def _get_dopamine_levels(self):
        return [self._no_dop()]    
    
    def _variable(self):
        
        self.kwargs['input_lists']= [
                                     ['C1', 'C2', 'CF']]

        
        l, self.dic = get_input_Go_NoGo(self.kwargs)      
            
        return l    
    
class Builder_Go_NoGo_only_D1D2_nodop_FS(Builder_Go_NoGo_only_D1D2_nodop_FS_base, 
                                     Mixin_dopamine, 
                                     Mixin_general_network, 
                                     Mixin_reversal_potential_striatum):
    pass



class Builder_Go_NoGo_only_D1D2_nodop_FS_oscillations_base(Builder_network):    

    def get_parameters(self, per, **kwargs):
        return Compete_with_oscillations(**{'other':Inhibition(),
                                            'perturbations':per})


    def _get_dopamine_levels(self):
        return [self._no_dop()]    
    
    def _variable(self):
        
        self.kwargs['input_lists']= [
                                     ['C1', 'C2', 'CF']]

        
        l, self.dic = get_input_Go_NoGo(self.kwargs)      
            
        return l    
    
class Builder_Go_NoGo_only_D1D2_nodop_FS_oscillations(
                                    Builder_Go_NoGo_only_D1D2_nodop_FS_oscillations_base, 
                                     Mixin_dopamine, 
                                     Mixin_general_network, 
                                     Mixin_reversal_potential_striatum):
    pass

class Builder_Go_NoGo_with_lesion_FS_ST_base(Builder_network):    

    def get_parameters(self, per, **kwargs):
        return Go_NoGo_compete(**{'other':Inhibition(),
                                  'perturbations':per})


    def _get_dopamine_levels(self):
        return [self._dop()]    
    
    def _variable(self):
        
        self.kwargs['input_lists']= [['C1','CF', 'CS'], 
                                     ['C1', 'C2', 'CF', 'CS']]

        
        l, self.dic = get_input_Go_NoGo(self.kwargs)      
        

        l = add_lesions_Go_NoGo(l)    
        

        
        return l    


class Builder_Go_NoGo_with_lesion_FS_ST(Builder_Go_NoGo_with_lesion_FS_ST_base, 
                                     Mixin_dopamine, 
                                     Mixin_general_network, 
                                     Mixin_reversal_potential_striatum):
    pass

class Builder_Go_NoGo_with_lesion_FS_ST_pulse_base(Builder_network):    

    def get_parameters(self, per, **kwargs):
        return Go_NoGo_compete(**{'other':Inhibition(),
                                   'perturbations':per})


    def _get_dopamine_levels(self):
        return [self._dop()]    
    
    def _variable(self):
        
        self.kwargs['input_lists']= [['C1', 'C2', 'CF', 'CS_pulse']]

        ll=[]
        for pulse in self.kwargs['p_pulses']: 
            self.kwargs['p_pulse']=pulse
            l, self.dic = get_input_Go_NoGo(self.kwargs)      
            ll+=l
        return ll    


class Builder_Go_NoGo_with_lesion_FS_ST_pulse(Builder_Go_NoGo_with_lesion_FS_ST_pulse_base, 
                                     Mixin_dopamine, 
                                     Mixin_general_network, 
                                     Mixin_reversal_potential_striatum):
    pass
class Builder_Go_NoGo_with_lesion_FS_ST_pulse_oscillation_base(Builder_network):    

    def get_parameters(self, per, **kwargs):
        return Compete_with_oscillations(**{'other':Inhibition(),
                                            'perturbations':per})


    def _get_dopamine_levels(self):
        return [self._dop()]    
    
    def _variable(self):
        
        self.kwargs['input_lists']= [['C1', 'C2', 'CF', 'CS_pulse']]

        ll=[]
        for pulse in self.kwargs['p_pulses']: 
            self.kwargs['p_pulse']=pulse
            l, self.dic = get_input_Go_NoGo(self.kwargs)      
            ll+=l
        return ll    


class Builder_Go_NoGo_with_lesion_FS_ST_pulse_oscillation(Builder_Go_NoGo_with_lesion_FS_ST_pulse_oscillation_base, 
                                     Mixin_dopamine, 
                                     Mixin_general_network, 
                                     Mixin_reversal_potential_striatum):
    pass
class Builder_Go_NoGo_with_GP_scenarios_FS_base(Builder_network):    

    def get_parameters(self, per, **kwargs):
        return Go_NoGo_compete(**{'other':Inhibition(**kwargs),
                                  'perturbations':per})


    def _get_dopamine_levels(self):
        return [self._dop()]    
    
    def _variable(self):
        
        self.kwargs['input_lists']= [['C1', 'C2', 'CF']]

        
        l, self.dic = get_input_Go_NoGo(self.kwargs)      
        

        l = set_lesions_scenarios_GPe_Go_NoGo(l)    
        

        
        return l    


class Builder_Go_NoGo_with_GP_scenarios__FS(Builder_Go_NoGo_with_GP_scenarios_FS_base, 
                      Mixin_dopamine, 
                      Mixin_general_network, 
                      Mixin_reversal_potential_striatum):
    pass
 
class Builder_Go_NoGo_with_GP_scenarios_FS_ST_base(Builder_network):    

    def get_parameters(self, per, **kwargs):
        return Go_NoGo_compete(**{'other':Inhibition(**kwargs),
                                  'perturbations':per})

    def _get_dopamine_levels(self):
        return [self._dop()]    
    
    def _variable(self):
        
        self.kwargs['input_lists']= [['C1', 'C2', 'CF', 'CS']]

        
        l, self.dic = get_input_Go_NoGo(self.kwargs)      
        

        l = set_lesions_scenarios_GPe_Go_NoGo(l)    
        

        
        return l    


class Builder_Go_NoGo_with_GP_scenarios_FS_ST(Builder_Go_NoGo_with_GP_scenarios_FS_ST_base, 
                      Mixin_dopamine, 
                      Mixin_general_network, 
                      Mixin_reversal_potential_striatum):
    pass 
        
class Builder_Go_NoGo_with_lesion_FS_act_base(Builder_network):    

    def get_parameters(self, per, **kwargs):
        return Go_NoGo_compete(**{'other':Inhibition(**kwargs),
                                  'perturbations':per})


    def _get_dopamine_levels(self):
        return [self._dop()]    
    
    def _variable(self):
        
        self.kwargs['input_lists']= [['C1','CF'], 
                                     ['C1', 'C2', 'CF']]
        
        self.kwargs['act']=['M1', 'M2','FS', 'GI', 'SN']
        self.kwargs['act_input'] = ['C1', 'C2', 'CF']
        self.kwargs['no_act'] =[ 'GA', 'ST']
        l, self.dic = get_input_Go_NoGo(self.kwargs)      
        l = add_lesions_Go_NoGo(l)    
        
        for i, _ in enumerate(l):
            l[i]+= pl({'conn':{'FS_M1_gaba':{'rule':'set-not_set'},
                               'FS_M2_gaba':{'rule':'set-not_set'}}},'=',**{'name':'FS_act'})
        
        return l    


class Builder_Go_NoGo_with_lesion_FS_act(Builder_Go_NoGo_with_lesion_FS_act_base, 
                      Mixin_dopamine, 
                      Mixin_general_network, 
                      Mixin_reversal_potential_striatum):
    pass  

class Builder_Go_NoGo_with_lesion_GA_act_base(Builder_network):    

    def get_parameters(self, per, **kwargs):
        return Go_NoGo_compete(**{'other':Inhibition(**kwargs),
                                  'perturbations':per})


    def _get_dopamine_levels(self):
        return [self._dop()]    
    
    def _variable(self):
        
        self.kwargs['input_lists']= [['C1'], 
                                     ['C1', 'C2']]
        
        self.kwargs['act']=['M1', 'M2','GA', 'GI', 'SN']
        self.kwargs['act_input'] = ['C1', 'C2']
        self.kwargs['no_act'] =[ 'FS', 'ST']
        l, self.dic = get_input_Go_NoGo(self.kwargs)      
        l = add_lesions_Go_NoGo(l)    
        
        for i, _ in enumerate(l):
            l[i]+= pl({'conn':{'GI_GA_gaba':{'rule':'set-set'},
                               'GA_M1_gaba':{'rule':'set-not_set'},
                               'GA_M2_gaba':{'rule':'set-not_set'}}},'=',**{'name':'GA_act'})
        
        return l    


class Builder_Go_NoGo_with_lesion_GA_act(Builder_Go_NoGo_with_lesion_GA_act_base, 
                      Mixin_dopamine, 
                      Mixin_general_network, 
                      Mixin_reversal_potential_striatum):
    pass  



class Builder_single_base(Builder_abstract):   
    
    def get_parameters(self, per, **kwargs):
        return Single_unit(**{'other':Inhibition(**kwargs),
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
#         perturbation_consistency(perturbations, nets[key].par)

        nets[key].par.update_perturbations(perturbations)

def perturbation_consistency(pl, par):
    err=[]
    for p in pl:
        print p
        if not misc.dict_haskey(par.dic, p.keys):
            err.append(p.keys)
    if err:
        
        msg='Keys missing:\n'
        for keys in err:
            msg+='.'.join(keys)+'\n'
        
        raise LookupError(msg)
        
def compute_dependables(obj, dout, kwargs, key, _set=None):
    for attr  in kwargs.keys():
        
        if (len(attr)>=len('mean_rate_slices') and 
            attr[0:16]=='mean_rate_slices'):            
#             pp(kwargs)
            #pick out kwargs for the signal
            _kwargs=kwargs[attr]
            if _set!=None:
                update_kwargs_with_set_dic(_kwargs, _set)
                
            u=obj.get_mean_rate_slices(**_kwargs)
            key[-1]=attr
            dout=misc.dict_recursive_add(dout, key, u)
             


def update_kwargs_with_set_dic(k, s):
    name='set_'+str(s)
    if name in k.keys():
        k.update(k[name])
    k['set'] = s

def compute(d, models, attr, **kwargs_dic):
    dout={}
    for keys, val in misc.dict_iter(d):
        
        print 'Computing',keys
        if not  isinstance(val, Data_unit_base) and not isinstance(val, Data_units_relation):
            continue
        if keys[1] not in models:
            continue
        for a in attr:
            
            
            if a[-1].isdigit():
                a_name=a[0:-2]
            else:
                a_name=a
            
            module=misc.import_module('core.network.manager')
            
            #module=manager then there is an 
            #function of name a_name there.
            #The function can slice using set information
            #from a spike obj 
            call=getattr(module, a_name)
            
            k=kwargs_dic.get(a, {}).copy()
            sets=k.pop('sets', None)
            
            
#             if (keys[1]=='GA_GA'
#                 and a=='phases_diff_with_cohere'):
#                 k['inspect_phases_diff']=True
# #                 k['inspect_phase_diff']=True
#             if a=='phases_diff_with_cohere':
#                 k['inspect_phases_diff']=True
#                 print keys[1]
            
            if sets!=None:
                
                #ensure that sets i not more than
                #size of val (spk signal)
                max_sets=val.wrap.shape[1]
                
                sets=[s for s in sets if s<max_sets ]
                
                for s in sets:

                    update_kwargs_with_set_dic(k, s)
                    
                    #call cuts out from set
                    u=call(val, **k)
                    
                    name='set_'+str(s)
                    key=[keys[0], name, keys[1], a]
                    dout=misc.dict_recursive_add(dout, key, u)
                    compute_dependables(u, dout, k, key, s)
                
            else:
                u=call(val, **k)
                key=keys[0:2]+[a]
                dout=misc.dict_recursive_add(dout, key, u)
                compute_dependables(u, dout, k, key)
    
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
    _set=kwargs.pop('set', 0)
    return data[:,_set].get_firing_rate(**kwargs)


def firing_rate_set(data, **kwargs):
    sets=kwargs.pop('sets', [0])
 
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
       'local_num_threads':k.get('local_num_threads', 2),
       }
    return d    

def get_networks(Builder, k_builder={}, k_director={}, k_engine={}, k_default_params={}):
    
    builder = Builder(**k_builder)
    director = Director()
    director.set_builder(builder)
    info, nets = director.get_networks(k_director, k_engine, k_default_params)
    return info, nets, builder

def get_storage(file_name, nets=None):


    sd = Storage_dic.load(file_name, nets)

    
#     sd.add_info(info)
    
#     sd.garbage_collect()
    return sd

def get_storage_list(nets, path, info):        
    sd_list=[]  
    for net in nets: 
        sd_list.append(get_storage(path+'/'+net, info))
    return sd_list

def get_model_conn_perturbation():
    
    def get_perturbation_dics(c, w_rel):
        d = {}
        for key in c.keys():
            for conn in c[key]:
                u = {key:{'nest':{conn:{'weight':w_rel}}}}
                d = misc.dict_update(d, u)
        return d
                
    c={'FS_FS':['FS_FS_gaba'],
       'FS_MS':['FS_M1_gaba',
                'FS_M2_gaba'],
       'GA_FS':['GA_FS_gaba'],
       'GA_GA':['GA_GA_gaba'],
       'GA_GI':['GA_GI_gaba'],
       'GA_M1':['GA_M1_gaba'],
       'GA_M2':['GA_M2_gaba'],
       'GI_GA':['GI_GA_gaba'],
       'GI_GI':['GI_GI_gaba'],
       'GI_SN':['GI_SN_gaba'],
       'GI_ST':['GI_ST_gaba'],
       'M1_SN':['M1_SN_gaba'],
       'M2_GI':['M2_GI_gaba'],
       'MS_MS':['M1_M1_gaba',
                'M1_M2_gaba',
                'M2_M1_gaba',
                'M2_M2_gaba'],
       'ST_GA':['ST_GA_ampa'],
       'ST_GI':['ST_SN_ampa'],
       }
    mod=[0.75, 1.25,
         0.5, 1.5, 
         0.25, 1.75]
    l=[]
    
    l+=[pl(**{'name':'no_pert'})]
    for w_rel in mod:
        d=get_perturbation_dics(c, w_rel)
        for key, val in d.items():
            l.append(pl(val,'*', **{'name':key+'_pert_'+str(w_rel)}))
        
    return l

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

def mean_rate_slices(data, **k):
    _set=k.pop('set', 0)
    return data[:,_set].get_mean_rate_slices(**k) 


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
        storage_dic.save_dic(d, **{'use_hash':False})
                    
def spike_statistic(data,**k):
    return data.get_spike_stats(**k)
        
class TestModuleFunctions(unittest.TestCase):
    
    def setUp(self):
        self.d=dummy_data_dic()
        self.file_name=HOME+'/tmp/manager_unittest'
        self.main_path=HOME+'/tmp/manager_unittest/'    
        self.kwargs_default_params={}
        
    def test_get_model_conn_perturbation(self):
        l=get_model_conn_perturbation()
        self.assertEqual(len(l), 16*6+1)

#     def test_data_functions(self):
#         
#         for method in [
#                      'firing_rate',
#                      'mean_rates',
#                      'spike_statistic',
#                      ]:
#             mod = __import__(__name__)
#             call=getattr(mod, method)
#             for _, val in misc.dict_iter(self.d):
#                 if not isinstance(val, Data_unit_spk):
#                     continue
#                 _=call(val)  
# 
    def test_get_metwork(self):
        info, nets, builder=get_networks(Builder_network, 
                                         k_builder={'netw':{'size':10}}, 
                                         k_engine={'verbose':False},
                                         k_default_params=self.kwargs_default_params)
        
        self.assertTrue(type(info)==str)
        self.assertTrue(type(nets)==dict)
        self.assertTrue(isinstance(builder, Builder_abstract))
        
    def test_get_input_Go_NoGo(self):
        
        kwargs={'duration':[1000.0, 500.0],
                'input_lists': [['C1', 'FS'], 
                                ['C1', 'C2', 'FS']],
                'other_scenario':True}
        l=get_input_Go_NoGo(kwargs)
        for ll in l[0]:
            for p in ll:
                print p
                     
#     def test_load(self):
#         self.test_save()
#         d1=self.s.load_dic(self.file_name)    
#         self.assertTrue(isinstance(d1,dict))
     
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
              'mean_rates_0',
              'spike_statistic']
         
        dmrs={'intervals':[[0,100], [200, 300], [300, 400],
                           [600,700], [800, 900], [900, 1000]],
              'repetition':2, 'threads':2}
        kwargs_dic={'firing_rate':{'mean_rate_slices':dmrs,
                                   'mean_rate_slices_0':dmrs}}
         
        models=['dummy1','dummy2']
        d = run(Network_mockup)
        d = compute(d, models, attr,**kwargs_dic)
        attr.append('spike_signal')
        self.assertListEqual(d.keys(), ['net1'])
        self.assertListEqual(sorted(d['net1'].keys()), models)
        self.assertListEqual(sorted(d['net1']['dummy1'].keys()), 
                             sorted(attr+['mean_rate_slices', 'mean_rate_slices_0']))
        self.assertListEqual(sorted(d['net1']['dummy2'].keys()), 
                             sorted(attr+['mean_rate_slices', 'mean_rate_slices_0']))                
               
#     def test_save(self):
#         self.s=Storage_dic(self.file_name)
#         save(self.s, self.d)
#         self.assertTrue(os.path.isdir(self.file_name+'/'))    
#         self.assertTrue(os.path.isfile(self.file_name+'.pkl'))    

             
    def tearDown(self):
        if os.path.isfile(self.file_name+'.pkl'):
            os.remove(self.file_name+'.pkl')        
            
            
class TestBuilderMixin(object):
    
    def test_1_perturbations_functions(self):
        v=1
        for method in ['_get_general',
                           '_get_striatal_reversal_potentials',
                           '_get_dopamine_levels',
                           '_get_variable',
                           'get_perturbations',
                           ]:
            call=getattr(self.builder, method)
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
        
        _, nets=director.get_networks({}, {}, {})
        for net in nets.values():
            self.assertTrue(isinstance(net, Network_base))
        
class TestBuilder_network_base(unittest.TestCase):
    def setUp(self):
        self.builder=Builder_network()
        self.builder.kwargs['p_pulses']=[1.]


class TestBuilder_network(TestBuilder_network_base, TestBuilderMixin, 
                          TestDirectorMixin):
    pass

class TestBuilder_single_base(unittest.TestCase):
    def setUp(self):
        self.builder=Builder_network()
        self.builder.kwargs['p_pulses']=[1.]

class TestBuilder_single(TestBuilder_single_base, TestBuilderMixin, 
                          TestDirectorMixin):
    pass


class TestBuilder_inhibition_striatum_base(unittest.TestCase):
    def setUp(self):
        self.builder=Builder_inhibition_striatum()
        self.builder.kwargs['p_pulses']=[1.]

class TestBuilder_inhibition_striatum(TestBuilder_inhibition_striatum_base, TestBuilderMixin, 
                          TestDirectorMixin):
    pass


class TestBuilder_MSN_cluster_compete_base(unittest.TestCase):
    def setUp(self):
        self.builder=Builder_MSN_cluster_compete()
        self.builder.kwargs['p_pulses']=[1.]

class TestBuilder_MSN_cluster_compete(TestBuilder_MSN_cluster_compete_base, 
                                      TestBuilderMixin, 
                                      TestDirectorMixin):
    pass

class TestBuilder_Go_NoGo_with_lesion_FS_base(unittest.TestCase):
    def setUp(self):
        self.builder=Builder_Go_NoGo_with_lesion_FS()
        self.builder.kwargs['p_pulses']=[1.]

class TestBuilder_Go_NoGo_with_lesion_FS(TestBuilder_Go_NoGo_with_lesion_FS_base, 
                                      TestBuilderMixin, 
                                      TestDirectorMixin):
    pass

class TestBuilder_Go_NoGo_with_lesion_FS_ST_pulse_base(unittest.TestCase):
    def setUp(self):
        self.builder=Builder_Go_NoGo_with_lesion_FS_ST_pulse()
        self.builder.kwargs['p_pulses']=[1.]

class TestBuilder_Go_NoGo_with_lesion_FS_ST_pulse(TestBuilder_Go_NoGo_with_lesion_FS_ST_pulse_base, 
                                      TestBuilderMixin, 
                                      TestDirectorMixin):
    pass

if __name__ == '__main__':
    
    test_classes_to_run=[
                        TestModuleFunctions,
                        TestBuilder_network,
                        TestBuilder_single,
                        TestBuilder_inhibition_striatum,
                        TestBuilder_MSN_cluster_compete,
                        TestBuilder_Go_NoGo_with_lesion_FS,
                        TestBuilder_Go_NoGo_with_lesion_FS_ST_pulse,
                        ]
    suites_list = []
    for test_class in test_classes_to_run:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)
    unittest.TextTestRunner(verbosity=2).run(big_suite)  


  