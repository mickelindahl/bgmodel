'''
Created on Apr 18, 2014

@author: lindahlm
'''
import itertools
import numpy
import os
import unittest

from copy import deepcopy        
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
                                            Slow_wave,
                                            Slow_wave2)

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
#             perturbation_consistency(p, par)
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

    def get_parameters(self, per):
        
#         per.get_
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
    
   
    def get_parameters(self, per):
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

    def get_parameters(self, per):
        return Slow_wave(**{'other':Inhibition(),
                            'perturbations':per})
     
class Builder_slow_wave(Builder_slow_wave_base, 
                      Mixin_dopamine, 
                      Mixin_general_network, 
                      Mixin_reversal_potential_striatum):
    pass

class Builder_slow_wave2_base(Builder_abstract):    
      
    def _get_striatal_reversal_potentials(self):
        return [self._low()]    

    def get_parameters(self, per):
        return Slow_wave2(**{'other':Inhibition(),
                            'perturbations':per})
     
class Builder_slow_wave2(Builder_slow_wave2_base, 
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

def get_input_Go_NoGo(kwargs):
    n_sets=kwargs.get('n_sets', 2)
    
    res = kwargs.get('resolution', 5)
    rep = kwargs.get('repetition', 5)
    input_lists = kwargs.get('input_lists', [['C1'], ['C1', 'C2']])
    v = numpy.linspace(1, 3, res)
    x, y = numpy.meshgrid(v, v)
    x, y = x.ravel(), y.ravel()
    p0 = reduce(lambda x, y:x + y, [[1, p] for p in y])
    p1 = reduce(lambda x, y:x + y, [[1, p] for p in x])
    durations = [900., 100.] * res * res
    l = []
    for inp_list in input_lists:
        d = {}
        for node in ['M1', 'M2', 'GI', 'SN']:
            d = misc.dict_update(d, {'node':{node:{'n_sets':2}}})
        
        for conn, rule in zip(['M1_SN_gaba', 'M2_GI_gaba', 'GI_SN_gaba'], 
            ['set-set', 'set-set', 'all-all']):
            d = misc.dict_update(d, {'conn':{conn:{'rule':rule}}})

        for inp in inp_list :
            if inp not in ['C1', 'C2']:
                continue
            d = misc.dict_update(d, {'node':{inp:{'n_sets':n_sets}}})
            params = {'type':'burst3', 
                'params':{'n_set_pre':n_sets, 
                    'repetitions':rep}}
            d_sets = {}
            d_sets.update({str(0):{'active':True, 
                                   'amplitudes':p0, 
                                   'durations':durations, 
                                   'proportion_connected':1}, 
                           
                           str(1):{'active':True, 
                                   'amplitudes':p1, 
                                   'durations':durations, 
                                   'proportion_connected':1}})
            
            params['params'].update({'params_sets':d_sets})
            d = misc.dict_update(d, {'netw':{'input':{inp:params}}})

        for inp in inp_list :
            if inp in ['C1', 'C2']:
                continue
            
            d = misc.dict_update(d, {'node':{inp:{'n_sets':n_sets}}})
            params = {'type':'burst3', 
                      'params':{'n_set_pre':n_sets, 
                                'repetitions':rep}}
            d_sets = {}
            d_sets.update({str(0):{'active':True, 
                                   'amplitudes':p0+p1, 
                                   'durations':durations, 
                                   'proportion_connected':1}})
            
            params['params'].update({'params_sets':d_sets})
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

    def get_parameters(self, per):
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


def add_lesions_Go_NoGo(l):
    lesions = [{'conn':{'M1_M1_gaba':{'lesion':True}, 
                        'M1_M2_gaba':{'lesion':True}, 
                        'M2_M1_gaba':{'lesion':True}, 
                        'M2_M2_gaba':{'lesion':True}}}, 
                {'conn':{'FS_M1_gaba':{'lesion':True}, 
                        'FS_M2_gaba':{'lesion':True}}}, 
                {'conn':{'GA_M1_gaba':{'lesion':True}, 
                        'GA_M2_gaba':{'lesion':True}}}]
    names = ['no-MS-MS', 'no-FS', 'no_GP']
    for lesion, name in zip(lesions, names):
        l += [deepcopy(l[1])]
        l[-1] += pl(lesion, 
            '=', **
            {'name':name})
    
    return l

class Builder_Go_NoGo_with_lesion_base(Builder_network):    

    def get_parameters(self, per):
        return Go_NoGo_compete(**{'other':Inhibition(),
                       'perturbations':per})


    def _get_dopamine_levels(self):
        return [self._dop()]    
    
    def _variable(self):
        
        l, self.dic = get_input_Go_NoGo(self.kwargs)      
        l = add_lesions_Go_NoGo(l)             
        return l    


class Builder_Go_NoGo_with_lesion(Builder_Go_NoGo_with_lesion_base, 
                      Mixin_dopamine, 
                      Mixin_general_network, 
                      Mixin_reversal_potential_striatum):
    pass     


  
        
        
class Builder_Go_NoGo_with_lesion_FS_base(Builder_network):    

    def get_parameters(self, per):
        return Go_NoGo_compete(**{'other':Inhibition(),
                       'perturbations':per})


    def _get_dopamine_levels(self):
        return [self._dop()]    
    
    def _variable(self):
        
        l, self.dic = get_input_Go_NoGo(self.kwargs)      
        l = add_lesions_Go_NoGo(l)    
        
        for ll in l:
            ll+=pl()
        
        return l    


class Builder_Go_NoGo_with_lesion_FS(Builder_Go_NoGo_with_lesion_FS_base, 
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
    if 'mean_rate_slices' in kwargs.keys():            
        kwargs=kwargs['mean_rate_slices']
        if _set!=None:
            update_kwargs_with_set_dic(kwargs, _set)
            
        u=obj.get_mean_rate_slices(**kwargs)
        key[-1]='mean_rate_slices' 
        dout=misc.dict_recursive_add(dout, key, u)
             


def update_kwargs_with_set_dic(k, s):
    name='set_'+str(s)
    if name in k.keys():
        k.update(k[name])
    k['set'] = s

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
            
            k=kwargs_dic.get(a, {}).copy()
            sets=k.pop('sets', None)
            
            if sets!=None:
                
                max_sets=val.wrap.shape[1]
                
                sets=[s for s in sets if s<max_sets ]
                
                for s in sets:

                    update_kwargs_with_set_dic(k, s)
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
        self.assertTrue(type(nets)==dict)
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
        
        dmrs={'intervals':[[0,100], [200, 300], [300, 400],
                         [600,700], [800, 900], [900, 1000]],
              'repetition':2, 'threads':2}
        kwargs_dic={'firing_rate':{'mean_rate_slices':dmrs}}
        
        models=['dummy1','dummy2']
        d = run(Network_mockup)
        d = compute(d, models, attr,**kwargs_dic)
        attr.append('spike_signal')
        self.assertListEqual(d.keys(), ['net1'])
        self.assertListEqual(sorted(d['net1'].keys()), models)
        self.assertListEqual(sorted(d['net1']['dummy1'].keys()), sorted(attr+['mean_rate_slices']))
        self.assertListEqual(sorted(d['net1']['dummy2'].keys()), sorted(attr+['mean_rate_slices']))                
        
     
    def test_save(self):
        self.s=Storage_dic(self.file_name)
        save(self.s, self.d)
        self.assertTrue(os.path.isdir(self.file_name+'/'))    
        self.assertTrue(os.path.isfile(self.file_name+'.pkl'))    

             
    def tearDown(self):
        if os.path.isfile(self.file_name+'.pkl'):
            os.remove(self.file_name+'.pkl')        
        
  