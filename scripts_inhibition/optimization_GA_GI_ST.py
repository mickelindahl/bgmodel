'''
Created on Jun 4, 2015

@author: mikael
'''

import oscillation_perturbations_new_beginning_slow4 as op
     
from simulate import (get_file_name, get_file_name_figs,
                      get_path_nest)
from scripts_inhibition.oscillation_common import mallet2008

from toolbox.network.manager import (get_networks, 
                                     add_perturbations, 
                                     get_storage_list, 
                                     save, load,)
# from toolbox.network.manager import Builder_GA_GI_ST as builder
from toolbox.network.manager import Builder_beta as builder
from toolbox import misc
from toolbox.network.default_params import Perturbation_list as pl
from toolbox.network.engine import Network_list
from toolbox.network.optimization import Fmin
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
        #         self.sim_time=k.get('sim_time')
#         self.sim_stop=k.get('sim_time')
#         self.size=k.get('size') #only using TI, TA and ST for such a network
        
        
    def builder(self):
        return {}
    
    def default_params(self):
        d={}
        return d
    
    def director(self):
        return {'nets_to_run':self.nets_to_run}  
       
    def engine(self):
        d={'record':['spike_signal'],
           'verbose':False}        
        return d

    def par(self):
        d={'simu':{
#                    'local_num_threads':4,
                   'print_time':False,
#                    'sim_time':self.sim_time,
#                    'sim_stop':self.sim_stop,
                   'start_rec':1000.0
                   },
#                    'mm_params':{'to_file':False, 'to_memory':False},
#                    'sd_params':{'to_file':False, 'to_memory':True},
           'netw':{
#                    'size':self.size,
                   'optimization':self.opt_setups()[self.tp_name]}}
        
        misc.dict_update(d,self.target_perturbations()[self.tp_name])
        return d

    def target_perturbations(self):
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
        return out

    def opt_setups(self):
        out={}
        d={'f':['GI', 'ST'],
           'x':['node.EI.rate', 'node.CS.rate'],
           'x0':[1700, 230]}
        out['beta_control']=d
        
        d={'f':['GI', 'GA', 'ST'],
           'x':['node.EI.rate','node.EA.rate', 'node.CS.rate'],
           'x0':[1700, 200, 230]}
        out['beta_lesion']=d
        
        d={'f':['GI', 'ST'],
           'x':['node.EI.rate', 'node.CS.rate'],
           'x0':[1000, 180]}
        out['sv_control']=d
        
        d={'f':['GI', 'GA','ST'],
           'x':['node.EI.rate','node.EA.rate', 'node.CS.rate'],
           'x0':[800, 100, 180]}
        out['sw_lesion']=d
        
        return out  
    
#     def osc(self):
#         d={'node':{'C1':{'rate':1.1},
#                    'C2':{'rate':1.1},
#                    'CF':{'rate':1.1}}}
#         return d
        
         
def to_perturbation_list(d, name=''):
    return pl(d, '=', **{'name':name})
    


def main(*args, **kwargs):
    setup=kwargs.get( 'setup')#, Setup())
    script_name=kwargs.get('script_name')#,__file__.split('/')[-1][0:-3])
    builder=kwargs.get('builder')
    info, nets, _ = get_networks(builder, 
                                 setup.builder(), 
                                 setup.director(),
                                 setup.engine(),
                                 setup.default_params())
    
     
    perturbation_list=kwargs.get('perturbation_list')#,op.get()[0])
    perturbation_list+=pl(setup.par(), '=', **{'name':setup.tp_name})
#     perturbation_list+=pl(setup.osc(), '*')
    
    print perturbation_list
    
    for p in sorted(perturbation_list.list):
        print p
    
    add_perturbations(perturbation_list, nets)
    
    key=nets.keys()[0]
    file_name = get_file_name(script_name, nets[key].par)
#     file_name_figs = get_file_name_figs(script_name,  nets[key].par)
    path_nest=get_path_nest(script_name, nets.keys(), nets[key].par)
    
    print nets.keys()
    
    for net in nets.values():
        net.set_path_nest(path_nest)
    
    # Adding nets no file name
    sd_list=get_storage_list(nets.keys(), file_name, info)
    
    for net, sd in zip(nets, sd_list):

        nl=Network_list(nets.values())
        #nd.add(name, **kwargs)
        
        kwargs={'model':nl,
                'call_get_x0':'get_x0',
                'call_get_error':'sim_optimization', 
                'verbose':True,}
        
        f=Fmin('GA_GI_ST_opt', **kwargs)
        h=f._fmin() 
        
        dd={setup.tp_name.split('_')[0]:{net:{'opt':h}}}
        save(save(sd, dd))

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

