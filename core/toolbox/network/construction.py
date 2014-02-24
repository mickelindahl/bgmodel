'''
Created on Jun 18, 2013

@author: lindahlm
'''
from toolbox.network import structure
from toolbox.network.data_processing import Data_units_dic, Dud_list
from toolbox import my_nest, data_to_disk , misc
from toolbox.my_population import MyPoissonInput, MyInput
from toolbox.misc import Stopwatch, Stop_stdout
import toolbox

from copy import deepcopy
from toolbox.network.default_params import (Inhibition, Slow_wave, Bcpnn_h0,  
                                            Bcpnn_h1, Single_unit, 
                                            Unittest, Perturbation_list)
import nest # Can not be first then I get segmentation Fault
import numpy
import pylab
import time
import unittest
import os, sys


class Network_base(object):
    def __init__(self, name, *args,  **kwargs):
        '''
        Constructor
        '''
        self.calibrated=False
        self.built=False
        self.connected=False
        
        self.class_par=Inhibition
        self.conns=None
        
        self.dud=None
        self.input_class=MyPoissonInput
        self.input_params={} #set in inputs       
        self.name=name
        self.nodes=None
        self.dic_rep=kwargs.get('dic_rep', {})
        self.perturbation=kwargs.get('perturbation', Perturbation_list())
        self._par=None
        self.pops=None
        
        self.reset=kwargs.get('reset',False)
        self.record_activity_attr=kwargs.get('record_activity_attr',[])
        self.record_weights_from=kwargs.get('record_weights_from',[])
        self.replace_perturbation=kwargs.get('replace_perturbation',[])
        self.run_counter=0
           
        self.save_conn= kwargs.get('save_conn', True)
        self.sub_folder=kwargs.get('sub_folder', '')
        self._sim_start=None
        self.sim_started=False
        #self._sim_stop=None
        self.sim_stopped=False
        self.sim_time_progress=0.0
                
        self.stopwatch={}
        
        self.units_list=[]
        self.update_par_rep=kwargs.get('update_par_rep',[]) 
        self.nodes={}
        
        self.verbose=kwargs.get('verbose', 'True')


    @property
    def par(self):
        if self._par==None:
            k={'dic_rep':self.dic_rep, 
               'perturbations': self.perturbation}
            self._par=self.class_par(**k)  
        return self._par
    
    @property
    def path_data(self):
        if self.sub_folder:
            sf=self.sub_folder+'/'
        else:
            sf=''
        return (self.get_data_root_path +self.__class__.__name__ +'/'+sf)
    
    @property
    def path_pictures(self):
        if self.sub_folder:
            sf=self.sub_folder+'-'
        else:
            sf=''
        return (self.get_figure_root_path+self.__class__.__name__ +'-'+sf)
    
    @property
    def path_nest(self):
        return self.par.get_path_nest()

    @property
    def params_popu(self):
        return self.par.get_popu()
   
    @property
    def params_nest(self):
        return self.par.get_nest()
   
    @property
    def params_stru(self):
        return self.par.get_stru()
    
    @property
    def params_conn(self):
        return self.par.get_conn()
      
    
    @property
    def threads(self):
        return self.par.get_threads()
        
    @property
    def sim_time(self):      
        return self.par.get_sim_time()    
     
    @property
    def sim_start(self):
        if not self._sim_start:
            self._sim_start=self.start_rec
        if self.sim_started:
            self._sim_start=self.sim_stop
        return self._sim_start
    
    @sim_start.setter
    def sim_start(self, val):
        self._sim_start=val
        
    @property
    def sim_stop(self):
        return self.par.get_sim_stop()  
        
        
    @property
    def start_rec(self):
        return self.par.get_start_rec()       
    
    @property
    def stop_rec(self):
        return self.par.get_stop_rec()       

    @property
    def fopt(self):
        return self.par.get_fopt()

    @property
    def xopt(self):
        return self.par.get_xopt()

    @property
    def x0opt(self):
        return self.par.get_x0opt()

    def __hash__(self):
        return hash(str(self.par.dic))


    def clear_dud(self):
        self.dud=Data_units_dic() 

    def iter_optimization(self):
        f=self.fopt
        x=self.xopt
        x0=self.x0opt
        for a,b,c in zip(f,x,x0):
            yield a, b, c
        
    def do_calibrate(self):
        self.calibrated=True
        self._do_calibrate()
    
    def _do_calibrate(self):
        raise NotImplementedError
    
    def do_build(self):
        if self.built: return
        if not self.calibrated: self.do_calibrate()
        self._do_build()        
        self.built=True
        
    def _do_build(self):
        raise NotImplementedError
    
    def do_connect(self):
        if self.connected: return
        if not self.built: self.do_build()
        self._do_connect()
        self.connected=True
        
    def _do_connect(self):
        raise NotImplementedError
    
    def do_run(self):
        raise NotImplementedError
    
    def do_reset(self):
        self.built=False
        self.connected=False

    def do_preprocessing(self):
        raise NotImplementedError
    
    def do_postprocessing(self):
        raise NotImplementedError
      
    def do_delete_nest_data(self):
        raise NotImplementedError

    def get(self, attr, *args, **kwargs):
        if hasattr(self.par, 'get_'+attr):
            call=getattr(self.par, 'get_'+attr)
            v=call(*args, **kwargs)
        
        return v
    
    def get_dud(self):
        return self.dud

    def get_data_root_path(self):
        raise NotImplementedError

    def get_figure_root_path(self):
        raise NotImplementedError

    def get_name(self):
        return self.name

    def get_path_data(self):
        return self.path_data

    def get_perturbations(self, stim, stim_name='', op='+'):
        replace_perturbation=[]
        for s in stim:            

            l=[stim_name, float(s), op]
            p=Perturbation_list(stim_name, l)
            replace_perturbation.append(p) 
        return replace_perturbation

    def get_pertubation_list(self):
        return self.perturbation
    
    def get_xopt_length(self):
        return len(self.xopt)

    def get_x0(self):
        return self.x0opt

    def init_optimization(self, x0):
        self.record_activity_attr=['spike_signal']
        self.reset=True
        l=[]
        for x, val in zip(self.xopt, x0):
            l.append([x, val,'='])
        
        p=Perturbation_list('_'.join(self.xopt), l)    
        self.perturbation=p
            
    
    def set_kernel_status(self):
        #@todo: add with stop_stdpout here 
        if not os.path.isdir(self.path_nest):
            msg='No such directory. Need to create {}'.format(self.path_nest)
            raise IOError(msg)
        
        my_nest.SetKernelStatus({'print_time':self.par['simu']['print_time'],
                                 'data_path':self.path_nest, 
                                 'overwrite_files': True})    
 
    def set_perturbation_list(self, val):
        self.perturbation=val 
     

    def set_print_time(self, val):
        self.par.set_print_time(val)
           
    def set_sim_time(self, t):
        self.par.set_sim_time(t)
    
    def set_sim_stop(self, t):
        self.par.set_sim_stop(t)
 
    def simulation_loop(self):
        dud=Data_units_dic()  
        while True:
            self.do_preprocessing()
            self.do_run()
            self.do_postprocessing(dud)

            if self.reset: 
                self.do_reset()
           
            if self.sim_stop<=self.sim_time_progress:
                break    
            
        self.do_delete_nest_data() 
        return dud
        
    def _sim_XX_curve(self, flag, stim=[], stim_time=0, model='',  **kwargs):    
    
        if flag=='currents':
            rp=self.get_perturbations(stim, **kwargs)
            self.replace_perturbation=rp   
            activity_attr=['isis']
        if flag=='rates':
            ru=self.get_perturbations(stim, **kwargs)  
            self.replace_perturbation=ru
            activity_attr=['mean_rate']
        
        self.record_activity_attr=['spike_signal']
                
        self.set_sim_stop(stim_time*len(stim))
        self.set_sim_time(stim_time)
        self.set_print_time(False)
        self.reset=False
          
        dud=self.simulation_loop()
        du=dud.get_model(model)
        for key in activity_attr:
            du.compute_set(key)
            du.set_stimulus(key, 'x', stim)
            
        return dud
    
    def sim_IF_curve(self, **kwargs):
        if 'op' not in kwargs.keys():
            kwargs['op']='+'
        return self._sim_XX_curve(flag='currents', **kwargs)

    def sim_FF_curve(self, **kwargs):
        if 'op' not in kwargs.keys():
            kwargs['op']='='
        return self._sim_XX_curve(flag='rates', **kwargs)

    def sim_optimization(self, x0):
        
        if self.run_counter==0:
            self.init_optimization(x0)
        else:
            for p, val in zip(self.perturbation, x0):
                p.set_val(val)
                  
        if self.sim_time!=self.sim_stop:
            raise RuntimeError(('simulation time and simulation stop needs',
                                ' to be equal'))
        dud=self.simulation_loop()
        
        d=self.pops.get('target_rate')
        dud.set('target_rate', d)
        
        dud.compute_set('mean_rate')
        
        e=dud.get_mean_rate_error(**{'models':self.fopt})
        self.clear_dud() 
        return e
        
class Network_list(object):
    
    def __init__(self, network_list, **kwargs):
        self.dud_returns=['simulation_loop',
                          'sim_IF_curve',
                          'sim_FF_curve'
                          ]
        self.l=network_list
        self.allowed=[
                      'get',
                      'sim_FF_curve']
    @property
    def x_slices(self):
        x=[]
        n, m=0,0
        for net in self:
            m+=net.get_xopt_length()
            x.append(slice(n,m))
            n+=m
        return x
    
    def __getattr__(self, name):
        if name in self.allowed:
            self.attr=name
            return self._caller
        else:
            raise AttributeError(name)            
    
    def __repr__(self):
        return self.__class__.__name__+':'+str([str(l) for l in self])     
                   
    def __iter__(self):
        
        for val in self.l:
            yield val

    def _caller(self, *args, **kwargs):
        a=[]
        
        for obj in self:
            call=getattr(obj, self.attr)
            d=call(*args, **kwargs)
            if d:
                a.append(d) 
        if self.attr in self.dud_returns:
            a=Dud_list(a)              
        return a

    def get_path_data(self):
        return self.l[0].get_path_data()
        
    def append(self, val):
        self.l.append(val)
    
#    def add(self, *a, **k):
#
#        class_name=k.get('class', 'Inhibition')
#        the_class=misc.import_class('toolbox.network.construction.'
#                                    +class_name)
#        self.dic[a[0]]=the_class(*a, **k)
    
    def get_x0(self):
        x0=[]
        for net in self:
            x0+=net.get_x0() 
        return x0
       
    def sim_optimization(self, x):
        e=[]
        for net, x_slice in zip(self, self.x_slices):
            e+=net.sim_optimization(x[x_slice])
        self.run_counter+=1
        return e
        
class Network(Network_base):
    
                        
    def _do_calibrate(self):
        '''
        Possibility to change par.
        '''
        
        with Stop_stdout(not self.verbose), Stopwatch('Calibrating...',
                                                      self.stopwatch):
            pass
        
    def _do_build(self):
        '''
        Build network units as nodes and population. Nodes represent spacial 
        properties and populations holds the nest representation.
        '''
          
        with Stop_stdout(not self.verbose), Stopwatch('Building...',
                                                      self.stopwatch):
            my_nest.ResetKernel(threads=self.threads, print_time=False)  
            print self.par['simu']['sd_params']      
            self.nodes, self.pops=structure.build(self.params_nest, 
                                                      self.params_stru,
                                                      self.params_popu)              
        
        
    def _do_connect(self):
        '''Connect all nodes in the model'''
     
        with Stop_stdout(not self.verbose), Stopwatch('Connecting...',
                                                      self.stopwatch):        
            args=[self.pops, self.nodes, self.params_nest, 
                  self.params_conn, self.save_conn, True]
            
            structure.connect(*args)

    def do_run(self):
        
        if not self.connected: self.do_connect()
       
        if not os.path.isdir(self.path_nest):
            data_to_disk.mkdir(self.path_nest)
        
        with Stop_stdout(not self.verbose), Stopwatch('Simulating...',
                                                      self.stopwatch):        
            self.set_kernel_status()
            my_nest.Simulate(self.sim_time)       
            self.sim_time_progress+=self.sim_time
             
       
    def do_delete_nest_data(self): 
        with Stop_stdout(not self.verbose):
            for filename in os.listdir(self.path_nest):
                if filename.endswith(".gdf"):
                    print 'Deleting: ' +self.path_nest+'/'+filename
                    os.remove(self.path_nest+'/'+filename)
       
    def do_preprocessing(self):
  
        with Stop_stdout(not self.verbose), Stopwatch('Preprocessing...'):  
            if self.update_par_rep:
                self.dic_rep=misc.dict_merge(self.dic_rep, 
                                             self.update_par_rep[0])
                del self.update_par_rep[0]
                
            if self.replace_perturbation:
                self.perturbation = self.replace_perturbation[0]
                del self.replace_perturbation[0]
            
            if self.pops:           
                params_dic=self.par.get_popu_nest_params()
                setup_dic= self.par.get_pop_dic_spike_setup()
                for pop in self.pops:
                    params=params_dic[pop.get_name()]
                    my_nest.SetStatus(pop.ids, [params]*len(pop.ids))
                    
                    if pop.get_name() in setup_dic.keys():
                        spike_setup=setup_dic[pop.get_name()]       
                        
                        for k in spike_setup: 
                            pop.update_spike_times(**k)             
                    
                          
    def do_postprocessing(self, dud):
        
        with Stop_stdout(not self.verbose), Stopwatch('Postprocessing...'):
            for attr in self.record_activity_attr:
                
                # Set data for data units for each data call
                signals_dic=self.pops.get(attr)
                dud.add(attr, signals_dic)  
            
            self.run_counter+=1
           
    def get_simtime_data(self):
        #@todo: fix
        t_total=sum(self.stopwatch.values())
        s='Total time:{1} (sec) cali/built/conn/run (%) {2}/{3}/{4}/{5} '
        
        s=s.format( t_total, 
                    int(100*self.time_calibrated/t_total),
                    int(100*self.time_built/t_total),
                    int(100*self.time_connected/t_total),
                    int(100*self.time_run/t_total))
        return s
      
    def get_spikes_binned(self, models, res, clip=0):
        rdb={} 

        for model in models:
            self.get_spikes(model)
            pop=self.nodes[model].population
            times, rdb[model]=pop.signals['spikes'].raw_data_binned(self.sim_start, self.sim_stop, res, clip)
            
        return rdb

class Inhibition(Network):

    def get_data_root_path(self):
        return toolbox.get_data_root_path('inhibition')

    def get_figure_root_path(self):
        return toolbox.get_figure_root_path('inhibition')
    
              
class Slow_wave(Network):  
    def __init__(self, dic_rep={}, perturbation=None, **kwargs):
        super( Slow_wave, self ).__init__(dic_rep, perturbation, **kwargs)       
        # In order to be able to convert super class object to subclass object   
        self.class_par=Slow_wave

class Single_units_activity(Inhibition):    
    
    def __init__(self,  name, *args, **kwargs):
        super( Single_units_activity, self ).__init__(name, *args, **kwargs)       
        # In order to be able to convert super class object to subclass object        
        self.class_par=kwargs.get('par', Single_unit)
        self.study_name=kwargs.get('study_name', 'M1')
        self.input_models=kwargs.get('input_models', ['C1p', 'FSp', 'M1p', 
                                                      'M2p','GAp'])
    
    @property
    def par(self):
        if self._par==None:
            kwargs={'network_node':self.study_name}
            self._par=self.class_par( self.dic_rep, self.perturbation, 
                                                 **kwargs)  
        
        self._par.per=self.perturbation
        return self._par


#    def FF_curve(self,  freq, tStim, model):
#        return self. _IF_curve(self,  freq, tStim, self.study_name)       
        

    def voltage_trace(self):
        pass
                
         
class Single_units_in_vitro(Network):
    
    
    def __init__(self, dic_rep={}, perturbation=None,  **kwargs):
    
        super( Single_units_in_vitro, self ).__init__( dic_rep, perturbation, 
                                                        **kwargs)       
        # In order to be able to convert super class object to subclass object        
        
        self.model_name=kwargs.get('model_name','default_name')
        self.n=kwargs.get('n', 1)
        d1={'pop_params':{'params':{'V_m':-61.,
                                    'u': float(0) }}}
        d2={'randomization_nodes':{'C_m':False, 'V_th':False, 'V_m':False}}
        d=misc.dict_merge(self.dic_rep, {'simu':{'print_time':False},
                                         'node':{self.model_name:d1},
                                         'netw':d2})
        self.dic_rep=d
            
    @property
    def params_popu(self):
        k=self.model_name
        d=deepcopy(self.par.get_popu()[k])
        d={k:d}
        d[k]['n']=self.n
        d[k]['I_e']=self.params_node[k]['I_vitro']
        return d
   
    @property
    def params_stru(self):
        k=self.model_name
        d=deepcopy(self.par.get_stru()[k])
        d={k:d}
        d[k]['n']=self.n
        return d
   
    @property
    def params_conn(self):
        return {}       
    
    def do_calibrate(self):  
        self.set_kernel_status()         
 

    def IF_curve(self,  currents, tStim, model):
        return self. _XX_curve(self, 'currents', currents, tStim,
                                self.model_name)
   
    
    def IF_variation(self, currents, tStim, randomization=['C_m']):
        #if not self.built: self.do_build()
        #if not self.calibrated: self.do_calibrate()
        
        for key in self.par['netw']['randomization_nodes'].keys():
            if key in randomization:
                d={'netw':{'randomization_nodes':{key:True}}}
                self.dic_rep=misc.dict_merge(self.dic_rep, d)
                #self.par['netw'][key]=True
            else:
                d={'netw':{'randomization_nodes':{key:False}}}
                self.dic_rep=misc.dict_merge(self.dic_rep, d)
                #self.dic_rep.update({'netw':{key:False}})
                #self.par['netw'][key]=True
                
        #self.do_randomize_params(randomization)
        return self.IF_curve(currents, tStim)    
    
    def IV_curve(self,  currents, tStim):    
        if not self.built: self.do_build()
        if not self.calibrated: self.do_calibrate()
  
        pop=self.pops[self.model_name]
        mm_params={'interval':0.1, 'start':self.par['simu']['start_rec'],
                   'stop':self.par['simu']['stop_rec'], 
                   'record_from':['V_m']}       
        pop.set_mm(True, mm_params)
        
        data=[]
        for pop_id in sorted(pop.ids):
            data.append(pop.IV_I_clamp(currents, pop_id, tStim))  
        
        data=numpy.array(data)
        
        currents=data[:,0][0]
        voltage=data[:,1][0]   
        return currents, voltage
    

    def voltage_respose_curve(self, currents, times, start, stop):
        if not self.built: self.do_build()
        if not self.calibrated: self.do_calibrate()
        
        pop=self.nodes[self.model_name].population
        mm_params={'interval':0.1, 'start':self.sim_start,  'stop':self.sim_stop, 
                   'record_from':['V_m']}       
        pop.set_mm(True, mm_params)
        
        data=[]
        for pop_id in sorted(pop.ids):
            data.append(pop.voltage_response(currents, times, start, stop, pop_id))
        
        if self.n==1:idx=0
        else:idx=slice(0,self.n)
        
        data=numpy.array(data)
        times=data[:,0][idx]
        voltages=data[:,1][idx]  
        return times, voltages

class Unittest_net(Network):    
    
    def __init__(self,  name, *args, **kwargs):
        super( Unittest_net, self ).__init__(name, *args, **kwargs)       
        # In order to be able to convert super class object to subclass object   
        self.class_par=Unittest
        
    @property
    def path_data(self):        
        return ('/afs/nada.kth.se/home/w/u1yxbcfw/results/papers/unittest'
                +'/'+self.name +'/')
    
    @property
    def path_pictures(self):
        return ('/afs/nada.kth.se/home/w/u1yxbcfw/projects/papers/unittest'
                +'/pictures'+'/'+self.name +'-')
           
class Bcpnn_h0(Inhibition):    
    
    def __init__(self,  dic_rep={}, perturbation=None, **kwargs):
        super( Bcpnn_h0, self ).__init__(dic_rep, perturbation, **kwargs)       
        # In order to be able to convert super class object to subclass object   
        self.class_par=Bcpnn_h0
        
    @property
    def path_data(self):        
        return '/afs/nada.kth.se/home/w/u1yxbcfw/results/papers/bcpnn'+'/'+self.name +'/'
    
    @property
    def path_pictures(self):
        return '/afs/nada.kth.se/home/w/u1yxbcfw/projects/papers/bcpnnbg/pictures'+'/'+self.name +'-'
        
class Bcpnn_h1(Bcpnn_h0):    
    
    def __init__(self,  dic_rep={}, perturbation=None, **kwargs):
        super( Bcpnn_h1, self ).__init__(dic_rep, perturbation, **kwargs)       
        # In order to be able to convert super class object to subclass object   
        self.class_par=Bcpnn_h1
 
 
 

def network_kwargs():
    kwargs={'save_conn':False, 
            'sub_folder':'unit_testing', 
            'verbose':False,
            'record_activity_attr':['spike_signal'],
            'record_weights_from':[]}
    return kwargs     
 
class TestUnittest(unittest.TestCase):
    kwargs=network_kwargs()
    curr=[400,500,700]
    rates=range(2000, 3500, 500)
    def update_par_rep(self):
        update_par_rep=[]
        for c in self.curr:
            dic_rep={'node':{'n2':{'I_vivo':float(c)}}}
            update_par_rep.append(dic_rep) 
        return update_par_rep
    
    def replace_perturbation(self):
        replace_perturbation=[]
        for c in  self.curr:
            
            p=Perturbation_list('Curr', ['node.n2.pop_params.params.I_e', 
                                         float(c), '+'])
            replace_perturbation.append(p) 
        return replace_perturbation
         
    
    
    def setUp(self):
        
        self.class_network=Unittest_net
        #self.fileName=self.class_network().path_data+'network'
        self.model_list=['n1']
        dic_rep={}
        dic_rep.update({'simu':{'sim_stop':3000.0,
                             'mm_params':{'to_file':False, 'to_memory':True},
                             'sd_params':{'to_file':False, 'to_memory':True}},
                             'netw':{'size':9.},
                             'node':{'n1':{'mm':False,
                                           'sd':True}}})
        self.kwargs.update({'dic_rep':dic_rep})
        self.dic_rep=dic_rep
        self.name='net1'
        
  
    def test_1_build(self):
        network=self.class_network(self.name, **self.kwargs)
        network.do_build()
     
    def test_2_connect(self):
        network=self.class_network(self.name, **self.kwargs)
        network.do_connect()    

    def test_3_run(self):
        network=self.class_network(self.name, **self.kwargs)
        network.do_run()  
        
    def test_4_reset(self):
        network=self.class_network(self.name, **self.kwargs)  
        network.do_run()
        network.do_reset()
        network.do_run()

    def test_5_simulation_loop(self):
        
        dic_rep=deepcopy(self.dic_rep)
        dic_rep['node']['n1']['nest_params']={'I_e':700.0}
        self.kwargs.update({'dic_rep':dic_rep})
        network=self.class_network(self.name, **self.kwargs)  
        network.reset=False
        network.set_sim_stop(1000.0)     
        dud=network.simulation_loop()    
        pylab.figure()   
        dud['n1'].compute_set('firing_rate',*[1],**{'average':True})
        dud['n1'].plot_firing_rate(**{'win':100})  
        n_sets=len(network.pops['n1'].sets)
        dud['n1'].plot_firing_rate(**{'win':100, 'sets':range(n_sets)})      
        pylab.show()
#    
#    def test_6_simulation_loop_x3_update_par_rep(self):
#        kwargs=deepcopy(self.kwargs)
#        kwargs.update({'update_par_rep':self.update_par_rep()})
#        network=self.class_network(self.name, **kwargs)  
#        network.reset=False
#        network.simulation_loop()
#        pylab.figure()
#        network.dud['n2'].plot_firing_rate(pylab.subplot(221), **{'win':100})
#        
#        kwargs.update({'update_par_rep':self.update_par_rep()})
#        network=self.class_network(self.name, **kwargs)  
#        network.reset=True
#        network.simulation_loop()       
#        network.dud['n2'].plot_firing_rate(pylab.subplot(222),**{'win':100})
#        
#        network.dud['n2'].plot_hist_isis(pylab.subplot(223))
#        #pylab.show()    
#
#    def test_7_simulation_loop_x3_replace_pertubations(self):
#        kwargs=deepcopy(self.kwargs)
#        kwargs.update({'replace_perturbation':self.replace_perturbation()})
#        network=self.class_network(self.name, **kwargs)  
#        network.reset=False
#        network.simulation_loop()
#        pylab.figure()
#        network.dud['n2'].plot_firing_rate(pylab.subplot(221), **{'win':100})
#        
#        
#        kwargs.update({'replace_perturbation':self.replace_perturbation()})
#        network=self.class_network(self.name, **kwargs)  
#        network.reset=True
#        network.simulation_loop()
#        network.dud['n2'].plot_firing_rate(pylab.subplot(222),**{'win':100})
#        
#        network.dud['n2'].plot_hist_isis(pylab.subplot(223))
#        #pylab.show()  
#                      
#    def test_90_IF_curve(self):
#        network=self.class_network(self.name, **self.kwargs)  
#        kwargs={'stim':self.curr,
#                'stim_name':'node.n2.pop_params.params.I_e',
#                'stim_time':500.0,
#                'model':'n2'}
#        dud=network.sim_IF_curve(**kwargs)
#        pylab.figure()
#        
#        dud['n2'].plot_IF_curve()
#        #pylab.show() 
                       
#    def test_91_FF_curve(self):
#        dic_rep={'simu':{'sd_params':{'to_file':False, 'to_memory':True}},
#                         'netw':{'size':36.}}
#        self.kwargs.update({'dic_rep':dic_rep})
#        network=self.class_network(self.name, **self.kwargs)  
#        kwargs={'stim':self.rates,
#                'stim_name':'node.n1.rate',
#                'stim_time':3000.0,
#                'model':'n2'}
#        dud=network.sim_FF_curve(**kwargs)
#        pylab.figure()
#        dud['n2'].plot_FF_curve()
#        pylab.show()
##
#    def test_92_voltage_trace(self):
#        dic_rep=deepcopy(self.dic_rep)
#        dic_rep['node']['n2']['pop_params']['mm']=True
#        self.kwargs.update({'dic_rep':dic_rep})
#        network=self.class_network(self.name, **self.kwargs)  
#        pr=network.get_perturbations(self.rates, 'node.n1.rate', '=')
#        network.replace_perturbation=pr
#        network.reset=False
#        network.simulation_loop()
#        pylab.figure()
#        network.dud['n2'].plot_voltage_trace()
#        #pylab.show()
#        
#    def test_93_optimization(self):
#        dic_rep=deepcopy(self.dic_rep)
#        
#        x0=[2580.0]
#        opt={'f':['n2'],
#             'x':['node.n1.rate'],
#             'x0':x0}
#        dic_rep.update({'simu':{'sim_stop':1000.0,
#                                'sim_time':1000.0},
#                        'netw':{'optimization':opt}})
#        self.kwargs.update({'dic_rep':dic_rep})
#        network=self.class_network(self.name, **self.kwargs)  
# 
#        e1=network.sim_optimization(x0)
#        e2=network.sim_optimization([3000.0])
#        
#        self.assertAlmostEqual(e1[0], 0, delta=3)
#        self.assertAlmostEqual(e2[0], 16.6, delta=2)
#        #pylab.show()
        
class TestNetwork_dic(unittest.TestCase):
    def setUp(self):
        self.x0=[3000.0, 3000.0]
        opt={'f':['n2', 'n3'],
             'x':['node.n1.rate', 'node.n4.rate'],
             'x0':self.x0}
        
        dic_rep={}
        dic_rep.update({'simu':{'sim_stop':1000.0,
                                'stim_time':1000.0,
                             'mm_params':{'to_file':False, 'to_memory':True},
                             'sd_params':{'to_file':False, 'to_memory':True}},
                             'netw':{'size':20.,
                                     'optimization':opt}})

        k={'class':'Unittest',
            'save_conn':False, 
            'sub_folder':'unit_testing', 
            'verbose':False,
            'dic_rep':dic_rep}
        self.network_kwargs=k
        self.nd=Network_list()
        #nd.add(name, **kwargs)
        self.kwargs={'models':self.nd}
    
    def test_x_slice(self):
        self.nd.add('net1', **self.network_kwargs) 
        #self.nd.sim_optimization([3000, 3000])
        s=self.nd.x_slices
        self.assertEqual(s,[slice(0,2)])
        
        self.nd.add('net2', **self.network_kwargs) 
        s=self.nd.x_slices
        self.assertEqual(s,[slice(0,2),slice(2,4)]) 
     
    def test_sim_optimization(self):
        self.nd.add('net1', **self.network_kwargs) 
        e1=self.nd.sim_optimization([3000., 3000.])
       
        self.nd.add('net2', **self.network_kwargs) 
        e2=self.nd.sim_optimization([3000., 3000., 3100., 3100.])
        
        self.assertAlmostEquals(sum(e1),sum(e2[0:2]), delta=0.1)
        self.assertFalse(sum(e1)==sum(e2))
                           
class TestStructureInhibition(unittest.TestCase):
    
    nest.sr("M_WARNING setverbosity") #silence nest output
    
    
    
    def do_reset(self):
        self.kwargs={'save_conn':False, 'sub_folder':'unit_testing', 'verbose':True}
        self.dic_rep={'simu':{'start_rec':1.0, 'stop':100.0,'sim_time':100., 
                              'sd_params':{'to_file':True, 'to_memory':False},
                              'threads':4, 'print_time':False},
             'netw':{'size':500.0},
             }
        
    def change_fan_in(self):
        self.dic_rep.update({'conn':{'M1_M1_gaba':{'fan_in0':5},
                                     'M1_M2_gaba':{'fan_in0':5},
                                     'M2_M1_gaba':{'fan_in0':5},
                                     'M2_M2_gaba':{'fan_in0':5},
                                     'M1_SN_gaba':{'fan_in0':5},
                                     'M2_GI_gaba':{'fan_in0':5}}})
        
    def build_connect(self):
        network=self.class_network(self.name, **self.kwargs)
        
        for key, val in network.par['conn'].items():
            #print key
            if val['delay_setup']['type']=='uniform':
                val['delay_setup']['type']='constant'
                val['delay_setup']['params']=(val['delay_setup']['params']['max']+val['delay_setup']['params']['min'])/2.
        
        network.do_build()
        network.do_connect()
        return network 
    
    def setUp(self):
        self.do_reset()
        self.change_fan_in()
        self.dic_rep['netw'].update({'size':50.0, 'sub_sampling':{'M1':20.0,'M2':20.0}})
        self.class_network=Inhibition
        
        self.fileName=self.class_network().path_data+'network'
        self.model_list=['M1','M2', 'FS', 'GA', 'GI', 'ST', 'SN']
            
    
    def test_a_build_connect_run(self):
        
        network=self.build_connect()
        network.do_run()
        data_to_disk.pickle_save(network, self.fileName)
        
        
    def test_reset(self):
        network=self.build_connect()    
        network.do_run()
        network.do_reset()
        network.do_run()
        self.assertEqual(network.start_rec, network.sim_start)
        
    def test_get_firing_rate(self):
        
        network=data_to_disk.pickle_load(self.fileName)
        my_nest.SetKernelStatus({'data_path':network.path_nest})
        fr=network.get_firing_rate(self.model_list)
        
        for k in self.model_list:
            self.assertEqual(len(fr[k]['times']), network.par['simu']['sim_time']-1)
            self.assertEqual(len(fr[k]['rates']), network.par['simu']['sim_time']-1)

    def test_get_firing_rate_sets(self):
        
        network=data_to_disk.pickle_load(self.fileName)
        my_nest.SetKernelStatus({'data_path':network.path_nest})
        d=network.get_firing_rate_sets(self.model_list)
        
        for k in self.model_list:
            self.assertEqual(len(d[k]['times']), network.par['simu']['sim_time']-1)
            self.assertEqual(d[k]['rates'].shape[1], network.par['simu']['sim_time']-1)
            self.assertEqual(d[k]['rates'].shape[0], network.par['node'][k]['n_sets'])

    def test_get_raster(self):
        
        network=data_to_disk.pickle_load(self.fileName)
        my_nest.SetKernelStatus({'data_path':network.path_nest})
        d=network.get_rasters(self.model_list)
        
        for k in self.model_list:
            self.assertIsInstance(d[k][0], numpy.ndarray)
            self.assertEqual(d[k][0].shape[0], 2)
            self.assertListEqual(list(d[k][1]), range(network.par['node'][k]['n']))

    def test_get_rasters_sets(self):
        
        network=data_to_disk.pickle_load(self.fileName)
        my_nest.SetKernelStatus({'data_path':network.path_nest})
        d=network.get_rasters_sets(self.model_list)
        
        for k in self.model_list:
            n_sets=network.par['node'][k]['n_sets']
            n=network.par['node'][k]['n']
            self.assertEqual(len(d[k]), n_sets)
            i=0
            acum=0
            for l in d[k]:
                self.assertIsInstance(l[0], numpy.ndarray)
                self.assertEqual(l[0].shape[0], 2)
                self.assertEqual(len(l[0][0]),len(l[0][1]))
                
                
                m=len(list(range(i, n, n_sets)))
                self.assertListEqual(list(l[1]), range(acum, acum+m))
                i+=1
                acum+=m


class TestStructureSingle_units_activity(TestStructureInhibition):
    def setUp(self):
        self.do_reset()
        self.class_network=Single_units_activity    
        self.dic_rep['netw'].update({'size':10})  
        #for key in ['M1_M1_gaba', 'M1_M2_gaba', 'M2_M1_gaba', 'M2_M2_gaba']:
        #    del self.dic_rep['conn'][key]
        self.fileName=self.class_network().path_data+'network'             
        self.model_list=['M1']
        
class TestStructureSlow_wave(TestStructureInhibition):
    
    def setUp(self):
        self.do_reset()
        self.change_fan_in() 
        self.dic_rep['netw'].update({'size':50.0, 'sub_sampling':{'M1':20.0,'M2':20.0}}) 
        self.class_network=Slow_wave
        
        self.fileName=self.class_network().path_data+'network'
        self.model_list=['M1','M2', 'FS', 'GA', 'GI', 'ST', 'SN']
        
class TestStructureBcpnn_h0(TestStructureInhibition):
    
    def setUp(self):
        self.do_reset()
        self.change_fan_in()
        self.dic_rep['netw'].update({'sub_sampling':{'M1':40.0,'M2':40.0, 'CO':600.0},
                                      'size':100.})     
        self.class_network=Bcpnn_h0
        
        self.fileName=self.class_network().path_data+'network'
        self.model_list=['CO', 'M1','M2', 'FS', 'GA', 'GI', 'ST', 'SN']
            
class TestStructureBcpnn_h1(TestStructureBcpnn_h0):

    def setUp(self):
        self.do_reset()
        self.change_fan_in()
        self.dic_rep['netw'].update({'sub_sampling':{'M1':40.0,'M2':40.0, 'CO':600.0},
                                      'size':100.})      
        self.class_network=Bcpnn_h1        
        self.fileName=self.class_network().path_data+'network'
        self.model_list=['CO', 'M1','M2', 'F1', 'F2', 'GA', 'GI', 'ST', 'SN']



          
if __name__ == '__main__':
    
    test_classes_to_run=[
                         TestUnittest,
#                         TestUnittest_bcpnn,
#                         TestNetwork_dic,
                       ]
    suites_list = []
    for test_class in test_classes_to_run:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)
    unittest.TextTestRunner(verbosity=2).run(big_suite)  
    #suite = unittest.TestLoader().loadTestsFromTestCase(TestStructureBcpnn_h0)
    #suite = unittest.TestLoader().loadTestsFromTestCase(TestStructureSlow_wave)
    #suite = unittest.TestLoader().loadTestsFromTestCase(TestStructureInhibition)
    #suite = unittest.TestLoader().loadTestsFromTestCase(TestStructureSingle_units_activity)
    
    #suite = unittest.TestSuite()
    #suite.addTest(TestStructureSlow_wave('test_build_connect_run'))
    
    #unittest.TextTestRunner(verbosity=2).run(suite)
    #unittest.main()             
    
    
    
        