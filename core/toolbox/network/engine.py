'''
Created on Jun 18, 2013

@author: lindahlm
'''
import numpy
import os
import pylab
import unittest
import sys
import pprint
pp=pprint.pprint

from copy import deepcopy

from toolbox import my_nest, data_to_disk , misc
from toolbox.misc import Stopwatch, Stop_stdout
from toolbox.my_signals import Data_generic
from toolbox.network import structure
from toolbox.network.data_processing import (Data_unit_spk,
                                             Data_unit_vm,
                                             Dud_list)
from toolbox.network.default_params import (Inhibition, Slow_wave, Bcpnn_h0,  
                                            Bcpnn_h1, Single_unit, 
                                            Unittest, Unittest_extend, 
                                            Perturbation,
                                            Perturbation_list,
                                            Unittest_bcpnn_dopa, Unittest_stdp,
                                            Unittest_bcpnn)



class Network_base(object):
    def __init__(self, name, *args,  **kwargs):
        '''
        Constructor
        '''
        self.calibrated=False
        self.built=False
        self.connected=False
        
        self.name=name
        self.surfs=None

        self.par=kwargs.get('par', Unittest())
        
        self.pops=None
        
        self.reset=kwargs.get('reset',False)
        self.record=kwargs.get('record',  ['spike_signal'])
        self.record_weights_from=kwargs.get('record_weights_from',[])
        self.replace_perturbation=kwargs.get('replace_perturbation',[])
        self.run_counter=0
           
        self.save_conn= kwargs.get('save_conn', True)
        self.sub_folder=kwargs.get('sub_folder', '')
        self._sim_start=None
        self.sim_started=False
        self._sim_stop=None
        self.sim_stopped=False
        self.sim_time_progress=0.0
                
        self.stopwatch={}
        self.update_par_rep=kwargs.get('update_par_rep',[]) 
        
        self.verbose=kwargs.get('verbose', 'True')

    def __repr__(self):
        return self.__class__.__name__+':'+self.name   
        
    @property
    def path_data(self):
        return self.par.get_path_data()
    
    @property
    def path_pictures(self):
        return self.par.get_path_figure()
    
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
    def params_surfs(self):
        return self.par.get_surf()
    
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


    def get_data_root_path(self):
        raise NotImplementedError

    def get_figure_root_path(self):
        raise NotImplementedError

    def get_name(self):
        return self.name

    def get_print_time(self):
        return self.par['simu']['print_time']
    def get_path_data(self):
        return self.path_data

    def get_perturbations(self, stim, stim_name='', op='+', **kwargs):
        replace_perturbation=[]
        for s in stim:            

#             l=[stim_name, float(s), op]
            keys=stim_name.split('.')
            d=misc.dict_recursive_add({}, keys, s)
            p=Perturbation_list(d, op, **{'name':stim_name})
            replace_perturbation.append(p) 
        
        return replace_perturbation

    def get_pertubation_list(self):
        return self.perturbation

    def get_perturbations_optimization(self,x0):
        l=[]
        for x, val in zip(self.xopt, x0):
            keys=x.split('.')
            p=Perturbation(keys, val, '=')
            l.append(p)
            
        return l

    def set_replace_pertubation(self, val):
        self.replace_perturbation=val

    def get_sim_time(self):
        return self.par.get_sim_time()
    
    def get_start_rec(self):
        return self.par.get_start_rec()
    
    
    def get_sim_stop(self):
        return self.par.get_sim_stop()
    
    def get_single_unit(self):
        try:
            return self.par.get_single_unit()
        except Exception as e:
            s='\nparameter class {} do not have single_unit'
            s=s.format(self.par)
            raise type(e)(e.message + s), None, sys.exc_info()[2]

    def get_single_unit_input(self):
        try:
            return self.par.get_single_unit_input()
        except Exception as e:
            s='\nparameter class {} do not have single_unit'
            s=s.format(self.par)
            raise type(e)(e.message + s), None, sys.exc_info()[2]
        
    def get_xopt_length(self):
        return len(self.xopt)

    def get_x0(self):
        return self.x0opt


    def init_optimization(self):
        self.record=['spike_signal']
        self.reset=True
        

            
    def set_kernel_status(self):
        #@todo: add with stop_stdpout here 
        if not os.path.isdir(self.path_nest):
            msg='No such directory. Need to create {}'.format(self.path_nest)
            raise IOError(msg)
        
        my_nest.SetKernelStatus({'print_time':self.get_print_time(),
                                 'data_path':self.path_nest, 
                                 'overwrite_files': True})    
 
    def set_perturbation_list(self, val):
        self.par.set_perturbation_list(val)   

    def set_print_time(self, val):
        self.par.set_print_time(val)
           
    def set_sim_time(self, t):
        self.par.set_sim_time(t)
    
    def set_sim_stop(self, t):
        self.par.set_sim_stop(t)
    
    def simulation_loop(self):
        d={}
        self.do_reset()
        
        while True:
            self.do_preprocessing()
            self.do_run()
            self.do_postprocessing(d)

            if self.reset: 
                self.do_reset()
           
            if self.sim_stop<=self.sim_time_progress:
                break    
            
        self.do_delete_nest_data() 
        return d
        
    def _sim_XX_curve(self, flag, stim=[], stim_time=0, model='',  **kwargs):    
    
        if flag=='IF':
            rp=self.get_perturbations(stim, **kwargs)
            self.replace_perturbation=rp   
            self.record=['spike_signal']
        if flag=='FF':
            ru=self.get_perturbations(stim, **kwargs)  
            self.replace_perturbation=ru
            self.record=['spike_signal']
        if flag=='IV':
            ru=self.get_perturbations(stim, **kwargs)  
            self.replace_perturbation=ru
            self.record=['voltage_signal']        
         
        self.set_sim_stop(stim_time*len(stim))
        self.set_sim_time(stim_time)
        self.set_print_time(False)
        self.reset=False
          
        d=self.simulation_loop()        
#         dud['spike_signal']['GI'].plot_firing_rate(win=1000)
#         pylab.show()
        return d

    def sim_IV_curve(self, **kwargs):
        if 'op' not in kwargs.keys():
            kwargs['op']='='
        return self._sim_XX_curve(flag='IV', **kwargs)
    
    def sim_IF_curve(self, **kwargs):
        if 'op' not in kwargs.keys():
            kwargs['op']='='
        return self._sim_XX_curve(flag='IF', **kwargs)

    def sim_FF_curve(self, **kwargs):
        if 'op' not in kwargs.keys():
            kwargs['op']='='
        return self._sim_XX_curve(flag='FF', **kwargs)

    def sim_optimization(self, x0):

        if self.run_counter==0:
            self.init_optimization()
            
        idx=[]
        vals=[]
        for i in range(len(x0)):
            if x0[i]>=0:
                continue
            
            idx.append(i)
            vals.append(x0[i])
            x0[i]=0.
        print x0
        l=self.get_perturbations_optimization(x0)
        
        for p in l:
            self.update_perturbation(p)
                  
        if self.sim_time!=self.sim_stop:
            raise RuntimeError(('simulation time and simulation stop needs',
                                ' to be equal'))
        
        d1=self.simulation_loop()
        d2=self.pops.get('target_rate')
        
        e=[]
        for model in self.fopt:
            ss=d1[model]['spike_signal']
            ss.set_target_rate(d2[model])
            e.append(ss.get_mean_rate_error(**{'t_start':self.get_start_rec()}))
        
#         print ss.get_mean_rate_error(**{'t_start':self.get_start_rec()})
#        print ss.get_mean_rate_error(**{'t_stop':self.get_start_rec()})
       
        for i, val in zip(idx, vals):
            e[i]+=val
        
#         self.clear_dud(Data_unit_spk) 
        return e
    
    def set_par_perturbations(self, val):
        self.par.set_perturbation_list(val)

    def update_perturbation(self, perturbation):
        self.par.update_perturbation(perturbation)

    def update_perturbations(self, perturbation_list):
        self.par.update_perturbations(perturbation_list)
        
    def update_dic_rep(self, dic_rep):
        self.par.update_dic_rep(dic_rep)

        
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
    
    def get_x0(self):
        x0=[]
        for net in self:
            x0+=net.get_x0() 
        return x0
       
    def sim_optimization(self, x):
        e=[]
        for net, x_slice in zip(self, self.x_slices):
            e+=net.sim_optimization(x[x_slice])
#         self.run_counter+=1
        return e
        
class Network(Network_base):
    
                        
    def _do_calibrate(self):
        '''
        Possibility to change par.
        '''
        
        with Stop_stdout(not self.verbose), Stopwatch('Calibrating...\n',
                                                      self.stopwatch):
            pass
        
    def _do_build(self):
        '''
        Build network units as nodes and population. Nodes represent spacial 
        properties and populations holds the nest representation.
        '''
          
        with Stop_stdout(not self.verbose), Stopwatch('Building...\n',
                                                      self.stopwatch):
            my_nest.ResetKernel(threads=self.threads, print_time=False)  
#             print self.par['simu']['sd_params']      
#             t=self.params_surfs
            self.surfs, self.pops=structure.build(self.params_nest, 
                                                      self.params_surfs,
                                                      self.params_popu)              
        
        
    def _do_connect(self):
        '''Connect all nodes in the model'''
     
        with Stop_stdout(not self.verbose), Stopwatch('Connecting...\n',
                                                      self.stopwatch):        
#             print self.params_conn
            args=[self.pops, self.surfs, self.params_nest, 
                  self.params_conn, self.verbose]
            
            self.conns=structure.connect(*args)

    def do_run(self):
        
        if not self.connected: self.do_connect()
       
        if not os.path.isdir(self.path_nest):
            data_to_disk.mkdir(self.path_nest)
        s='Simulating {} ms from {} ms (rec from {} ms): ...'
        s=s.format(str(self.sim_time),
                   str(self.sim_time_progress), 
                   str(self.get_start_rec()))
        with Stop_stdout(not self.verbose), Stopwatch(s,  self.stopwatch):        
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
                self.update_dic_rep(self.update_par_rep[0])
                del self.update_par_rep[0]
                
            if self.replace_perturbation:
                self.set_par_perturbations(self.replace_perturbation[0])
#                 print self.replace_perturbation[0]
                del self.replace_perturbation[0]
            
            if self.pops:           
                params_dic=self.par.get_popu_nest_params()
                setup_dic= self.par.get_spike_setup()
                for pop in self.pops:
                    params=params_dic[pop.get_name()]
                    my_nest.SetStatus(pop.ids, [params]*len(pop.ids))
                    
                    if pop.get_name() in setup_dic.keys():
                        spike_setup=setup_dic[pop.get_name()]       
                        
                        for k in spike_setup: 
                            pop.update_spike_times(**k)             
                    
                          
    def do_postprocessing(self, d):
        
        with Stop_stdout(not self.verbose), Stopwatch('Postprocessing...'):
            
            for p in self.record:
                # Set data for data units for each data call
                #print self.sim_time_progress
                #print my_nest.GetStatus([12])[0]['events']['times']
                dd=self.pops.get(p)
                for name, v in dd.items():
                    d=fill_duds_node(d, name, p, v)
#                 duds[p].add(p, self.pops.get(p))  
    
            for source, target, prop in self.record_weights_from:
                for p in prop:
                    x,y=my_nest.GetConnProp(self.pops[source].ids,
                                          self.pops[target].ids,
                                          p, 
                                          self.sim_time_progress)
                    
#                     x=[self.sim_time_progress for e in y]
                    
                    d=fill_duds_conn(d, source+'_'+target, p, x, y)
            
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


# class Inhibition(Network):
# 
#     def get_data_root_path(self):
#         return toolbox.get_data_root_path('inhibition')
# 
#     def get_figure_root_path(self):
#         return toolbox.get_figure_root_path('inhibition')
    
              
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
    def params_surfs(self):
        k=self.model_name
        d=deepcopy(self.par.get_surf()[k])
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
    

#     def voltage_respose_curve(self, currents, times, start, stop):
#         if not self.built: self.do_build()
#         if not self.calibrated: self.do_calibrate()
#         
#         pop=self.surfs[self.model_name].population
#         mm_params={'interval':0.1, 'start':self.sim_start,  'stop':self.sim_stop, 
#                    'record_from':['V_m']}       
#         pop.set_mm(True, mm_params)
#         
#         data=[]
#         for pop_id in sorted(pop.ids):
#             data.append(pop.voltage_response(currents, times, start, stop, pop_id))
#         
#         if self.n==1:idx=0
#         else:idx=slice(0,self.n)
#         
#         data=numpy.array(data)
#         times=data[:,0][idx]
#         voltages=data[:,1][idx]  
#         return times, voltages



class Unittest_net(Network):    
    
    def __init__(self,  name, *args, **kwargs):
        super( Unittest_net, self ).__init__(name, *args, **kwargs)       
        # In order to be able to convert super class object to subclass object   
        self.class_par=Unittest
        

class Bcpnn_h0(Inhibition):    
    
    def __init__(self,  dic_rep={}, perturbation=None, **kwargs):
        super( Bcpnn_h0, self ).__init__(dic_rep, perturbation, **kwargs)       
        # In order to be able to convert super class object to subclass object   
        self.class_par=Bcpnn_h0
        
 
class Bcpnn_h1(Bcpnn_h0):    
    
    def __init__(self,  dic_rep={}, perturbation=None, **kwargs):
        super( Bcpnn_h1, self ).__init__(dic_rep, perturbation, **kwargs)       
        # In order to be able to convert super class object to subclass object   
        self.class_par=Bcpnn_h1
 
def fill_duds_conn(d, name, p, x, y):

    if not name in d.keys():
        d[name]={}

          
    for key, x, y in zip(x.keys(), x.values(), y.values()):
        if not key in d[name].keys():
            d[name][key]={}
        if not p in d[name][key].keys():
            d[name][key][p]=Data_generic(**{'x':[], 'y':[]})    
        
        obj=d[name][key][p]
        
        X=obj.x
        Y=obj.y
        
        if X==[]:
            X=numpy.array([x])
            Y=numpy.array([y])
        else:
            X=numpy.concatenate((X,[x]), axis=0)
            Y=numpy.concatenate((Y,[y]), axis=0)
        
        obj.x=X
        obj.y=Y
        
    return d
 

def fill_duds_node(d, name, p, v):
    
    if p=='spike_signal':
        v=Data_unit_spk(name, v)
    if p=='voltage_signal':
        v=Data_unit_vm(name, v)  
          
    if not name in d:
        d=misc.dict_recursive_add(d, [name, p],v)
    elif not p in d[name]:
        d=misc.dict_recursive_add(d, [name, p],v)
    else:
        d[name][p].add(v)
        
    
    return d

def network_kwargs():
    kwargs={'save_conn':False, 
            'sub_folder':'unit_testing', 
            'verbose':True,
            'record':['spike_signal'],
            'record_weights_from':[]}
    return kwargs     
 
def plot_plastic(duds):
    vm = duds['voltage_signal']
    spk = duds['spike_signal']
    pylab.figure()
    ax = pylab.subplot(311)
    obj = spk['n1'].wrap.as_spike_list()
    vm['n1'].plot(**{'ax':ax, 'id_list':[0], 'spike_signal':obj})
    obj = spk['n2'].wrap.as_spike_list()
    vm['n2'].plot(**{'ax':ax, 'id_list':[0], 'spike_signal':obj})
    ax.legend(['Neuron 1', 'Neuron 2'])
    ax = pylab.subplot(312)
    spk['n1'].plot_firing_rate(ax)
    ax = pylab.subplot(312)
    spk['n2'].plot_firing_rate(ax)
    ax.legend(['Neuron 1', 'Neuron 2'])
    ax = pylab.subplot(313)
    ax.plot(numpy.array(duds['n1_n2']['weight']), **{'label':'Weight'})
    ax.legend()


def plot_plastic_dopa(d):
    spk={}
    for name in  ['n1','n2','m1']:
        spk[name]=d[name]['spike_signal']
    pylab.figure()

    ax=pylab.subplot(211)  
    spk['n1'].cmp('firing_rate').plot(ax, **{'win':100,'linewidth':2.0})
    spk['n2'].cmp('firing_rate').plot(ax, **{'win':100,'linewidth':2.0})
    spk['m1'].cmp('firing_rate').plot(ax, **{'win':100,'linewidth':2.0})
    ax.legend(['Neuron 1', 'Neuron 2','Dopamine neurons'])

    ax=pylab.subplot(212)    
    
    
    dc=d['n1_n2']['n1_n2_nest_bcpnn_dopa']     
    p1=(dc['p_i'].y
        *dc['p_j'].y)
    p2=dc['p_ij'].y/p1
    p=numpy.log(p2)
    
    c=misc.make_N_colors('jet', 12)
    ax.plot(dc['p_i'].x[:,0] , p[:,0], **{'label':'weight', 
                                                        'linewidth':2.0,
                                                        'color':c[0]} )
    ax.set_ylabel('Weight')
    dc['n'].plot(ax, **{'label':'n', 'color':'c', 'linewidth':2.0})
    dc['k'].plot(ax, **{'label':'Dopamine (Kappa)', 
                        'color':'k', 'linewidth':2.0})
    dc['k_filtered'].plot(ax, **{'label':'Filtered (Kappa)',
                                  'color':'m', 'linewidth':2.0})
    ax.legend()   
    
    pylab.figure()
    ax=pylab.subplot(311)    
    i=1
    for v in ['z_i', 'z_j', 'z_j_c', 'e_i', 'e_j', 
              'e_ij', 'e_j_c', 'e_ij_c',]:
        dc[v].plot(ax, **{'label':v, 'color':c[i], 'linewidth':2.0})  
        i+=1
  
    ax.legend()   
     
    ax=pylab.subplot(312)    
  
    for v in [ 'p_i', 'p_j',  'p_ij']:
        dc[v].plot(ax, **{'label':v, 'color':c[i], 'linewidth':2.0}) 
        i+=1

#     ax.legend()
         
#     pylab.figure()
    i=1
    ax=pylab.subplot(313)     
  
    for v in [ 'p_i', 'p_j',  'p_ij',]:     
        ax.plot(dc[v].x,dc[v].y/max(dc[v].y), 
                **{'label':v, 'color':c[i], 'linewidth':2.0})
        i+=1
 
    ax.plot(dc[v].x, p1/max(p1),  **{'label':'p_i*p_j', 'color':c[i], 
                                     'linewidth':2.0})   
    ax.plot(dc[v].x, p2, **{'label':'p_i*p_j/p_ij', 'color':c[i+1], 
                            'linewidth':2.0})
    ax.legend()
             
class TestMixin_1(object):
    pass
#     def test_1_build(self):
#         network=self.class_network(self.name, **self.kwargs)
#         network.do_build()
#              
#     def test_2_connect(self):
#         network=self.class_network(self.name, **self.kwargs)
#         network.do_connect()    
#         
#     def test_3_run(self):
#         network=self.class_network(self.name, **self.kwargs)
#         network.do_run()  
#                 
#     def test_4_reset(self):
#         network=self.class_network(self.name, **self.kwargs)  
#         network.do_run()
#         network.do_reset()
#         network.do_run() 


class TestMixin_2(object):
    def update_par_rep(self, node):
        update_par_rep=[]
        for c in self.curr:
            dic_rep={'node':{node:{'I_vivo':float(c)}}}
            update_par_rep.append(dic_rep) 
        return update_par_rep
    
    def replace_perturbation(self):
        replace_perturbation=[]
        for c in  self.curr:
            d={'node':{self.node:{'nest_params':{'I_e': float(c)}}}}
            p=Perturbation_list(d , '+', **{'name':'Curr'})
            replace_perturbation.append(p) 
        return replace_perturbation
         
    def replace_perturbation2(self):
        replace_perturbation=[]
        for c in  self.rates:
            d={'node':{self.input:{'rate':float(c)}}} 
            p=Perturbation_list(d, '=', **{'name':'Rate'})
            replace_perturbation.append(p) 
        return replace_perturbation  
    
    def test_5_simulation_loop(self):
                
        d={'node':{self.node:{'nest_params':{'I_e':400.0}}}}
        network=self.class_network(self.name, **self.kwargs)  
        network.update_dic_rep(d)
        network.reset=False
        network.set_sim_stop(1000.0)     
        d=network.simulation_loop()
        spk=d[self.node]['spike_signal']  
         
        pylab.figure()
        ax=pylab.subplot(111)   
 
        spk.cmp('firing_rate').plot(ax,**{'win':100, 'label':'Mean'})  
        spk[:,0].cmp('firing_rate').plot(ax,**{'win':100, 'label':'Set 1'}) 
        spk[:,1].cmp('firing_rate').plot(ax,**{'win':100, 'label':'Set 2'}) 
        spk[:,2].cmp('firing_rate').plot(ax,**{'win':100, 'label':'Set 3'}) 
#         pylab.show()
#      
    def test_6_simulation_loop_x3_update_par_rep(self):
        kwargs=deepcopy(self.kwargs)
        kwargs.update({'update_par_rep':self.update_par_rep(self.node)})
        network=self.class_network(self.name, **kwargs)  
        network.reset=False
        d=network.simulation_loop()
        spk=d[self.node]['spike_signal'] 
        pylab.figure()
        ax=pylab.subplot(221)
        spk.cmp('firing_rate').plot(ax, **{'win':100, 'label':'Mean'})
        spk[:,0].cmp('firing_rate').plot(ax, **{'win':100, 'label':'Set 1'})
                  
        kwargs.update({'update_par_rep':self.update_par_rep(self.node)})
        network=self.class_network(self.name, **kwargs)  
        network.reset=True
        d=network.simulation_loop()
        spk=d[self.node]['spike_signal']      
     
        ax=pylab.subplot(222)
        spk[0,:].cmp('firing_rate').plot(ax,**{'win':100, 'label':'Run 1'})
        spk[1,:].cmp('firing_rate').plot(ax,**{'win':100, 'label':'Run 2'})
        spk[2,:].cmp('firing_rate').plot(ax,**{'win':100, 'label':'Run 3'}) 
        spk.cmp('isi').hist(pylab.subplot(223))
#         pylab.show() 
          
#   
    def test_7_simulation_loop_x3_replace_pertubations(self):
        kwargs=deepcopy(self.kwargs)
        kwargs.update({'replace_perturbation':self.replace_perturbation()})
        network=self.class_network(self.name, **kwargs)  
        network.reset=False
        d=network.simulation_loop()
        spk=d[self.node]['spike_signal']
        pylab.figure()
        ax=pylab.subplot(221)
        spk.cmp('firing_rate').plot(ax, **{'win':100})
                   
                   
        kwargs.update({'replace_perturbation':self.replace_perturbation()})
        network=self.class_network(self.name, **kwargs)  
        network.reset=True
        d=network.simulation_loop()
        spk=d[self.node]['spike_signal']
        spk.reset('firing_rate') 
        ax=pylab.subplot(222)
        spk[0,:].cmp('firing_rate').plot(ax,**{'win':100, 'label':'Run 1'})
        spk[1,:].cmp('firing_rate').plot(ax,**{'win':100, 'label':'Run 2'})
        spk[2,:].cmp('firing_rate').plot(ax,**{'win':100, 'label':'Run 3'}) 
        spk.cmp('isi').hist(pylab.subplot(223))
#         pylab.show()  
# 
    def test_90_IV_curve(self):
        network=self.class_network(self.name, **self.kwargs)  
        kwargs={'stim':self.curr_IV,
                'stim_name':'node.'+self.node+'.nest_params.I_e',
                'stim_time':500.0,
                'model':self.node}
        d=network.sim_IV_curve(**kwargs)
        vm=d[self.node]['voltage_signal']
        pylab.figure()
        ax=pylab.subplot(111)
        vm.cmp('IV_curve').plot(ax, x=self.curr_IV,
                                     **{'label':'Mean', 'marker':'o'})
#         pylab.show() 
                        
    def test_90_IF_curve(self):
        network=self.class_network(self.name, **self.kwargs)  
        kwargs={'stim':self.curr,
                'stim_name':'node.'+self.node+'.nest_params.I_e',
                'stim_time':500.0,
                'model':self.node}
        d=network.sim_IF_curve(**kwargs)
        pylab.figure()
        spk=d[self.node]['spike_signal']
        ax=pylab.subplot(111)
        spk.cmp('IF_curve').plot(ax, **{'linewidth':3.0, 'color':'b'})
        spk.cmp('IF_curve').plot(ax, part='first',**{'linewidth':3.0, 
                                                      'color':'g'})#       
#         pylab.show() 
                              
    def test_91_FF_curve(self):
        dic_rep={'simu':{'sd_params':{'to_file':False, 'to_memory':True}},
                         'netw':{'size':36.}}
        self.kwargs.update({'dic_rep':dic_rep})
        network=self.class_network(self.name, **self.kwargs)  
        kwargs={'stim':self.rates,
                'stim_name':'node.'+self.input+'.rate',
                'stim_time':3000.0,
                'model':self.node}
        d=network.sim_FF_curve(**kwargs)
        spk=d[self.node]['spike_signal']
        pylab.figure()
        ax=pylab.subplot(111)      
        spk.cmp('mean_rate_parts').plot_FF(ax,  x=self.rates, 
                                           **{'label':'Mean'})
        spk[:,0].cmp('mean_rate_parts').plot_FF(ax, x=self.rates,
                                                **{'label':'Set 1'})
#         pylab.show()
# 
    def test_92_voltage_trace(self):
        dic_rep=deepcopy(self.dic_rep)
           
        dic_rep['node'][self.node]['mm']['active']=True
        d={'node':{self.node:{'nest_params':{'I_e':400.0}}}}
           
        dic_rep=misc.dict_update(dic_rep,d)
           
        kwargs=deepcopy(self.kwargs)
        par=Unittest(**{'dic_rep':dic_rep})
        kwargs.update({ 'par':par,
                       'record':['voltage_signal', 'spike_signal']})
        #kwargs.update({'replace_perturbation':self.replace_perturbation2()})
        network=self.class_network(self.name, 
                                   **kwargs)          
        network.reset=False
        d=network.simulation_loop()
        vm=d[self.node]['voltage_signal']
        spk=d[self.node]['spike_signal']
        pylab.figure()
                 
        obj=spk.wrap.as_spike_list()
        ax=pylab.subplot(111)
        obj=vm.cmp('voltage_traces', **{ 'spike_signal':obj})      
        obj.plot(ax, **{'id_list':[0]})
                 
#         pylab.show()
 
    def test_93_optimization(self):
        dic_rep=deepcopy(self.dic_rep)
          
        x0=[3000.]
        opt={'f':[self.node],
             'x':['node.i1.rate'],
             'x0':x0}
        dic_rep.update({'simu':{'sim_stop':3000.0,
                                'sim_time':3000.0},
                        'netw':{'optimization':opt}})
        self.kwargs.update({'par':self.class_par(**{'dic_rep':dic_rep})})
        network=self.class_network(self.name, **self.kwargs)  
   
        e1=network.sim_optimization(x0)
        e2=network.sim_optimization([3000.0+200.0])
#         print e1,e2
        self.assertAlmostEqual(e1[0], 18, delta=3)
        self.assertAlmostEqual(e2[0], 28, delta=2)
#         #pylab.show()
        

class TestMixinPlastic(object):           
    def test_simulation_loop(self):
        dic_rep=self.dic_rep
        d={'verbose':True,
            'par':self.class_par(**{'dic_rep':dic_rep,
                                    'other':self.other,
                                     }),
            'record':['voltage_signal', 
                       'spike_signal'],
            'record_weights_from':[['n1', 'n2', ['weight',
                                                 ]]]}            
        self.kwargs.update(d)

        network=self.class_network(self.name, **self.kwargs)  
        network.reset=False     
        duds=network.simulation_loop() 
        plot_plastic(duds)
        pylab.show()  
        
class TestMixinPlasticDopa(object):
    def test_simulation_loop(self):
        dic_rep=self.dic_rep
        d={'verbose':True,
            'par':self.class_par(**{'dic_rep':dic_rep,
                                    'other':self.other,
                                     }),
            'record':['voltage_signal', 
                      'spike_signal'],
            'record_weights_from':[['n1', 'n2', ['weight',
                                                'k',
                                                'k_filtered',
                                                'n',
                                                'm',
                                                'z_i',
                                                'z_j',
                                                'z_j_c',
                                                'e_i',
                                                'e_j',
                                                'e_ij',
                                                'e_j_c',
                                                'e_ij_c',
                                                'p_i',
                                                'p_j',
                                                'p_ij',
                                                 ]]]}               
        self.kwargs.update(d)

        network=self.class_network(self.name, **self.kwargs)  
        network.reset=False     
        duds=network.simulation_loop()
        plot_plastic_dopa(duds)
        pylab.show() 
 
class TestMixinSetup(object):
    def _setUp(self, **kwargs):
        dic_rep=kwargs.get('dic_rep',{})
        d={'simu':{'sim_stop':3000.0,
                    'threads':4,
                    'mm_params':{'to_file':False, 'to_memory':True},
                    'sd_params':{'to_file':False, 'to_memory':True}},
            'netw':{'size':9.},
            'node':{self.node:{'mm':{'active':True},
                               'sd':{'active':True}}}}
        dic_rep=misc.dict_update(d, dic_rep)
        self.kwargs.update({'verbose':False,
                            'par':self.class_par(**{'dic_rep':dic_rep})})
        self.dic_rep=dic_rep

class TestUnittest_base(unittest.TestCase):
    kwargs=network_kwargs()
    curr=numpy.array([100,500,700],dtype=float)
    curr_IV=numpy.array(range(0,400,110),dtype=float)
    rates=numpy.array(range(2500, 3000, 100), dtype=float)
       
    def setUp(self):    
        self.class_network=Network
        self.class_par=Unittest
        self.name='net1'
        self.node='n1'
        self.input='i1'
        self._setUp()

        
class TestUnittest(
                   TestUnittest_base, 
                   TestMixin_1, 
                   TestMixin_2,
                   TestMixinSetup,
                   ):
    pass

class TestUnittestExtend_base(unittest.TestCase):
    kwargs=network_kwargs()
    curr=numpy.array([100,500,700],dtype=float)
    curr_IV=numpy.array(range(0,400,110),dtype=float)
    rates=numpy.array(range(200, 500, 100),dtype=float)
       
    def setUp(self):    
        self.class_network=Network
        self.class_par=Unittest_extend
        self.name='net1'
        self.node='n1'
        self.input='i1'
        self._setUp()

        
class TestUnittestExtend(
                   TestUnittest_base, 
                   TestMixin_1, 
                   TestMixin_2,
                   TestMixinSetup,
                   ):
    pass        
class TestUnittestBcpnnDopa_base(unittest.TestCase):
    kwargs=network_kwargs()
    
    def setUp(self):
        self.other=Unittest()
        self.class_network=Network
        self.class_par=Unittest_bcpnn_dopa
        dic_rep={}
#         ['n_nuclei']={'n1':1, 'n2':1, 'm1':50}
        dic_rep.update({'simu':{'sim_stop':1000.0,
                                'sim_time':10.0,
                                'threads':1,
                         'mm_params':{'to_file':False, 'to_memory':True},
                         'sd_params':{'to_file':False, 'to_memory':True}},
                         'netw':{'size':52,
                                 'n_nuclei':{'m1':52}},
                         
                         'node':{'n1':{'mm':{'active':False},
                                       'sd':{'active':True}},
                                 'n2':{'mm':{'active':False},
                                       'sd':{'active':True}},
                                 'm1':{'mm':{'active':False},
                                       'sd':{'active':True}}}})
        self.kwargs.update({'verbose':True,
                            'par':self.class_par(**{'dic_rep':dic_rep,
                                                     'other':self.other}),
                           })
        self.dic_rep=dic_rep
        self.name='net1'        
        
   
class TestUnittestBcpnnDopa(TestUnittestBcpnnDopa_base, TestMixin_1,
                        TestMixinPlasticDopa):
    pass

class TestUnittestStdp_base(unittest.TestCase):
    kwargs=network_kwargs()
    
    def setUp(self):
        
        self.class_network=Network
        self.class_par=Unittest_stdp
        #self.fileName=self.class_network().path_data+'network'
        self.model_list=['n1']
        self.other=Unittest_bcpnn_dopa(**{'other':Unittest()})
        dic_rep={}
        
        dic_rep.update({'simu':{'sim_stop':5000.0,
                                'sim_time':100.0,
                                'threads':1,
                         'mm_params':{'to_file':False, 'to_memory':True},
                         'sd_params':{'to_file':False, 'to_memory':True}},
                         'netw':{'size':11},
                         'node':{'n1':{'mm':{'active':True},
                                       'sd':{'active':True}}}})
        
        self.kwargs.update({'verbose':True,
                            'par':self.class_par(**{'dic_rep':dic_rep,
                                                     'other':self.other}),
                           })
        self.dic_rep=dic_rep
        self.name='net1'        
      
 
class TestUnittestStdp(
                       TestUnittestStdp_base, 
                       TestMixin_1,
                       TestMixinPlasticDopa,
                       ):
    pass

class TestUnittestBcpnn_base(unittest.TestCase):
    kwargs=network_kwargs()
    
    def setUp(self):
        
        self.class_network=Network
        self.class_par=Unittest_bcpnn
        #self.fileName=self.class_network().path_data+'network'
        self.model_list=['n1']
        self.other=Unittest_bcpnn_dopa(**{'other':Unittest()})
        dic_rep={}
        
        dic_rep.update({'simu':{'sim_stop':10000.0,
                                'sim_time':100.0,
                                'threads':2,
                         'mm_params':{'to_file':False, 'to_memory':True},
                         'sd_params':{'to_file':False, 'to_memory':True}},
                         'netw':{'size':2},
                         'node':{'n1':{'mm':{'active':True},
                                       'sd':{'active':True}}}})
        
        self.kwargs.update({'verbose':True,
                            'par':self.class_par(**{'dic_rep':dic_rep,
                                                     'other':self.other}),
                           })
        self.dic_rep=dic_rep
        self.name='net1'        
      
 
class TestUnittestBcpnn(TestUnittestBcpnn_base, TestMixin_1,
                        TestMixinPlastic):
    pass



class TestSinlge_unit_base(unittest.TestCase):
    kwargs=network_kwargs()
    curr=range(200,400,50)#[-300,500,700]
    curr_IV=range(-300,200, 100)
    rates=range(0, 1600, 500)
       
    def setUp(self):    
        self.class_network=Network
        self.class_par=Single_unit
        self.name='net1'
        self.node='FS'
        self.input='CFp'

        d={'simu':{'sim_stop':3000.0,
                   'sim_time':1000.0,
                   'start_rec':0.0,
                   'stop_rec':3000.0,
                   'print_time':False,
                    'threads':1,
                    'mm_params':{'to_file':False, 'to_memory':True},
                    'sd_params':{'to_file':False, 'to_memory':True},
                    },
            'netw':{'size':9.,
                    'single_unit':self.node},
            'node':{self.node:{'mm':{'active':True},
                               'sd':{'active':True},
                               'n_sets':3},
                    'FSp':{'lesion':True},
                    'GAp':{'lesion':True},
                    'CFp':{'lesion':False, 'rate':0.0},
                    }}
        par=self.class_par(**{'dic_rep':d,
                               'other':Inhibition()})
        self.kwargs.update({'verbose':True,
                            'par':par})
        self.dic_rep=d

        
class TestSingle_unit(
                   TestSinlge_unit_base, 
                   TestMixin_1, 
                   TestMixin_2,
                   ):
    pass

class TestNetwork_dic(unittest.TestCase):
    def setUp(self):
        self.x0=[3000.0, 3000.0]
        opt={'f':['n1', 'n2'],
             'x':['node.i1.rate', 'node.i2.rate'],
             'x0':self.x0}
        
        dic_rep={}
        dic_rep.update({'simu':{'sim_stop':1000.0,
                                'sim_time':1000.0,
                             'mm_params':{'to_file':False, 'to_memory':True},
                             'sd_params':{'to_file':False, 'to_memory':True}},
                        'netw':{'rand_nodes':{'C_m':False, 
                                              'V_th':False, 
                                              'V_m':False},
                                'size':50.,
                                'optimization':opt},
#                         'conn':{'n1_n2':{'lesion':True}},
                        })

        k={'class':'Unittest',
            'save_conn':False, 
            'sub_folder':'unit_testing', 
            'verbose':True,
            'par':Unittest_extend(**{'dic_rep':dic_rep, 'other':Unittest()})}
        self.network_kwargs=k
#         self.nd=Network_list()
        #nd.add(name, **kwargs)
#         self.kwargs={'models':self.nd}
    
    def test_x_slice(self):
        nd=Network_list([Network('net1', **self.network_kwargs)])
#         self.nd.add('net1', **self.network_kwargs) 
        #self.nd.sim_optimization([3000, 3000])
        s=nd.x_slices
        self.assertEqual(s,[slice(0,2)])
        
        nd.append( Network('net2', **self.network_kwargs)) 
        s=nd.x_slices
        self.assertEqual(s,[slice(0,2),slice(2,4)]) 
     
    def test_sim_optimization(self):
        nd=Network_list([Network('net1', **self.network_kwargs)])
#         self.nd.add('net1', **self.network_kwargs) 
        e1=nd.sim_optimization([3000., 3000.])
        nd.append( Network('net2', **self.network_kwargs)) 
        e2=nd.sim_optimization([3000., 3000., 3100., 3100.])
        
        self.assertAlmostEquals(sum(e1),sum(e2[0:2]), delta=0.1)
        self.assertFalse(sum(e1)==sum(e2))



                           
# class TestStructureInhibition(unittest.TestCase):
#     
#     nest.sr("M_WARNING setverbosity") #silence nest output
#     def do_reset(self):
#         self.kwargs={'save_conn':False, 'sub_folder':'unit_testing', 'verbose':True}
#         self.dic_rep={'simu':{'start_rec':1.0, 'stop':100.0,'sim_time':100., 
#                               'sd_params':{'to_file':True, 'to_memory':False},
#                               'threads':4, 'print_time':False},
#              'netw':{'size':500.0},
#              }
#         
#     def change_fan_in(self):
#         self.dic_rep.update({'conn':{'M1_M1_gaba':{'fan_in0':5},
#                                      'M1_M2_gaba':{'fan_in0':5},
#                                      'M2_M1_gaba':{'fan_in0':5},
#                                      'M2_M2_gaba':{'fan_in0':5},
#                                      'M1_SN_gaba':{'fan_in0':5},
#                                      'M2_GI_gaba':{'fan_in0':5}}})
#         
#     def build_connect(self):
#         network=self.class_network(self.name, **self.kwargs)
#         
#         for key, val in network.par['conn'].items():
#             #print key
#             if val['delay_setup']['type']=='uniform':
#                 val['delay_setup']['type']='constant'
#                 val['delay_setup']['params']=(val['delay_setup']['params']['max']
#                                               +val['delay_setup']['params']['min'])/2.
#         
#         network.do_build()
#         network.do_connect()
#         return network 
#     
#     def setUp(self):
#         self.do_reset()
#         self.change_fan_in()
#         self.dic_rep['netw'].update({'size':50.0, 
#                                      'sub_sampling':{'M1':20.0,'M2':20.0}})
#         self.class_network=Inhibition
#         
#         self.fileName=self.class_network().path_data+'network'
#         self.model_list=['M1','M2', 'FS', 'GA', 'GI', 'ST', 'SN']
#             
#     def test_a_build_connect_run(self):
#         
#         network=self.build_connect()
#         network.do_run()
#         data_to_disk.pickle_save(network, self.fileName)
# 
#         
#     def test_reset(self):
#         network=self.build_connect()    
#         network.do_run()
#         network.do_reset()
#         network.do_run()
#         self.assertEqual(network.start_rec, network.sim_start)
#         
#     def test_get_firing_rate(self):
#         
#         network=data_to_disk.pickle_load(self.fileName)
#         my_nest.SetKernelStatus({'data_path':network.path_nest})
#         fr=network.get_firing_rate(self.model_list)
#         
#         for k in self.model_list:
#             self.assertEqual(len(fr[k]['times']), network.par['simu']['sim_time']-1)
#             self.assertEqual(len(fr[k]['rates']), network.par['simu']['sim_time']-1)
# 
#     def test_get_firing_rate_sets(self):
#         
#         network=data_to_disk.pickle_load(self.fileName)
#         my_nest.SetKernelStatus({'data_path':network.path_nest})
#         d=network.get_firing_rate_sets(self.model_list)
#         
#         for k in self.model_list:
#             self.assertEqual(len(d[k]['times']), network.par['simu']['sim_time']-1)
#             self.assertEqual(d[k]['rates'].shape[1], network.par['simu']['sim_time']-1)
#             self.assertEqual(d[k]['rates'].shape[0], network.par['node'][k]['n_sets'])
# 
#     def test_get_raster(self):
#         
#         network=data_to_disk.pickle_load(self.fileName)
#         my_nest.SetKernelStatus({'data_path':network.path_nest})
#         d=network.get_rasters(self.model_list)
#         
#         for k in self.model_list:
#             self.assertIsInstance(d[k][0], numpy.ndarray)
#             self.assertEqual(d[k][0].shape[0], 2)
#             self.assertListEqual(list(d[k][1]), range(network.par['node'][k]['n']))
# 
#     def test_get_rasters_sets(self):
#         
#         network=data_to_disk.pickle_load(self.fileName)
#         my_nest.SetKernelStatus({'data_path':network.path_nest})
#         d=network.get_rasters_sets(self.model_list)
#         
#         for k in self.model_list:
#             n_sets=network.par['node'][k]['n_sets']
#             n=network.par['node'][k]['n']
#             self.assertEqual(len(d[k]), n_sets)
#             i=0
#             acum=0
#             for l in d[k]:
#                 self.assertIsInstance(l[0], numpy.ndarray)
#                 self.assertEqual(l[0].shape[0], 2)
#                 self.assertEqual(len(l[0][0]),len(l[0][1]))
#                 
#                 
#                 m=len(list(range(i, n, n_sets)))
#                 self.assertListEqual(list(l[1]), range(acum, acum+m))
#                 i+=1
#                 acum+=m
# 
# 
# class TestStructureSingle_units_activity_base(unittest.TestCase):
#     def setUp(self):
#         self.do_reset()
#         self.class_network=Single_units_activity    
#         self.dic_rep['netw'].update({'size':10})  
#         #for key in ['M1_M1_gaba', 'M1_M2_gaba', 'M2_M1_gaba', 'M2_M2_gaba']:
#         #    del self.dic_rep['conn'][key]
#         self.fileName=self.class_network().path_data+'network'             
#         self.model_list=['M1']
# 
# class TestStructureSingle_units_activity(TestStructureSingle_units_activity_base, TestMixin_1):
#     pass
# 
# 
# class TestStructureSingle_units_in_vitro_base(unittest.TestCase):
#     def setUp(self):
#         self.do_reset()
#         self.class_network=Single_units_activity    
#         self.dic_rep['netw'].update({'size':10})  
#         #for key in ['M1_M1_gaba', 'M1_M2_gaba', 'M2_M1_gaba', 'M2_M2_gaba']:
#         #    del self.dic_rep['conn'][key]
#         self.fileName=self.class_network().path_data+'network'             
#         self.model_list=['M1']
# 
# class TestStructureSingle_units_in_vitro(TestStructureSingle_units_activity_base, TestMixin_1):
#     pass
#         
# class TestStructureSlow_wave(TestStructureInhibition):
#     
#     def setUp(self):
#         self.do_reset()
#         self.change_fan_in() 
#         self.dic_rep['netw'].update({'size':50.0, 'sub_sampling':{'M1':20.0,'M2':20.0}}) 
#         self.class_network=Slow_wave
#         
#         self.fileName=self.class_network().path_data+'network'
#         self.model_list=['M1','M2', 'FS', 'GA', 'GI', 'ST', 'SN']
#         
# class TestStructureBcpnn_h0(TestStructureInhibition):
#     
#     def setUp(self):
#         self.do_reset()
#         self.change_fan_in()
#         self.dic_rep['netw'].update({'sub_sampling':{'M1':40.0,'M2':40.0, 'CO':600.0},
#                                       'size':100.})     
#         self.class_network=Bcpnn_h0
#         
#         self.fileName=self.class_network().path_data+'network'
#         self.model_list=['CO', 'M1','M2', 'FS', 'GA', 'GI', 'ST', 'SN']
#             
# class TestStructureBcpnn_h1(TestStructureBcpnn_h0):
# 
#     def setUp(self):
#         self.do_reset()
#         self.change_fan_in()
#         self.dic_rep['netw'].update({'sub_sampling':{'M1':40.0,'M2':40.0, 'CO':600.0},
#                                       'size':100.})      
#         self.class_network=Bcpnn_h1        
#         self.fileName=self.class_network().path_data+'network'
#         self.model_list=['CO', 'M1','M2', 'F1', 'F2', 'GA', 'GI', 'ST', 'SN']

       
if __name__ == '__main__':
    
    test_classes_to_run=[
#                         TestUnittest,
#                         TestUnittestExtend,
                        TestUnittestBcpnnDopa,
#                         TestUnittestStdp,
#                         TestUnittestBcpnn,
#                         TestSingle_unit
#                         TestNetwork_dic,
                       ]
    suites_list = []
    for test_class in test_classes_to_run:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)
    unittest.TextTestRunner(verbosity=2).run(big_suite)  
                
