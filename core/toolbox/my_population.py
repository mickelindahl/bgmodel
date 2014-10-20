''''
Mikael Lindahl August 2011


 Creates a group of neurons and attaches recording devices spike detector and
 multimeter. 

 Usage:

 Declare a mygroup: 
   mygroup = MyGroup(models = 'iaf_neuron', n=1, params = {}, mm_dt=1.0, sname='', spath='', sd=True, mm=True)


 When simulation is finished conductance, current and voltage lists can be
 obtained by calling the function get_signal() with parameter dataype, 
 ('g' for conductance, 'c' for current, 'v' for voltage and 's' or 'spike' for
 spike date) and my_nest recordable type (except for spike data). With 
 save_signal() and load_signal() data can instead be written and read from disk.
 
 Function can take several modes and number of neurons per model. Then the models
 has to have the same recordables. Also if parameters is provided in params these
 are set for all the models. 
'''



import numpy
import os
import my_signals
import my_nest
import my_topology
import copy
import unittest
import pprint
pp=pprint.pprint
import pylab

from numpy.random import random_integers
from my_signals import (MyConductanceList, MyCurrentList, 
                        MyVmList, MySpikeList, SpikeListMatrix,
                        VmListMatrix)
from toolbox import data_to_disk, misc
from toolbox.parallelization import comm, Barrier


#from numpy.random import RandomState
#random_integers  = RandomState(3).random_integers 

        
class MyGroup(object):
    '''
    MyGroup(self, model = 'iaf_neuron', n=1, params = {}, mm_dt=1.0, sname='', spath='', sd=True, mm=True)
    
    Arguments:
        model      my_nest model type, can be a list
        n           number of model to create, can be a list
        params      common parameters for model to be set
        mm_dt       multimeter recording precision
        sname       file basename
        spath       Path to save file at 
        sd          boolean, True if spikes should me recorded
        mm          boolean, True if mulitmeter should record  
        
    ''' 
    def __init__(self, name, **kwargs ):
        '''
        Constructor
        
        Arguments:
            ids         provide ids if nodes already created
            model       my_nest model type, can be a list
            n           number of model to create, can be a list
            params      common parameters for model to be set
            mm_dt       multimeter recording precision
            sname       file basename
            spath       Path to save file at 
            sd          boolean, True if spikes should me recorded
            mm          boolean, True if mulitmeter should record  
        '''         
        
        model=kwargs.get('model', 'iaf_neuron')
        n=kwargs.get('n', 1)
        params=kwargs.get('params',{})          
            
        ids = kwargs.get('ids', my_nest.Create(model,n, params))
        self._ids=slice(ids[0],ids[-1],1)
        

        self.local_ids=[]
        for _id in self.ids:       
            if my_nest.GetStatus([_id], 'local') :
                self.local_ids.append(_id)
        
        self.model=model
        self.name=name
        self.n=n
        self.sets=kwargs.get('sets', misc.my_slice(0, n, 1))

    @property
    def ids(self):
        return list(range(self._ids.start, self._ids.stop+1, self._ids.step))
                    
    def __getitem__(self, key):
        ''' 
        Calling self with one index
        '''
        ids=numpy.array(self.ids) 
        ids=ids[key]

        if type(ids).__name__ in ['int', 'int64']:
            return ids
        else:
            return list(ids)

    def __len__(self):
        ''' 
        Return lenght of ids list. Neccesary to have to be able to call 
        self[1:-1] where -1 inforce length lookup
        '''
        return len(self.ids)   
             
        
    def __repr__(self):
        return self.__class__.__name__+':'+self.name    
    
    def __str__(self):
        return self.__class__.__name__+':'+self.name 
                     
    def get(self, attr, **k):
               
        if hasattr(self, 'get_'+ attr): 
            call=getattr(self, 'get_'+ attr)
        elif hasattr(self, attr):
            call=getattr(self, attr)
        else:
            return None
        
        if self.isrecorded(attr):
            return call(**k)
        else:
            return None

    def get_name(self):
        return self.name
           
    def set_random_states_nest(self):        
        msd = 1000 # masterseed

        n_vp = my_nest.GetKernelStatus ( 'total_num_virtual_procs' )

        msdrange1 = range(msd , msd+n_vp )

        pyrngs = [numpy.random.RandomState(s) for s in msdrange1 ]
        msdrange2 = range(msd+n_vp+1 , msd+1+2*n_vp )
        my_nest.SetKernelStatus(params={'grng_seed': msd+n_vp ,'rng_seeds': msdrange2})   
        return pyrngs
             
class VolumeTransmitter(MyGroup):
    def __init__(self, *args, **kwargs):
    
        super( VolumeTransmitter, self ).__init__(*args, **kwargs)        
        self._init_extra_attributes(**kwargs)

    def _init_extra_attributes(self,*args, **kwargs):
        self.syn_target=kwargs.get('syn_target','')
    
    def get_syn_target(self):
        return self.syn_target

class MyNetworkNode(MyGroup):
    def __init__(self, *args, **kwargs):
        
        super( MyNetworkNode, self ).__init__(*args, **kwargs)     

        self._init_extra_attributes(**kwargs)

    def _init_extra_attributes(self,*args, **kwargs):
        # Add new attribute
           
#         self.sd_params=kwargs.get('sd_parans')            
        self.mm=self.create_mm(self.name, kwargs.get('mm', {} ))  
        

        
        self.rand   = kwargs.get('rand',{})
        
        self.receptor_types=my_nest.GetDefaults(self.model).get('receptor_types',{})   
        self.sd=self.create_sd(self.name, kwargs.get('sd', {} ))  

 
        
        self.signals         = {}        # dictionary with signals for current, conductance, voltage or spikes                
        self._signaled        = {}        # for ech signal indicates if it have been loaded from nest or not
        self.target_rate=kwargs.get('rate', 10.0)
        

        
        if not {}==self.rand:
            self.model_par_randomize()
        
    @property
    def recordables(self):
        # Pick out recordables and receptor types using first model.
        try: 
            return my_nest.GetDefaults(self.model)['recordables']
        except: 
            return [] 
    @property
    def signaled(self):
        if not self._signaled:
            for rec in self.recordables:
                self._signaled[rec]=False
            self._signaled['spikes']=False
#             self._signaled=signaled
        return self._signaled

    @property
    def spike_signal(self):
        return self._signal('s', 'spikes')

    @property
    def voltage_signal(self):
        return self._signal('v', 'V_m')
  
    def is_new_recording(self, flag, recordable):
        if flag in ['s','spikes']:
#             stop=max(1,my_nest.GetKernelStatus('time')-1)
            stop=max(1,my_nest.GetKernelTime()-1)
        else:
#             stop=my_nest.GetKernelStatus('time')
            stop=my_nest.GetKernelTime()
        if self.signals[recordable].t_stop==stop:
            return False
        else:
            return True
  
    def _signal(self, flag, recordable):
        if self.signaled[recordable]:
            if not self.is_new_recording( flag, recordable):
                return self.signals[recordable]
            else:
                start=self.signals[recordable].t_stop
        else:
            start=0.0
        
        if not (start==my_nest.GetKernelTime()):
            if flag in ['s', 'spikes']:   
                start=max(0,start-1)
                #Todo with delay in spike recording I experienced
                stop=max(1,my_nest.GetKernelTime()-1)
                 
            else:
                stop=my_nest.GetKernelTime()
                
            signal=self.get_signal(flag, recordable=recordable, 
                                   start=start, 
                                   stop=stop) 
            if flag in ['s', 'spikes']:
                signal.complete(self.ids)
            self.signals[recordable]=signal
            self.signaled[recordable]=True
        try:
            return self.signals[recordable]
        except:
            raise KeyError("{} signals not present".format(recordable)) 
        
    
    def create_mm(self, name, d_add):
        model=name+'_multimeter'
        if model not in my_nest.Models():
            my_nest.CopyModel('multimeter', model )
        d={'active':False,
           'id':[],
           'model':name+'_multimeter', 
           'params': {'record_from':['V_m'], 
                      'start':0.0, 
                      'stop':numpy.inf,
                      'interval':1.,
                      'to_file':False,
                      'to_memory':True}} # recodring interval (dt) 
        d=misc.dict_update(d, d_add) 
        if d['active']:
            _id=my_nest.Create(model, params=d['params'])
            my_nest.DivergentConnect(_id, self.ids)
            d.update({'id':_id, 'model':model})
        return d
    
    def create_sd(self, name, d_add):
        model=name+'_spike_detector'
        if model not in my_nest.Models():
            my_nest.CopyModel('spike_detector', model )
            
        d={'active':False,
           'params': {"withgid": True, 
                      'to_file':False, 
                      'to_memory':True }} 
        d=misc.dict_update(d, d_add) 
        if d['active']:
            _id=my_nest.Create(model, params=d['params'])
            my_nest.ConvergentConnect(self.ids, _id )
            d.update({'id':_id, 'model':model})
        
        return d 

    def create_raw_spike_signal(self, start, stop):
    #signal=load:spikes()
        if self.sd['params']['to_file']:
            
            
            
            n_vp=my_nest.GetKernelStatus(['total_num_virtual_procs'])[0]
            data_path=my_nest.GetKernelStatus(['data_path'])[0]
            files=os.listdir(data_path)
            file_names=[data_path+s for s in files 
             if s.split('-')[0]==self.sd['model']]
                
#             network_size=str(my_nest.GetKernelStatus(['network_size'])[0])
            
#             gid = str(self.sd['id'][0])
#             gid = '0' * (len(network_size) - len(gid)) + gid
#             n = len(str(n_vp))
#             s=data_path + '/' + self.sd['model'] + '-' + gid + '-'  
#             file_names = [s+ '0'* (n - len(str(vp))) + str(vp) 
#                            + '.gdf' for vp in range(n_vp)]
#             
            s, t = my_nest.get_spikes_from_file(file_names)
            
        else:  
            
            s,t=my_nest.get_spikes_from_memory(self.sd['id'])
            
            e = my_nest.GetStatus(self.sd['id'])[0]['events'] # get events
            s = e['senders'] # get senders
            t = e['times'] # get spike times
        
            
            if comm.is_mpi_used():
                s,t=my_nest.collect_spikes_mpi(s,t)
                

        if stop:
            s, t = s[t < stop], t[t < stop] # Cut out data
            s, t = s[t >= start], t[t >= start] # Cut out data
        signal = zip(s, t)
        return signal

    def _create_signal_object(self, dataType, recordable='spikes', start=None, 
                              stop=None ):
        '''
        -_create_signal_object(self, self, dataType, recordable='times', stop=None )
        Creates NeuroTool signal object for the recordable simulation data.  

        
        Arguments:
        dataType        type of data. 's' or 'spikes' for spike data, 
                        'g' for conductance data, 'c' for current data and 
                        'v' for voltage data
        recordable      Need to be supplied for conductance, current and 
                        voltage data. It is the name of my_nest recorded data with
                        multimeter, e.g. V_m, I_GABAA_1, g_NMDA.
        stop            end of signal in ms
        
        About files in nest
        [model|label]-gid-vp.[dat|gdf]
        The first part is the name of the model (e.g. voltmeter or spike_detector) or, 
        if set, the label of the recording device. The second part is the global id 
        (GID) of the recording device. The third part is the id of the virtual process 
        the recorder is assigned to, counted from 0. The extension is gdf for spike 
        files and dat for analog recordings. The label and file_extension of a 
        recording device can be set like any other parameter of a node using SetStatus.
        '''
           
        ids=self.local_ids      # Eacj processor has its set of ids
            
        # Spike data
        if dataType in ['s', 'spikes']:    
            signal = self.create_raw_spike_signal(start,  stop) # create signal 
            
        # Mulitmeter data, conductance, current or voltage data    
        elif dataType in ['g', 'c', 'v']:     
            mm_dt=self.mm['params']['interval']

            e = my_nest.GetStatus(self.mm['id'])[0]['events']    # get events 
            v = e[recordable]                           # get analog value
            s = e['senders']                            # get senders
            t = e['times']                              # get spike times 
            #import pylab
            #pylab.plot(e['V_m'][e['senders'] ==12])
            #pylab.show()
            
            
            
            if start!=None and stop!=None:
                s=s[(t>start)*(t<stop)]   
                v=v[(t>start)*(t<stop)] 
                t=t[(t>start)*(t<stop)]    
                #start, stop=t[0], t[-1] 
            if stop: 
                s, v = s[t<=stop], v[t<=stop]    # Cut out data
                start = stop - len(s)/len(ids)*float(mm_dt)
            else:
                start = t[0]        # start time for NeuroTools  
#             start, stop=t[0]-mm_dt/2, t[-1]+mm_dt/2    
            signal  = zip( s, v )                   # create signal  
            #abs(self.t_stop-self.t_start - self.dt * len(self.signal)) > 0.1*self.dt
              
             
            
        if dataType in ['s', 'spikes']: signal = MySpikeList( signal, ids, start, 
                                                            stop)             
        if dataType in ['g']: signal = MyConductanceList(signal, ids, mm_dt, 
                                                       start,stop)
        if dataType in ['c']: signal = MyCurrentList(signal, ids, mm_dt, start,
                                                   stop)
        if dataType in ['v']: signal = MyVmList(signal, ids, mm_dt, start, 
                                              stop)    
        
        return signal   
    
    def get_signal(self, dataType, recordable='spikes', start=None, stop=None ):
        '''
        get_signal(self, self, dataType, recordable='spikes', stop=None )
        Sets group NeuroTool signal object for the recordable simulation data.  

        
        Arguments:
        dataType        type of data. 's' or 'spikes' for spike data, 
                        'g' for conductance data, 'c' for current data and 
                        'v' for voltage data
        recordable      Need to be supplied for conductance, current and 
                        voltage data. It is the name of my_nest recorded data with
                        multimeter, e.g. V_m, I_GABAA_1, g_NMDA.
        stop            end of signal in ms
        '''
        signal=self._create_signal_object(dataType, recordable, start, stop)
        return signal
             
 

        
    def get_target_rate(self):
        return self.target_rate

    def get_spike_signal(self):
        l=list(self.iter_spike_signals(self.sets))
        return SpikeListMatrix(l)     
    
    def get_voltage_signal(self):
        l=list(self.iter_voltage_signals(self.sets))
        return VmListMatrix(l)
        
    def iter_spike_signals(self, sets):
        for se in sets:
            ids_sliced=self.ids[se.get_slice()]
            s=self.spike_signal
            yield self.spike_signal.id_slice(ids_sliced)
                
    def iter_voltage_signals(self, sets):
        for se in sets:
            ids_sliced=self.ids[se.get_slice()]
            
            yield self.voltage_signal.id_slice(ids_sliced)   


                
    def isrecorded(self, flag):

        if flag in ['spike', 's', 'spikes', 'spike_signal']:
            v=self.sd['id']!=[] 
               
        elif flag in ['voltage', 'v', 'voltages', 'voltage_signal']:
            v= self.mm['id']!=[] 
        else:
            v=True   
        if not v:
            RuntimeError('{} traces are not recorded'.format(flag)) 
        return v  
    
    def model_par_randomize(self):     
        '''
        Example:
        self.randomization={ 'C_m':{'active':True, 
                                    'gaussian':{'sigma':0.2*C_m, 'my':C_m}}}
        '''
        pyrngs=self.set_random_states_nest()
        for p, val in self.rand.iteritems():
            if not val['active']:
                continue
            
            local_nodes=[]           
#             st=my_nest.GetStatus(self.ids, ['local', 'gloabal_id', 'vp'])
            for _id in self.ids:
                ni=my_nest.GetStatus([_id])[0]
                if ni['local']:
                    local_nodes.append((ni['global_id'], ni['vp']))
            
#             local_nodes=[(ni['global_id'], ni['vp']) 
#                          for ni in st if ni['local']]
            
            for gid, vp in local_nodes:
                val_rand=numpy.round(get_random_number(pyrngs, vp, val),2)
                my_nest.SetStatus([gid],{p:val_rand})     



#  def model_par_randomize(pyrngs, rand_dic, ids):     
#         '''
#         Example:
#         self.randomization={ 'C_m':{'active':True, 
#                                     'gaussian':{'sigma':0.2*C_m, 'my':C_m}}}
#         '''
# #         pyrngs=self.set_random_states_nest()
#         for p, val in rand_dic.iteritems():
#             if not val['active']:
#                 continue
#                        
#             st=my_nest.GetStatus(ids)
#             local_nodes=[(ni['global_id'], ni['vp']) 
#                          for ni in st if ni['local']]
#             
#             for gid, vp in local_nodes:
#                 my_nest.SetStatus([gid], 
#                                   {p:self.get_random_number(pyrngs, vp, val)})     

             
    def voltage_response(self,  currents, times, start, sim_time, id):
               
        scg = my_nest.Create( 'step_current_generator', n=1 )  
        my_nest.SetStatus(scg, {'amplitude_times':times,
                                'amplitude_values':currents})
        
        rec=my_nest.GetStatus([id])[0]['receptor_types']
        my_nest.Connect(scg, [id], params = { 'receptor_type' : rec['CURR'] })
        
        my_nest.MySimulate(sim_time)
        
        self.get_signal( 'v','V_m', start=start, stop=sim_time)    
        self.get_signal( 's')#, start=start, stop=sim_time)       
        self.signals['V_m'].my_set_spike_peak( 15, spkSignal= self.signals['spikes'] )               
        voltage=self.signals['V_m'][id].signal
        times=numpy.arange(0, len(voltage)*self.mm['params']['interval'], 
                           self.mm['params']['interval'])
        
        if len(times)!=len(voltage):
            raise Exception('The vectors has to be the same length')
        
        return times, voltage
class MyInput():
        def __init__(self,  **kwargs):
            
            self.ids             = []
            self._local_ids      = []
            self.model           = 'poisson_generator'
            
        def set_spike_times(self, rates=[], times=[], t_stop=None, ids=None, seed=None, idx=None):
            t_starts=times
            t_stops=list(times[1:])+list([t_stop])
            
            params =[{'rate':v[0],'start':v[1], 'stop':v[2]} for v in zip(rates, t_starts, t_stops)]
            
            if len(params)==1:
                ids=my_nest.Create('poisson_generator', len(params), params[0])
            else:
                ids=my_nest.Create('poisson_generator', len(params), params)
            
            self.ids=list(ids)
            self.local_ids=list(ids) # Nedd to put on locals also
                
        
class MyPoissonInput(MyGroup):
        
    def __init__(self, name, **kwargs):
        ''' 
        Constructor 
        
        Inherited attributes:
        self.t_start        = float(t_start)
        self.t_stop         = t_stop
        self.dt             = float(dt)
        self.dimensions     = dims
        self.analog_signals = {}
        
        New attributes:
        self.ids = sorted( id_list )     # sorted id list
        
        '''
        
        model=kwargs.get('model', 'poisson_generator')
        
        df=my_nest.GetDefaults(model)
        if df['type_id'] in ['spike_generator','mip_generator',
                             'poisson_generator','poisson_generator_dynamic' ]:
            input_model=model
            type_model=df['type_id'] 
            kwargs['model']='parrot_neuron'
            
        super( MyPoissonInput, self ).__init__(name, **kwargs)            

        
        self._init_extra_attributes(input_model, type_model, **kwargs)

        if self.spike_setup!=[]:
            for k in self.spike_setup: 
                self.set_spike_times(**k)
             

    def _init_extra_attributes(self, input_model, type_model, **kwargs):
        
        # Add new attribute
        self.spike_setup=kwargs.get('spike_setup', [])
        self.input_model=input_model
        self.type_model=type_model
        self.ids_generator={}
     
    
    def set_spike_times(self, rates=[], times=[], t_stop=None, ids=None, 
                        seed=None, idx=None):
        
        df=my_nest.GetDefaults(self.input_model)['model']

        if ids is None and (not idx is None):            
            tmp_ids=numpy.array(self.ids)
            ids=list(tmp_ids[idx])
        if ids is None:            
            ids=self.ids 
                
        # Spike generator
        if 'spike_generator' == self.type_model:
            for id in ids:
                seed=random_integers(0,10**5)
                          
                spikeTimes=misc.inh_poisson_spikes( rates, times, t_stop=t_stop, n_rep=1, seed=seed)
                if any(spikeTimes):
                    my_nest.SetStatus([id], params={'spike_times':spikeTimes})
                    
        
        # MIP
        elif 'mip_generator' == self.type_model:
            c=df['p_copy']
            
            seed=random_integers(0,10**6)
            new_ids=[]
            t_starts=times
            t_stops=times[1:]+[t_stop]
            for id in ids:
                i=0
                for r, start, stop in rates, t_starts, t_stops:               
                    r_mother=r/c
                    params={'rate':r_mother, 'start':start, 'stop':stop,
                           'p_copy':c, 'mother_seed':seed}
                    if i==0:
                        my_nest.SetStatus(id, params)
                    else:                   
                        new_id=my_nest.Create('mip_generator', 1, params)
                        
                    new_ids.append(new_id)
            self.ids.append(new_ids)  
        
        # Poisson generator
        if 'poisson_generator' == self.type_model:

            t_starts=times
            t_stops=list(times[1:])+list([t_stop])
            
            params =[{'rate':v[0],'start':v[1], 'stop':v[2]} 
                     for v in zip(rates, t_starts, t_stops)]
            
            if len(params)==1:
                source_nodes=my_nest.Create('poisson_generator', 
                                            len(params), 
                                            params[0])*len(ids)
            else:
                source_nodes=my_nest.Create('poisson_generator', 
                                            len(params), params)*len(ids)
            
            target_nodes = numpy.array([[id_]*len(rates) for id_ in ids])      
            target_nodes = list(numpy.reshape(target_nodes, 
                                              len(rates)*len(ids), order='C'))    
            
            my_nest.Connect(source_nodes, target_nodes)         
            
            generators=[]
            if hash(tuple(ids)) in self.ids_generator.keys():
                generators=self.ids_generator[hash(tuple(idx))]
                
            generators=list(set(source_nodes).union(generators))
            self.ids_generator[hash(tuple(idx))]=sorted(generators)
            self.local_ids=list(self.ids) # Nedd to put on locals also
        
        if 'poisson_generator_dynamic'==self.type_model:
            source_nodes=my_nest.Create(self.type_model, 1, 
                                        {'timings':times,
                                         'rates':rates})*len(ids)
            target_nodes=ids
            my_nest.Connect(source_nodes, target_nodes)         
            
            generators=[]
            if hash(tuple(ids)) in self.ids_generator.keys():
                generators=self.ids_generator[hash(tuple(idx))]
                
            generators=list(set(source_nodes).union(generators))
            self.ids_generator[hash(tuple(idx))]=sorted(generators)
            
#             v=my_nest.GetStatus(ids, 'local')
#             self.local_ids=[_id for _id in zip(ids,v) if  # Nedd to put on locals also
           
 
    def update_spike_times(self,rates=[], times=[], t_stop=None, ids=None, seed=None, idx=None):
        if 'poisson_generator' == self.type_model:

            t_starts=times
            t_stops=list(times[1:])+list([t_stop])
            
            params =[{'rate':v[0],'start':v[1], 'stop':v[2]} 
                     for v in zip(rates, t_starts, t_stops)] 
            my_nest.SetStatus(self.ids_generator[hash(tuple(idx))], params)
            
                                    
class MyLayerGroup(MyGroup):
    
    def __init__(self, layer_props={}, model = 'iaf_neuron', n=1, params = {}, mm_dt=1.0, 
                 sname='', spath='', sname_nb=0, sd=False, sd_params={},
                 mm=False, record_from=[]): 
                 
                layer=my_topology.CreateLayer(layer_props)
                ids=my_nest.GetLeaves(layer)[0]
                model=layer_props['elements']
                super( MyLayerGroup, self ).__init__(  model, n, params, mm_dt, 
                                                   sname, spath, sname_nb, sd, 
                                                   sd_params, mm, record_from, ids)
                self._init_extra_attributes_layer_group(layer, layer_props) 
    


        
        
    def _init_extra_attributes_layer_group(self, layer, props):
              
        # Add new attribute
        self.layer_id=layer
        self.conn={}
        self.id_mod=[]
        
    def add_connection(self, source, type, props):
        ''' Store connection properties of an incomming synapse' ''' 
        name=str(source) + '_'+type
        self.conn[name]={}
        self.conn[name]['mask']=props['mask']
        self.conn[name]['kernel']=props['kernel']   
        
    def get_kernel(self, source, type):
        name=str(source) + '_'+type
        
        return self.conn[name]['kernel']
    
    def get_mask(self, source, type):
        name=str(source) + '_'+type
        
        return self.conn[name]['mask']

    def IF( self, I_vec, id = None, tStim = None ):    
        '''
        Function that creates I-F curve
        Inputs:
                id      - id of neuron to use for calculating 
                                 I-F relation
                tStim   - lenght of each step current injection 
                                in miliseconds
                I_vec   - step currents to inject
        
        Returns: 
                fIsi          - first interspike interval 
                mIsi          - mean interspike interval

                
        Examples:
                >> n  = nest.Create('izhik_cond_exp')
                >> sc = [ float( x ) for x in range( 10, 270, 50 ) ]
                >> f_isi, m_isi, l_isi = IF_curve( id = n, sim_time = 500, I_vec = sc ):
        '''
        
        
        if not id: id=self.ids[0]
        if isinstance( id, int ): id =[ id ] 

        
        fIsi, mIsi, lIsi  = [], [], []  # first, mean and last isi
        if not tStim: tStim = 500.0 
#         tAcum = my_nest.GetKernelStatus('time')   
        tAcum = my_nest.GetKernelTime()
    
        I_e0=my_nest.GetStatus(id)[0]['I_e'] # Retrieve neuron base current
        
        
        for I_e in I_vec:
            
            my_nest.SetStatus( id, params = { 'I_e': float(I_e+I_e0) } )   
            my_nest.SetStatus( id, params = { 'V_m': float(-61) } )            
            #my_nest.SetStatus( id, params = { 'w': float(0) } )
            my_nest.SetStatus( id, params = { 'u': float(0) } )
            simulate=True
            tStart=tAcum
            while simulate:
                my_nest.Simulate( tStim )
                tAcum+=tStim    
                
                self.get_signal('s', start=tStart, stop=tAcum)                          
                signal=self.signals['spikes'].time_slice(tStart, tAcum)
                signal=signal.id_slice(id)
                
                if signal.mean_rate()>0.1 or tAcum>20000:
                    simulate=False
                    
            isi=signal.isi()[0]      
            if not any(isi): isi=[1000000.] 
              
            fIsi.append( isi[ 0 ] )            # retrieve first isi
            mIsi.append( numpy.mean( isi ) )   # retrieve mean isi
            lIsi.append( isi[ -1 ] )           # retrieve last isi
        
        fIsi=numpy.array(fIsi)
        mIsi=numpy.array(mIsi)
        lIsi=numpy.array(lIsi)
        I_vec=numpy.array(I_vec)
            
        return I_vec, fIsi, mIsi, lIsi


    def I_PSE(self, I_vec, synapse_model, id=0,  receptor='I_GABAA_1'):
        '''
        Assure no simulations has been run before this (reset kernel).
        Function creates relation between maz size of postsynaptic 
        event (current, conductance, etc). The type is set by receptor. 
        
        Inputs:
            I_vec          - step currents to clamp at
            id             - id of neuron to use for calculating I-F relation
                             If not providet id=ids[0]
     
        Returns: 
            v_vec          - voltage clamped at
            size_vec       - PSE size at each voltage 
                
        Examples:
                >> n  = my_nest.Create('izhik_cond_exp')
                >> sc = [ float( x ) for x in range( -300, 100, 50 ) ] 
                >> tr_t, tr_v, v_ss = IV_I_clamp( id = n, tSim = 500, 
                I_vec = sc ):
        '''
        
        vSteadyState=[]
        
        if not id: id=self.ids[0]
        if isinstance( id, int ): id =[ id ]
        
        simTime  = 700. # ms
        spikes_at = numpy.arange(500., len(I_vec)*simTime,simTime) # ms
        
    
        voltage=[] # mV
        pse=[] # post synaptic event
        
        sg = my_nest.Create('spike_generator', params={'spike_times':spikes_at} )
    
        my_nest.Connect(sg, id, model=synapse_model)
        
        simTimeTot=0
        for I_e in I_vec:
            
            my_nest.SetStatus(self[:], params={'I_e':float(I_e)})
            my_nest.MySimulate(simTime)
            simTimeTot+=simTime
        
        
        self.get_signal( receptor[0].lower(),receptor, 
                         stop=simTimeTot ) # retrieve signal
        
        simTimeAcum=0
        
        
        for I_e in I_vec:
            
            
            size=[]
   
            signal=self.signals[receptor].my_time_slice(400+simTimeAcum, 
                                                        700+simTimeAcum)
            simTimeAcum+=simTime     
            # First signal object at position 1    
            clamped_at  = signal[1].signal[999]
            minV=min(signal[1].signal)
            maxV=max(signal[1].signal)
            if abs(minV-clamped_at)<abs(maxV-clamped_at):        
                size.append(max(signal[1].signal)-clamped_at)
                
            else:
                size.append(min(signal[1].signal)-clamped_at)
        
            voltage.append(clamped_at)
            pse.append(size[0])
                                   
    
        return voltage, pse
         
    def IV_I_clamp(self, I_vec, id = None, tStim = 2000):    
        '''
        Assure no simulations has been run before this (reset kernel).
        Function that creates I-V by injecting hyperpolarizing currents 
        and then measuring steady-state membrane (current clamp). Each 
        trace is preceded and followed by 1/5 of the simulation time 
        of no stimulation
        Inputs:
            I_vec          - step currents to inject
            id             - id of neuron to use for calculating I-F relation
            tStim          - lenght of each step current stimulation in ms

        Returns: 
            I_vec          - current for each voltage
            vSteadyState   - steady state voltage 
                
        Examples:
                >> n  = my_nest.Create('izhik_cond_exp')
                >> sc = [ float( x ) for x in range( -300, 100, 50 ) ] 
                >> tr_t, tr_v, v_ss = IV_I_clamp( id = n, tSim = 500, 
                I_vec = sc ):
        '''
        
        vSteadyState=[]
        
        if not id: id=self.ids[0]
        if isinstance( id, int ): id =[ id ] 
        
                                                                           
        tAcum  = 1    # accumulated simulation time, step_current_generator
                      # recuires it to start at t>0    
        
        scg = my_nest.Create( 'step_current_generator' )  
        rec=my_nest.GetStatus(id)[0]['receptor_types']
        my_nest.Connect( scg, id, params = { 'receptor_type' : rec['CURR'] } )             
        
        ampTimes=[]
        ampValues=[]
        for I_e in I_vec:

            ampTimes.extend([ float(tAcum) ])
            ampValues.extend([ float(I_e) ])
            tAcum+=tStim
            
        my_nest.SetStatus( scg, params = { 'amplitude_times':ampTimes,       
                                           'amplitude_values' :ampValues } )   
        my_nest.Simulate(tAcum)    
        
        self.get_signal( 'v','V_m', stop=tAcum ) # retrieve signal
        self.get_signal('s')
        if 0 < self.signals['spikes'].mean_rate():
            print 'hej'
        tAcum  = 1 
        for I_e in I_vec:
            
            if 0>=self.signals['spikes'].mean_rate(tAcum+10, tAcum+tStim):
                signal=self.signals['V_m'].my_time_slice(tAcum+10, tAcum+tStim)
                vSteadyState.append( signal[1].signal[-1] )
            tAcum+=tStim
    
            
        I_vec=I_vec[0:len(vSteadyState)]         
        return I_vec, vSteadyState    
    def stat_connections(self, source_layer):
        ''' Return mean and std of number of outgoing connections'''
        
        target_layer=self
        targets_per_source=[]
        #target_nodes=my_topology.GetTargetNodes(source_layer.ids, target_layer.layer_id)[0]
        
        
        for id in source_layer.ids:
            
            target_nodes=my_topology.GetTargetNodes([id], target_layer.layer_id)[0]
            targets_per_source.append(len(target_nodes))
        
        mean_targets_per_source=numpy.mean(numpy.array(targets_per_source))
        std_targets_per_source=numpy.std(numpy.array(targets_per_source))
        
        mean_source_per_target=mean_targets_per_source*len(source_layer.ids)/len(target_layer.ids)
        std_source_per_target=std_targets_per_source*len(source_layer.ids)/len(target_layer.ids)
        
        
        return (mean_targets_per_source, std_targets_per_source, 
                mean_source_per_target, std_source_per_target)
            
                
    def plot(self, ax=None, nodecolor='b', nodesize=20):        
        my_topology.MyPlotLayer(self.layer_id, ax,nodecolor, nodesize)
    
    def plot_targets(self, type, source_layer, src_nrn=None, 
                     tgt_model=None, syn_type=None, ax=None,
                     src_color='red', src_size=50, tgt_color='red', tgt_size=20,
                     mask_color='red', kernel_color='red'):
        
        if not src_nrn:
            src_nrn = my_topology.FindCenterElement ( source_layer.layer_id)
        
        kernel=self.get_kernel(source=source_layer, type=type) 
        mask=self.get_mask(source=source_layer, type=type)
        
        ax, n_targets=my_topology.MyPlotTargets(src_nrn, self.layer_id, ax = ax, 
                   mask=mask , kernel=kernel, src_size=src_size , 
                   tgt_color=tgt_color , tgt_size=tgt_size , 
                   kernel_color =kernel_color)
        return n_targets

class MyLayerPoissonInput(MyPoissonInput):
    
    def __init__(self, layer_props={},  ids=[], model = 'spike_generator', n=1, params = {}, mm_dt=1.0, 
                 sname='', spath='', sname_nb='', sd=False, sd_params={},
                 mm=False, record_from=[]):
                 
                layer=my_topology.CreateLayer(layer_props)
                ids=my_nest.GetLeaves(layer)[0]
                
                super( MyLayerPoissonInput, self ).__init__(model, n, params, mm_dt, 
                                                   sname, spath, sname_nb, sd, 
                                                   sd_params, mm, record_from,ids)
                self._init_extra_attributes_poisson_input(layer, layer_props)
          
    def _init_extra_attributes_poisson_input(self, layer, props):
        
        
        # Add new attribute
        self.layer_id=layer
        self.conn={}
        self.id_mod=[]
    
    def sort_ids(self, pos=[[0,0]]):     
        node=my_topology.FindCenterElement(self.layer_id)

        ids=self.ids
        d=numpy.array(my_topology.Distance(node*len(ids),ids))
        idx=sorted(range(len(d)), key=d.__getitem__, reverse=True)

        return idx 
    def plot(self, ax=None, nodecolor='b', nodesize=20):        
        my_topology.MyPlotLayer(self.layer_id, ax,nodecolor, nodesize)
    
def default_kwargs_net(n, n_sets):
    
    sets=[misc.my_slice(i, n,n_sets) for i in range(n_sets)]
    return  {'n':n, 
             'model':'iaf_cond_exp', 
             'mm':{'active':True,
                   'params':{'interval':1.0,
                             'to_memory':True, 
                             'to_file':False,
                             'record_from':['V_m']}},
             'params':{'I_e':280.0,
                       'C_m':200.0},
             'sd':{'active':True,
                   'params':{'to_memory':True, 
                             'to_file':False}},
             'sets':sets,
             'rate':10.0}

def default_kwargs_inp(n):
    
    sets=[misc.my_slice(i, n,1) for i in range(1)]
    return  {'n':n, 
             'model':'poisson_generator', 
             'sets':sets,
             'rate':10.0}


def default_spike_setup(n, stop):
    d={'rates':[10.0],
       'times':[1.0],
       't_stop':stop,
       'idx':range(n)}
    return d

def get_random_number(pyrngs, vp, val):
    
    
    if 'gaussian' in val.keys():
        par=val['gaussian']
        r=pyrngs[vp].normal(loc=par['my'], scale=par['sigma'])
        if 'cut' in par.keys():
            if r<par['my']-par['cut_at']*par['sigma']:
                r=par['my']-par['cut_at']*par['sigma']
            elif r>par['my']+par['cut_at']*par['sigma']:
                r=par['my']+par['cut_at']*par['sigma']
                
        return r
    
    if 'uniform' in val.keys():
        par=val['uniform']
        return pyrngs[vp].uniform(par['min'], par['max'])

def sim_group(sim_time, *args, **kwargs):
        
    g=MyNetworkNode(*args, **kwargs)
    my_nest.Simulate(sim_time)
    return g
    
        
from os.path import expanduser
import subprocess
class TestModule_functions(unittest.TestCase):
    
    def setUp(self):
        self.home=expanduser("~")
    
    def test_collect_spikes_mpi(self):

        data_path= self.home+('/results/unittest/my_population'
                         +'/collect_spikes_mpi/')
        script_name=os.getcwd()+('/test_scripts_MPI/'
                                 +'my_population_collect_spikes_mpi.py')
        
        np=4
        s0=[]
        for i in range(4):
            s0+=[float(i),float(i)]
        
        s0=numpy.array(s0)
        e0=numpy.array(s0)+1
        
        p=subprocess.Popen(['mpirun', '-np', str(np), 'python', 
                            script_name, data_path],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
        
        out, err = p.communicate()
#         print out
#         print err
        
        import pickle    

        f=open(data_path+'data.pkl', 'rb') #open in binary mode

        s1,e1=pickle.load(f)
        s1,e1=sorted(s1), sorted(e1)
        
        self.assertListEqual(list(s0), list(s1))
        self.assertListEqual(list(e0), list(e1))
        
        f.close()

from toolbox.data_to_disk import pickle_save, pickle_load
class TestMyNetworkNode(unittest.TestCase):
    
    my_nest.sr("M_WARNING setverbosity") #silence nest output
    
    #print my_nest.GetKernelStatus()

    def setUp(self):
        self.home=expanduser("~")
        self.n=12
        self.n_sets=3
        self.args=['unittest']
        self.kwargs=default_kwargs_net(self.n, self.n_sets)
        self.sim_time=10000.
        dp=self.home+'/results/unittest/my_population/nest/'
        data_to_disk.mkdir(dp)
        my_nest.ResetKernel(display=False, data_path=dp)
        my_nest.SetKernelStatus({'overwrite_files':True})
    
    def sim_group(self, **kwargs):
        kwargs=misc.dict_update(self.kwargs, kwargs)
        return sim_group(self.sim_time, *self.args, **self.kwargs)  
        
    def test_1_create(self):
        g=MyNetworkNode(*self.args, **self.kwargs)
     
    def test_21_get_spike_signal_from_memory(self):
        l=self.sim_group().get_spike_signal()
        self.assertEqual(l.shape[1], self.n_sets)
         
        mr=0
        for _,_,spk_list in my_signals.iter2d(l):
            mr+=spk_list.mean_rate()/l.shape[1]
        self.assertAlmostEqual(mr, 55.0, delta=0.1)

    def test_22_get_spike_signal_from_file(self):
        d={'sd':{'active':True,
                   'params':{'to_memory':False, 
                             'to_file':True}}}
        l=self.sim_group(**d).get_spike_signal()
        self.assertEqual(l.shape[1], self.n_sets)
         
        mr=0
        for _,_,spk_list in my_signals.iter2d(l):
            mr+=spk_list.mean_rate()/l.shape[1]
        self.assertAlmostEqual(mr, 55.0, delta=0.1)
    
    def test_3_get_voltage_signal(self):
        l=self.sim_group().get_voltage_signal()
        self.assertEqual(l.shape[1], self.n_sets)
 
    def test_4_multiple_threads(self):
        my_nest.SetKernelStatus({'local_num_threads':2})
        _=self.sim_group().get_spike_signal()
     
    def test_5_load_from_disk(self):
        from os.path import expanduser
        s = expanduser("~")
        s= s+'/results/unittest/my_population'
        data_to_disk.mkdir(s)
        my_nest.SetKernelStatus({'local_num_threads':2,
                                 'data_path':s,
                                 'overwrite_files': True,})
         
        self.kwargs['sd']['params'].update({'to_memory':False, 
                                            'to_file':True})
         
        g=self.sim_group().get_spike_signal()
        g[0].firing_rate(1, display=True)
#         pylab.show()  
        for filename in os.listdir(s):
            if filename.endswith(".gdf"):
                os.remove(s+'/'+filename)        
        

    def do_mpi(self, data_path, script_name, np):
        fileName = data_path + 'data_in.pkl'
        fileOut = data_path + 'data_out.pkl'
        pickle_save([self.sim_time, self.args, self.kwargs], fileName)
        p = subprocess.Popen(['mpirun', '-np', str(np), 'python', 
                              script_name, fileName, fileOut, data_path], 
    #                            stdout=subprocess.PIPE,
    #                            stderr=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        out, err = p.communicate()
    #         print out
    #         print err
        l = self.sim_group().get_spike_signal()
        mr0 = l.mean_rate()
        fr0 = l.firing_rate(1)
        return fileOut, mr0, fr0

    def test_61_mpi_run(self):
        data_path= self.home+'/results/unittest/my_population/mpi_run/'
        script_name=os.getcwd()+'/test_scripts_MPI/my_population_mpi_run.py'
        np=4
        

        
        fileOut, mr0, fr0 = self.do_mpi(data_path, script_name, np)
        
        mr1, fr1=pickle_load(fileOut)      
        self.assertEqual(mr0, mr1)
        self.assertListEqual(list(fr0), list(fr1))
        
    def test_62_mpi_run_from_disk(self):
        data_path= self.home+'/results/unittest/my_population/mpi_run_from_disk/'
        script_name=os.getcwd()+'/test_scripts_MPI/my_population_mpi_run_from_disk.py'
        np=4
       
        for filename in os.listdir(data_path):
            path=data_path+'/'+filename
            if os.path.isfile(path):
                os.remove(path)   
            
       
        fileOut, mr0, fr0 = self.do_mpi(data_path, script_name, np)
        
        ss=pickle_load(fileOut)      
        mr1=ss.mean_rate()
        fr1=ss.firing_rate(1)        

        self.assertEqual(mr0, mr1)
        self.assertListEqual(list(fr0), list(fr1))
                 

    def test_63_mpi_run_hyprid_from_disk(self):
        data_path= self.home+'/results/unittest/my_population/mpi_run_hybrid_from_disk/'
        script_name=os.getcwd()+'/test_scripts_MPI/my_population_mpi_run_hybrid_from_disk.py'
        np=4
            
            
#         os.environ['OMP_NUM_THREADS'] = '2'   
        fileName = data_path + 'data_in.pkl'
        fileOut = data_path + 'data_out.pkl'
        pickle_save([self.sim_time, self.args, self.kwargs], fileName)
        p = subprocess.Popen(['mpirun',  '-np', str(np), 'python', 
                              script_name, fileName, fileOut, data_path], 
    #                            stdout=subprocess.PIPE,
    #                            stderr=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        out, err = p.communicate()
    #         print out
    #         print err
        l = self.sim_group().get_spike_signal()
        mr0 = l.mean_rate()
        fr0 = l.firing_rate(1)
        return fileOut, mr0, fr0
        
        g=pickle_load(fileOut)      
        ss=g.get_spike_signal()
        mr1=ss.mean_rate()
        fr1=ss.firing_rate(1)        

        self.assertEqual(mr0, mr1)
        self.assertListEqual(list(fr0), list(fr1))     
            
class TestMyPoissonInput(unittest.TestCase):
    def setUp(self):
        
        self.n=1
        self.n_inp=10
        self.n_sets=1
        self.args_net=['unittest_net']
        self.args_inp=['unittest_inp']
        self.kwargs_net=default_kwargs_net(self.n, self.n_sets)
        self.kwargs_net['params'].update({'I_e':200.0})
        self.kwargs_inp=default_kwargs_inp(self.n_inp)
        self.sim_time=1000.
        my_nest.ResetKernel(display=False)
        
    @property
    def sim_group(self):
        g=MyNetworkNode(*self.args_net, **self.kwargs_net)
        i=MyPoissonInput(*self.args_inp, **self.kwargs_inp)
        i.set_spike_times(**default_spike_setup(self.n_inp, self.sim_time))
        my_nest.Connect(i.ids, 
                        g.ids*len(i.ids), 
                        params={'delay':1.0,
                                'weight':10.0}, 
                        model='static_synapse')
        my_nest.Simulate(self.sim_time)
        return g      
    
    def test_1_create(self):
        _=MyPoissonInput(*self.args_inp, **self.kwargs_inp)    

    def test_2_simulate_show(self):
        g=self.sim_group.get_voltage_signal()[0].plot(display=True) 
#         pylab.show()


               
if __name__ == '__main__':
    d={
        TestModule_functions:[
#                               'test_collect_spikes_mpi',
                              ],
        TestMyNetworkNode:[
#                            'test_1_create', 
#                             'test_21_get_spike_signal_from_memory', 
                            'test_22_get_spike_signal_from_file',
#                            'test_3_get_voltage_signal',
#                            'test_4_multiple_threads', 
#                            'test_5_load_from_disk', 
#                             'test_61_mpi_run',
#                             'test_62_mpi_run_from_disk',
#                             'test_63_mpi_run_hyprid_from_disk',
                         ],
       TestMyPoissonInput:[
#                            'test_create',
#                            'test_2_simulate_show',
                           ],

       }
    test_classes_to_run=d
    suite = unittest.TestSuite()
    for test_class, val in  test_classes_to_run.items():
        for test in val:
            suite.addTest(test_class(test))

    unittest.TextTestRunner(verbosity=2).run(suite)         