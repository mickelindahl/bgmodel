"""'
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
"""

import numpy
import os
import unittest
import pprint
import python.core.directories as dr
from numpy.random import randint as random_integers
from python.core import my_signals
from python.core import my_nest
from python.core.my_signals import CondunctanceListMatrix
from python.core.my_signals import (MyConductanceList, MyCurrentList,
                                    MyVmList, MySpikeList, SpikeListMatrix,
                                    VmListMatrix)
from python.core import data_to_disk, misc
from python.core.parallelization import comm, Barrier

pp = pprint.pprint


# from numpy.random import RandomState
# random_integers  = RandomState(3).random_integers


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

    def __init__(self, name, **kwargs):
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

        model = kwargs.get('model', 'iaf_neuron')
        n = kwargs.get('n', 1)
        params = kwargs.get('params', {})

        node_collection = my_nest.Create(model, n, params)
        ids = kwargs.get('ids', node_collection.tolist())#.get('global_id')
        self._ids = slice(ids[0], ids[-1], 1)

        self.local_ids = []
        self.local_ids = my_nest.GetLocalNodeCollection(node_collection).tolist()

        # for _id in self.ids:
        #     if my_nest.GetStatus([_id], 'local'):
        #         self.local_ids.append(_id)

        self.model = model
        self.name = name
        self.n = n
        self.sets = kwargs.get('sets', [misc.my_slice(0, n, 1)])

    @property
    def ids(self):
        return list(range(self._ids.start, self._ids.stop + 1, self._ids.step))

    def __getitem__(self, key):
        ''' 
        Calling self with one index
        '''
        ids = numpy.array(self.ids)
        ids = ids[key]

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
        return self.__class__.__name__ + ':' + self.name

    def __str__(self):
        return self.__class__.__name__ + ':' + self.name

    def __iter__(self):
        for i in self.ids:
            yield i

    def get(self, attr, **k):

        if hasattr(self, 'get_' + attr):
            call = getattr(self, 'get_' + attr)
        elif hasattr(self, attr):
            call = getattr(self, attr)
        else:
            return None

        if self.isrecorded(attr):
            return call(**k)
        else:
            return None

    def get_name(self):
        return self.name

    def set_random_states_nest(self):
        msd = 1000  # masterseed

        n_vp = my_nest.GetKernelStatus('total_num_virtual_procs')

        msdrange1 = range(msd, msd + n_vp)

        pyrngs = [numpy.random.RandomState(s) for s in msdrange1]
        # msdrange2 = range(msd + n_vp + 1, msd + 1 + 2 * n_vp)
        my_nest.SetKernelStatus(params={'rng_seeds': msd})
        return pyrngs


class VolumeTransmitter(MyGroup):
    def __init__(self, *args, **kwargs):
        super(VolumeTransmitter, self).__init__(*args, **kwargs)
        self._init_extra_attributes(**kwargs)

    def _init_extra_attributes(self, *args, **kwargs):
        self.syn_target = kwargs.get('syn_target', '')

    def get_syn_target(self):
        return self.syn_target


class MyNetworkNode(MyGroup):
    def __init__(self, *args, **kwargs):

        super(MyNetworkNode, self).__init__(*args, **kwargs)

        self._init_extra_attributes(**kwargs)

    def _init_extra_attributes(self, *args, **kwargs):
        # Add new attribute

        #         self.sd_params=kwargs.get('sd_parans')
        self.mm = self.create_mm(self.name, kwargs.get('mm', {}))

        self.rand = kwargs.get('rand', {})

        self.receptor_types = my_nest.GetDefaults(self.model).get('receptor_types', {})
        self.sd = self.create_sd(self.name, kwargs.get('sd', {}))

        self.signals = {}  # dictionary with signals for current, conductance, voltage or spikes
        self._signaled = {}  # for ech signal indicates if it have been loaded from nest or not
        self.target_rate = kwargs.get('rate', 10.0)

        if not {} == self.rand:
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
                self._signaled[rec] = False
            self._signaled['spikes'] = False
        #             self._signaled=signaled
        return self._signaled

    @property
    def spike_signal(self):
        return self._signal('s', 'spikes')

    @property
    def voltage_signal(self):
        return self._signal('v', 'V_m')

    def is_new_recording(self, flag, recordable):
        if flag in ['s', 'spikes']:
            #             stop=max(1,my_nest.GetKernelStatus('time')-1)
            stop = max(1, my_nest.GetKernelTime() - 1)
        else:
            #             stop=my_nest.GetKernelStatus('time')
            stop = my_nest.GetKernelTime()
        if self.signals[recordable].t_stop == stop:
            return False
        else:
            return True

    def _signal(self, flag, recordable):
        if self.signaled[recordable]:
            if not self.is_new_recording(flag, recordable):
                return self.signals[recordable]
            else:
                start = self.signals[recordable].t_stop
        else:
            start = 0.0

        if not (start == my_nest.GetKernelTime()):
            if flag in ['s', 'spikes']:
                start = max(0, start - 1)
                # Todo with delay in spike recording I experienced
                stop = max(1, my_nest.GetKernelTime() - 1)

            else:
                stop = my_nest.GetKernelTime()

            signal = self.get_signal(flag, recordable=recordable,
                                     start=start,
                                     stop=stop)
            if flag in ['s', 'spikes']:
                signal.complete(self.ids)
            self.signals[recordable] = signal
            self.signaled[recordable] = True
        try:
            return self.signals[recordable]
        except:
            raise KeyError("{} signals not present".format(recordable))

    def create_mm(self, name, d_add, **kw):
        model = name + '_multimeter'
        if model not in my_nest.get_models():
            my_nest.CopyModel('multimeter', model)
        d = {'active': False,
             'id': [],
             'model': name + '_multimeter',
             'params': {'record_from': ['V_m'],
                        'start': 0.0,
                        'stop': numpy.inf,
                        'interval': 1.,
                        'record_to': 'memory'
                        }}  # recodring interval (dt)
        d = misc.dict_update(d, d_add)
        if d['active']:
            _id = my_nest.Create(model, params=d['params'])
            if 'slice' in kw.keys():
                my_nest.DivergentConnect(_id, self.ids[kw['slice']])
            else:
                my_nest.DivergentConnect(_id, self.ids)

            d.update({'id': _id, 'model': model})
        return d

    def create_sd(self, name, d_add):
        model = name + '_spike_detector'
        if model not in my_nest.get_models():
            my_nest.CopyModel("spike_recorder", model)

        d = {'active': False,
             'params': {
                 # "withgid": True,
                 'start': 0.0,
                 'stop': numpy.inf,
                 'record_to': 'memory'}}
        d = misc.dict_update(d, d_add)
        if d['active']:
            _id = my_nest.Create(model, params=d['params'])

            my_nest.ConvergentConnect(self.ids, _id)

            d.update({'id': _id, 'model': model})

        return d

    def create_raw_spike_signal(self, start, stop):
        # signal=load:spikes()
        if self.sd['params']['record_to'] == 'file':

            n_vp = my_nest.GetKernelStatus(['total_num_virtual_procs'])[0]
            data_path = my_nest.GetKernelStatus(['data_path'])[0]
            files = os.listdir(data_path)
            file_names = [data_path + s for s in files
                          if s.split('-')[0] == self.sd['model']]

            s, t = my_nest.get_spikes_from_file(file_names)

        else:

            s, t = my_nest.get_spikes_from_memory(self.sd['id'])

            e = my_nest.GetStatus(self.sd['id'])[0]['events']  # get events
            s = e['senders']  # get senders
            t = e['times']  # get spike times

            if comm.is_mpi_used():
                s, t = my_nest.collect_spikes_mpi(s, t)

        if stop:
            s, t = s[t < stop], t[t < stop]  # Cut out data
            s, t = s[t >= start], t[t >= start]  # Cut out data
        signal = list(zip(s, t))
        return signal

    def _create_signal_object(self, dataType, recordable='spikes', start=None,
                              stop=None):
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

        ids = self.local_ids  # Eacj processor has its set of ids

        # Spike data
        if dataType in ['s', 'spikes']:
            signal = self.create_raw_spike_signal(start, stop)  # create signal

        # Mulitmeter data, conductance, current or voltage data    
        elif dataType in ['g', 'c', 'v']:
            mm_dt = self.mm['params']['interval']

            e = my_nest.GetStatus(self.mm['id'])[0]['events']  # get events
            v = e[recordable]  # get analog value
            s = e['senders']  # get senders
            t = e['times']  # get spike times
            # import pylab
            # pylab.plot(e['V_m'][e['senders'] ==12])
            # pylab.show()

            if start != None and stop != None:
                s = s[(t > start) * (t < stop)]
                v = v[(t > start) * (t < stop)]
                t = t[(t > start) * (t < stop)]
                # start, stop=t[0], t[-1]
            if stop:
                s, v = s[t <= stop], v[t <= stop]  # Cut out data
                start = stop - len(s) / len(ids) * float(mm_dt)
            else:
                start = t[0]  # start time for NeuroTools
            #             start, stop=t[0]-mm_dt/2, t[-1]+mm_dt/2
            signal = list(zip(s, v))  # create signal
            # abs(self.t_stop-self.t_start - self.dt * len(self.signal)) > 0.1*self.dt

        if dataType in ['s', 'spikes']: signal = MySpikeList(signal, ids, start,
                                                             stop)
        if dataType in ['g']: signal = MyConductanceList(signal, ids, mm_dt,
                                                         start, stop)
        if dataType in ['c']: signal = MyCurrentList(signal, ids, mm_dt, start,
                                                     stop)
        if dataType in ['v']: signal = MyVmList(signal, ids, mm_dt, start,
                                                stop)

        return signal

    def get_signal(self, dataType, recordable='spikes', start=None, stop=None):
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
        signal = self._create_signal_object(dataType, recordable, start, stop)
        return signal

    def get_target_rate(self):
        return self.target_rate

    def get_spike_signal(self):
        l = list(self.iter_spike_signals(self.sets))
        return SpikeListMatrix(l)

    def get_voltage_signal(self):
        l = list(self.iter_voltage_signals(self.sets))
        return VmListMatrix(l)

    def get_conductance_signals(self):
        d = {}
        for key in self.conductance_signals.keys():
            l = list(self.iter_voltage_signals(self.sets))
            d[key] = CondunctanceListMatrix(l)
        return d

    def iter_spike_signals(self, sets):
        for se in sets:
            ids_sliced = self.ids[se.get_slice()]
            s = self.spike_signal
            yield self.spike_signal.id_slice(ids_sliced)

    def iter_voltage_signals(self, sets):
        for se in sets:
            ids_sliced = self.ids[se.get_slice()]

            yield self.voltage_signal.id_slice(ids_sliced)

    def isrecorded(self, flag):

        if flag in ['spike', 's', 'spikes', 'spike_signal']:
            v = self.sd['active']

        elif flag in ['voltage', 'v', 'voltages', 'voltage_signal']:
            v = self.mm['active']
        else:
            v = True
        if not v:
            RuntimeError('{} traces are not recorded'.format(flag))
        return v

    def model_par_randomize(self):
        '''
        Example:
        self.randomization={ 'C_m':{'active':True, 
                                    'gaussian':{'sigma':0.2*C_m, 'my':C_m}}}
        '''
        pyrngs = self.set_random_states_nest()
        for p, val in self.rand.iteritems():
            if not val['active']:
                continue

            local_nodes = []
            #             st=my_nest.GetStatus(self.ids, ['local', 'gloabal_id', 'vp'])
            for _id in self.ids:
                ni = my_nest.GetStatus([_id])[0]
                if ni['local']:
                    local_nodes.append((ni['global_id'], ni['vp']))

            #             local_nodes=[(ni['global_id'], ni['vp'])
            #                          for ni in st if ni['local']]

            for gid, vp in local_nodes:
                val_rand = numpy.round(get_random_number(pyrngs, vp, val), 2)
                #                 print val_rand
                my_nest.SetStatus([gid], {p: val_rand})

    def voltage_response(self, currents, times, start, sim_time, id):

        scg = my_nest.Create('step_current_generator', n=1)
        my_nest.SetStatus(scg, {'amplitude_times': times,
                                'amplitude_values': currents})

        rec = my_nest.GetStatus([id])[0]['receptor_types']
        my_nest.Connect(scg, [id], params={'receptor_type': rec['CURR']})

        my_nest.MySimulate(sim_time)

        self.get_signal('v', 'V_m', start=start, stop=sim_time)
        self.get_signal('s')  # , start=start, stop=sim_time)
        self.signals['V_m'].my_set_spike_peak(15, spkSignal=self.signals['spikes'])
        voltage = self.signals['V_m'][id].signal
        times = numpy.arange(0, len(voltage) * self.mm['params']['interval'],
                             self.mm['params']['interval'])

        if len(times) != len(voltage):
            raise Exception('The vectors has to be the same length')

        return times, voltage

    def run_IF(self, I_vec, id=None, tStim=None):
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
        if not id: id = self.ids[0]
        if isinstance(id, int): id = [id]

        fIsi, mIsi, lIsi = [], [], []  # first, mean and last isi
        if not tStim: tStim = 500.0
        tAcum = 0

        I_e0 = my_nest.GetStatus(id)[0]['I_e']  # Retrieve neuron base current

        isi_list = []
        for I_e in I_vec:

            my_nest.SetStatus(id, params={'I_e': float(I_e + I_e0)})
            my_nest.SetStatus(id, params={'V_m': float(-61)})
            # my_nest.SetStatus( id, params = { 'w': float(0) } )

            rec = my_nest.GetStatus(id)[0]['recordables']

            for key in ['u', 'u1', 'u2']:
                if key not in rec:
                    continue
                my_nest.SetStatus(id, params={key: float(0)})

            simulate = True
            tStart = tAcum
            while simulate:
                my_nest.Simulate(tStim)
                tAcum += tStim

                #                 self.get_signal('s', start=tStart, stop=tAcum)

                kw = {'t_start': tStart, 't_stop': tAcum}
                signal = self.spike_signal.time_slice(**kw)

                if signal.mean_rate() > 0.1 or tAcum > 20000:
                    simulate = False

            isi = signal.isi()[0]
            if not any(isi): isi = [1000000.]

            fIsi.append(isi[0])  # retrieve first isi
            mIsi.append(numpy.mean(isi))  # retrieve mean isi
            if len(isi) > 100:
                lIsi.append(isi[-1])
            else:
                lIsi.append(isi[-1])  # retrieve last isi
            isi_list.append(numpy.array(isi))

        fIsi = numpy.array(fIsi)
        mIsi = numpy.array(mIsi)
        lIsi = numpy.array(lIsi)

        I_vec = numpy.array(I_vec)

        return I_vec, fIsi, mIsi, lIsi

    def run_IV_I_clamp(self, I_vec, id=None, tStim=2000):
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

        vSteadyState = []

        if not id: id = self.ids[0]
        if isinstance(id, int): id = [id]

        tAcum = 1  # accumulated simulation time, step_current_generator
        # recuires it to start at t>0

        scg = my_nest.Create('step_current_generator')
        rec = my_nest.GetStatus(id)[0]['receptor_types']
        my_nest.Connect(scg, my_nest.NodeCollection(id), syn_spec={'receptor_type': rec['CURR']})

        ampTimes = []
        ampValues = []
        for I_e in I_vec:
            ampTimes.extend([float(tAcum)])
            ampValues.extend([float(I_e)])
            tAcum += tStim

        my_nest.SetStatus(scg, params={'amplitude_times': ampTimes,
                                       'amplitude_values': ampValues})
        my_nest.Simulate(tAcum)

        #         self.get_signal( 'v','V_m', stop=tAcum ) # retrieve signal
        #         self.get_signal('s')
        if 0 < self.spike_signal.mean_rate():
            print('hej')
        tAcum = 1
        for I_e in I_vec:
            kw = {'t_start': tAcum + 10, 't_stop': tAcum + tStim}
            if 0 >= self.spike_signal.mean_rate(**kw):
                signal = self.voltage_signal.my_time_slice(tAcum + 10, tAcum + tStim)
                vSteadyState.append(signal[1].signal[-1])
            tAcum += tStim

        I_vec = I_vec[0:len(vSteadyState)]
        return numpy.array(I_vec), numpy.array(vSteadyState)


class MyInput():
    def __init__(self, **kwargs):

        self.ids = []
        self._local_ids = []
        self.model = 'poisson_generator'

    def set_spike_times(self, rates=[], times=[], t_stop=None, ids=None, seed=None, idx=None):
        t_starts = times
        t_stops = list(times[1:]) + list([t_stop])

        params = [{'rate': v[0], 'start': v[1], 'stop': v[2]} for v in zip(rates, t_starts, t_stops)]

        if len(params) == 1:
            ids = my_nest.Create('poisson_generator', len(params), params[0])
        else:
            ids = my_nest.Create('poisson_generator', len(params), params)

        self.ids = list(ids)
        self.local_ids = list(ids)  # Nedd to put on locals also


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

        model = kwargs.get('model', 'poisson_generator')

        df = my_nest.GetDefaults(model)
        if df['type_id'] in ['spike_generator', 'mip_generator',
                             'poisson_generator', 'poisson_generator_dynamic',
                             'my_poisson_generator']:
            input_model = model
            type_model = df['type_id']
            kwargs['model'] = 'parrot_neuron'

        super(MyPoissonInput, self).__init__(name, **kwargs)

        self._init_extra_attributes(input_model, type_model, **kwargs)

        if self.spike_setup != []:
            for k in self.spike_setup:
                self.set_spike_times(**k)

    def _init_extra_attributes(self, input_model, type_model, **kwargs):

        # Add new attribute
        self.spike_setup = kwargs.get('spike_setup', [])
        self.input_model = input_model
        self.type_model = type_model
        self.ids_generator = {}

    def set_spike_times(self, rates=[], times=[], t_stop=None, ids=None,
                        seed=None, idx=None):

        df = my_nest.GetDefaults(self.input_model)['model']

        if ids is None and (not idx is None):
            tmp_ids = numpy.array(self.ids)
            ids = list(tmp_ids[idx])
        if ids is None:
            ids = self.ids

            # Spike generator
        if 'spike_generator' == self.type_model:
            for id in ids:
                seed = random_integers(0, 10 ** 5)

                spikeTimes = misc.inh_poisson_spikes(rates, times, t_stop=t_stop, n_rep=1, seed=seed)
                if any(spikeTimes):
                    my_nest.SetStatus([id], params={'spike_times': spikeTimes})


        # MIP
        elif 'mip_generator' == self.type_model:
            c = df['p_copy']

            seed = random_integers(0, 10 ** 6)
            new_ids = []
            t_starts = times
            t_stops = times[1:] + [t_stop]
            for id in ids:
                i = 0
                for r, start, stop in rates, t_starts, t_stops:
                    r_mother = r / c
                    params = {'rate': r_mother, 'start': start, 'stop': stop,
                              'p_copy': c, 'mother_seed': seed}
                    if i == 0:
                        my_nest.SetStatus(id, params)
                    else:
                        new_id = my_nest.Create('mip_generator', 1, params)

                    new_ids.append(new_id)
            self.ids.append(new_ids)

            # Poisson generator
        elif self.type_model in ['my_poisson_generator',
                                 'poisson_generator']:

            t_starts = times
            t_stops = list(times[1:]) + list([t_stop])

            params = [{'rate': v[0], 'start': v[1], 'stop': v[2]}
                      for v in zip(rates, t_starts, t_stops)]

            if len(params) == 1:
                source_nodes = my_nest.Create('poisson_generator',
                                              len(params),
                                              params[0]) * len(ids)
            else:
                source_nodes = my_nest.Create('poisson_generator',
                                              len(params), params) * len(ids)

            target_nodes = numpy.array([[id_] * len(rates) for id_ in ids])
            target_nodes = list(numpy.reshape(target_nodes,
                                              len(rates) * len(ids), order='C'))
            #             pp(my_nest.GetStatus([2]))
            my_nest.Connect(source_nodes, target_nodes)

            generators = []
            if hash(tuple(ids)) in self.ids_generator.keys():
                generators = self.ids_generator[hash(tuple(idx))]

            generators = list(set(source_nodes).union(generators))
            self.ids_generator[hash(tuple(idx))] = sorted(generators)
            self.local_ids = list(self.ids)  # Nedd to put on locals also

        elif 'poisson_generator_dynamic' == self.type_model:
            source_nodes = my_nest.Create(self.type_model, 1,
                                          {'timings': times,
                                           'rates': rates}) * len(ids)
            target_nodes = ids
            my_nest.Connect(source_nodes, target_nodes)

            generators = []
            if hash(tuple(ids)) in self.ids_generator.keys():
                generators = self.ids_generator[hash(tuple(idx))]

            generators = list(set(source_nodes).union(generators))
            self.ids_generator[hash(tuple(idx))] = sorted(generators)

        #             v=my_nest.GetStatus(ids, 'local')
        #             self.local_ids=[_id for _id in zip(ids,v) if  # Nedd to put on locals also

        else:
            msg = 'type_model ' + self.type_model + ' is not accounted for in set_spike_times'
            raise ValueError(msg)

    def update_spike_times(self, rates=[], times=[], t_stop=None, ids=None, seed=None, idx=None):
        if 'poisson_generator' == self.type_model:
            t_starts = times
            t_stops = list(times[1:]) + list([t_stop])

            params = [{'rate': v[0], 'start': v[1], 'stop': v[2]}
                      for v in zip(rates, t_starts, t_stops)]
            my_nest.SetStatus(self.ids_generator[hash(tuple(idx))], params)


def default_kwargs_net(n, n_sets):
    d = {'tau_w': 20.,  # I-V relation, spike frequency adaptation
         'a_1': 3.,  # I-V relation
         'a_2': 3.,  # I-V relation
         'b': 200.,  # I-F relation
         'C_m': 80.,  # t_m/R_in
         'Delta_T': 1.8,
         'g_L': 3.,
         'E_L': -55.8,  #
         'I_e': 15.0,
         'V_peak': 20.,  #
         'V_reset': -65.,  # I-V relation
         'V_th': -55.2,  #
         'V_a': -55.8,  # I-V relation

         # STN-SNr
         'AMPA_1_Tau_decay': 12.,  # n.d.; set as for STN to GPE
         'AMPA_1_E_rev': 0.,  # n.d. same as CTX to STN

         # EXT-SNr
         'AMPA_2_Tau_decay': 5.0,
         'AMPA_2_E_rev': 0.,

         # MSN D1-SNr
         'GABAA_1_E_rev': -80.,  # (Connelly et al. 2010)
         'GABAA_1_Tau_decay': 12.,  # (Connelly et al. 2010)

         # GPe-SNr
         'GABAA_2_E_rev': -72.,  # (Connelly et al. 2010)
         'GABAA_2_Tau_decay': 5.,
         }
    path, sli_path = my_nest.get_default_module_paths(dr.HOME_MODULE)
    my_nest.install_module(path, sli_path, model_to_exist='izhik_cond_exp')

    sets = [misc.my_slice(i, n, n_sets) for i in range(n_sets)]
    return {'n': n,
            'model': 'my_aeif_cond_exp',
            'mm': {'active': True,
                   'params': {'interval': 1.0,
                              'record_to': 'memory',
                              # 'to_memory': True,
                              # 'to_file': False,
                              'record_from': ['V_m', 'g_AMPA_1_', 'g_GABAA_1']}},
            'params': d,
            'sd': {'active': True,
                   'params': {'record_to': 'memory',
                              #  'to_memory': True,
                              # 'to_file': False
                              }},
            'sets': sets,
            'rate': 10.0}


def default_kwargs_inp(n):
    sets = [misc.my_slice(i, n, 1) for i in range(1)]
    return {'n': n,
            'model': 'poisson_generator',
            'sets': sets,
            'rate': 10.0}


def default_spike_setup(n, stop):
    d = {'rates': [10.0],
         'times': [1.0],
         't_stop': stop,
         'idx': range(n)}
    return d


def get_nullcline_aeif(**kw):
    a_1 = kw.get('a_1')
    a_2 = kw.get('a_2')
    Delta_T = kw.get('Delta_T')
    g_L = kw.get('g_L')
    E_L = kw.get('E_L')
    V0 = numpy.array(kw.get('V'))
    V_a = kw.get('V_a')
    V_th = kw.get('V_th')

    V = V0[V0 < V_a]
    l0 = g_L * (V - E_L) - g_L * Delta_T * numpy.exp((V - V_th) / Delta_T) + a_1 * (V - V_a)

    V = V0[V0 >= V_a]
    l1 = g_L * (V - E_L) - g_L * Delta_T * numpy.exp((V - V_th) / Delta_T) + a_2 * (V - V_a)
    return V0, -numpy.array(list(l0) + list(l1))


def get_random_number(pyrngs, vp, val):
    if 'gaussian' in val.keys():
        par = val['gaussian']
        if par['my'] == 0:
            return 0.0

        else:
            r = pyrngs[vp].normal(loc=par['my'], scale=par['sigma'])

        #         print par.keys()
        if 'cut' in par.keys():

            if r < par['my'] - par['cut_at'] * par['sigma']:
                r = par['my'] - par['cut_at'] * par['sigma']
            elif r > par['my'] + par['cut_at'] * par['sigma']:
                r = par['my'] + par['cut_at'] * par['sigma']

        return r

    if 'uniform' in val.keys():
        par = val['uniform']
        return pyrngs[vp].uniform(par['min'], par['max'])


def sim_group(sim_time, *args, **kwargs):
    g = MyNetworkNode(*args, **kwargs)

    df = my_nest.GetDefaults('my_aeif_cond_exp')['receptor_types']

    inp_ex = my_nest.Create('poisson_generator', params={'rate': 30.})
    inp_ih = my_nest.Create('poisson_generator', params={'rate': 30.})

    for pre in inp_ex:
        for post in inp_ih:
            my_nest.Connect(pre, post, {'receptor_type': df['g_AMPA_1']})
    for pre in inp_ex:
        for post in inp_ih:
            my_nest.Connect(pre, post, {'receptor_type': df['g_GABAA_1']})

    my_nest.Simulate(sim_time)
    return g


from os.path import expanduser
import subprocess


class TestModule_functions(unittest.TestCase):

    def setUp(self):
        self.home = expanduser("~")

    def test_collect_spikes_mpi(self):
        data_path = self.home + ('/results/unittest/my_population'
                                 + '/collect_spikes_mpi/')
        script_name = os.getcwd() + ('/test_scripts_MPI/'
                                     + 'my_population_collect_spikes_mpi.py')

        np = 4
        s0 = []
        for i in range(4):
            s0 += [float(i), float(i)]

        s0 = numpy.array(s0)
        e0 = numpy.array(s0) + 1

        p = subprocess.Popen(['mpirun', '-np', str(np), 'python',
                              script_name, data_path],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)

        out, err = p.communicate()
        #         print out
        #         print err

        import pickle

        f = open(data_path + 'data.pkl', 'rb')  # open in binary mode

        s1, e1 = pickle.load(f)
        s1, e1 = sorted(s1), sorted(e1)

        self.assertListEqual(list(s0), list(s1))
        self.assertListEqual(list(e0), list(e1))

        f.close()


from python.core.data_to_disk import pickle_save, pickle_load


class TestMyNetworkNode(unittest.TestCase):
    my_nest.sli_run("M_WARNING setverbosity")  # silence nest output

    # print my_nest.GetKernelStatus()

    def setUp(self):
        self.home = expanduser("~")
        self.n = 12
        self.n_sets = 3
        self.args = ['unittest']
        self.kwargs = default_kwargs_net(self.n, self.n_sets)
        self.sim_time = 10000.
        dp = self.home + '/results/unittest/my_population/nest/'
        data_to_disk.mkdir(dp)
        my_nest.ResetKernel(display=False, data_path=dp)
        my_nest.SetKernelStatus({'overwrite_files': True})

    def sim_group(self, **kwargs):
        kwargs = misc.dict_update(self.kwargs, kwargs)
        return sim_group(self.sim_time, *self.args, **self.kwargs)

    def test_1_create(self):
        g = MyNetworkNode(*self.args, **self.kwargs)

    def test_21_get_spike_signal_from_memory(self):
        l = self.sim_group().get_spike_signal()
        self.assertEqual(l.shape[1], self.n_sets)

        mr = 0
        for _, _, spk_list in my_signals.iter2d(l):
            mr += spk_list.mean_rate() / l.shape[1]
        self.assertAlmostEqual(mr, 55.0, delta=0.1)

    def test_22_get_spike_signal_from_file(self):
        d = {'sd': {'active': True,
                    'params': {
                        'record_to': 'file',
                        # 'to_memory': False,
                        # 'to_file': True
                    }}}
        l = self.sim_group(**d).get_spike_signal()
        self.assertEqual(l.shape[1], self.n_sets)

        mr = 0
        for _, _, spk_list in my_signals.iter2d(l):
            mr += spk_list.mean_rate() / l.shape[1]
        self.assertAlmostEqual(mr, 55.0, delta=0.1)

    def test_3_get_voltage_signal(self):
        l = self.sim_group().get_voltage_signal()
        self.assertEqual(l.shape[1], self.n_sets)

    def test_3_get_conductance_signals(self):
        l = self.sim_group().get_conductance_signals()()
        self.assertEqual(l.shape[1], self.n_sets)

    def test_4_multiple_threads(self):
        my_nest.SetKernelStatus({'local_num_threads': 2})
        _ = self.sim_group().get_spike_signal()

    def test_5_load_from_disk(self):
        from os.path import expanduser
        s = expanduser("~")
        s = s + '/results/unittest/my_population'
        data_to_disk.mkdir(s)
        my_nest.SetKernelStatus({'local_num_threads': 2,
                                 'data_path': s,
                                 'overwrite_files': True, })

        self.kwargs['sd']['params'].update({
            'record_to': 'file',
            # 'to_memory': False,
            # 'to_file': True
        })

        g = self.sim_group().get_spike_signal()
        g[0].firing_rate(1, display=True)
        #         pylab.show()
        for filename in os.listdir(s):
            if filename.endswith(".gdf"):
                os.remove(s + '/' + filename)

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
        data_path = self.home + '/results/unittest/my_population/mpi_run/'
        script_name = os.getcwd() + '/test_scripts_MPI/my_population_mpi_run.py'
        np = 4

        fileOut, mr0, fr0 = self.do_mpi(data_path, script_name, np)

        mr1, fr1 = pickle_load(fileOut)
        self.assertEqual(mr0, mr1)
        self.assertListEqual(list(fr0), list(fr1))

    def test_62_mpi_run_from_disk(self):
        data_path = self.home + '/results/unittest/my_population/mpi_run_from_disk/'
        script_name = os.getcwd() + '/test_scripts_MPI/my_population_mpi_run_from_disk.py'
        np = 4

        for filename in os.listdir(data_path):
            path = data_path + '/' + filename
            if os.path.isfile(path):
                os.remove(path)

        fileOut, mr0, fr0 = self.do_mpi(data_path, script_name, np)

        ss = pickle_load(fileOut)
        mr1 = ss.mean_rate()
        fr1 = ss.firing_rate(1)

        self.assertEqual(mr0, mr1)
        self.assertListEqual(list(fr0), list(fr1))

    def test_63_mpi_run_hyprid_from_disk(self):
        data_path = self.home + '/results/unittest/my_population/mpi_run_hybrid_from_disk/'
        script_name = os.getcwd() + '/test_scripts_MPI/my_population_mpi_run_hybrid_from_disk.py'
        np = 4

        #         os.environ['OMP_NUM_THREADS'] = '2'
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

        g = pickle_load(fileOut)
        ss = g.get_spike_signal()
        mr1 = ss.mean_rate()
        fr1 = ss.firing_rate(1)

        self.assertEqual(mr0, mr1)
        self.assertListEqual(list(fr0), list(fr1))


class TestMyPoissonInput(unittest.TestCase):
    def setUp(self):
        self.n = 1
        self.n_inp = 10
        self.n_sets = 1
        self.args_net = ['unittest_net']
        self.args_inp = ['unittest_inp']
        self.kwargs_net = default_kwargs_net(self.n, self.n_sets)
        self.kwargs_net['params'].update({'I_e': 200.0})
        self.kwargs_inp = default_kwargs_inp(self.n_inp)
        self.sim_time = 1000.
        my_nest.ResetKernel(display=False)

    @property
    def sim_group(self):
        g = MyNetworkNode(*self.args_net, **self.kwargs_net)
        i = MyPoissonInput(*self.args_inp, **self.kwargs_inp)
        i.set_spike_times(**default_spike_setup(self.n_inp, self.sim_time))
        my_nest.Connect(i.ids,
                        g.ids * len(i.ids),
                        params={'delay': 1.0,
                                'weight': 10.0},
                        model='static_synapse')
        my_nest.Simulate(self.sim_time)
        return g

    def test_1_create(self):
        _ = MyPoissonInput(*self.args_inp, **self.kwargs_inp)

    def test_2_simulate_show(self):
        g = self.sim_group.get_voltage_signal()[0].plot(display=True)
    #         pylab.show()


if __name__ == '__main__':
    d = {
        TestModule_functions: [
            #                               'test_collect_spikes_mpi',
        ],
        TestMyNetworkNode: [
            #                            'test_1_create',
            #                             'test_21_get_spike_signal_from_memory',
            #                             'test_22_get_spike_signal_from_file',
            'test_3_get_voltage_signal',
            #                            'test_4_multiple_threads',
            #                            'test_5_load_from_disk',
            #                             'test_61_mpi_run',
            #                             'test_62_mpi_run_from_disk',
            #                             'test_63_mpi_run_hyprid_from_disk',
        ],
        TestMyPoissonInput: [
            #                            'test_create',
            #                            'test_2_simulate_show',
        ],

    }
    test_classes_to_run = d
    suite = unittest.TestSuite()
    for test_class, val in test_classes_to_run.items():
        for test in val:
            suite.addTest(test_class(test))

    unittest.TextTestRunner(verbosity=2).run(suite)
