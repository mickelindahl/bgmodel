'''
Created on Jun 18, 2013

@author: lindahlm
'''
from toolbox.network_connectivity import Units_input, Units_neuron, Structure_list
from toolbox import my_nest, data_to_disk, misc 
from toolbox.my_population import MyGroup, MyPoissonInput, MyInput

from copy import deepcopy
from toolbox.default_params import Par, Par_slow_wave, Par_bcpnn_h0, Par_bcpnn_h1
import nest # Can not be first then I get segmentation Fault
import numpy
import pylab
import time
import unittest
import os, sys

class Inhibition_base(object):
    '''
    Base model
    '''
    
    def __init__(self, par_rep={}, perturbation=None,  **kwargs):
        '''
        Constructor
        '''
        self.calibrated=False
        self.inputed=False
        self.built=False
        self.connected=False
        
        self.class_default_params=Par
        
        self.name=self.__class__.__name__ 
        self.input_class=MyPoissonInput
        self.input_params={} #set in inputs       
              
        self.stdout=None
        self.par_rep=par_rep
        self.perturbation=perturbation
        self._par=None
        
           
        self.save_conn= kwargs.get('save_conn', True)
        self.structures=Structure_list()
        self.sub_folder=kwargs.get('sub_folder', '')
    
        self.time_calibrated=None
        self.time_inputed=None
        self.time_built=None
        self.time_connected=None
        self.time_run=None
        
        self.units_list=[]
        self.units_dic={}
        
        self.verbose=kwargs.get('verbose', 'True')
        
    @property
    def par(self):
        if self._par==None:
            self._par=self.class_default_params( self.par_rep, self.perturbation)  
        return self._par
    
    @property
    def path_data(self):
        if self.sub_folder:
            sub_folder=self.sub_folder+'/'
        else:
            sub_folder=''
        return '/afs/nada.kth.se/home/w/u1yxbcfw/results/papers/inhibition'+'/'+self.name +'/'+sub_folder
    
    @property
    def path_pictures(self):
        if self.sub_folder:
            sub_folder=self.sub_folder+'-'
        else:
            sub_folder=''
        return '/afs/nada.kth.se/home/w/u1yxbcfw/projects/papers/inhibition/pictures'+'/'+self.name +'-'+sub_folder
    
    @property
    def path_nest(self):
        return self.path_data+'nest'
    
    @property
    def threads(self):
        return self.par['simu']['threads']  
        
    @property
    def sim_time(self):      
        return self.par['simu']['sim_time']    
       
    @property
    def start_rec(self):
        return self.par['simu']['start_rec']       
    
    @property
    def stop_rec(self):
        return self.par['simu']['stop_rec']       

    def stop_stdout(self, switch):
        if switch:
            f = open(os.devnull, 'w')
            self.stdout=sys.stdout
            sys.stdout = f
        else:
            sys.stdout.close()
            sys.stdout=self.stdout
    
    
    def save(self, save_at):
        ''' 
        Have to keep in mind to put pointer to unit objects in 
        structures when loading object
        '''
        
        data_to_disk.pickle_save(self, save_at)
    def load(self, save_at):
        
        obj=data_to_disk.pickle_load(save_at)
        
        for attr, value in obj.__dict__.iteritems():
            self.__dict__[attr]=value
        
        self.units_dic, self.units_list=self.structures.recover_units()
                  
    def calibrate(self):
        '''
        Compute all dependent variables.
        '''
        if not self.verbose: self.stop_stdout(True)
        t=time.time()
        print '\nCalibrating...'
        self.calibrated=True
        self.time_calibrated=int(time.time()-t)
        print 'Calibrated', self.time_calibrated  
        
        if not self.verbose: self.stop_stdout(False)
        
    def inputs(self):
        
        if self.inputed:
            return
        if not self.calibrated: self.calibrate()
        if not self.verbose: self.stop_stdout(True)
        t=time.time()
        print 'Creating inputs...'
        self.inputed=True
        

        for key, val in self.par['netw']['input'].items():
            for model in val['nodes']:
                if key=='constant':           
                
                    self.input_params[model]=[{'rates':[ self.par['node'][model]['rate']], 
                                               'times':[1.],
                                               'idx':range(self.par['node'][model]['n'])}]    

                if key=='oscillation':
                    ru=self.par['node'][model]['rate']*(2-val['p_amplitude_mod']) 
                    rd=self.par['node'][model]['rate']*val['p_amplitude_mod']
         
                    step=1000/2/val['freq']
                    cycles=int(self.par['simu']['sim_time']/(2*step)+1)
                    rates=[rd, ru]*cycles
                    times=numpy.arange(0, 2.*cycles*step, step)
                    self.input_params[model]=[{'rates':rates, 
                                               'times':times,
                                               'idx':range(self.par['node'][model]['n'])}]
                
                if key=='bcpnn': 
                    tt=val['time']
                    p=val['p_amplitude']
                    for i in range(val['n_set_pre']):
                        if i==0: self.input_params[model]=[]
                            
                        idx1=list(range(i, self.par['node'][model]['n'] , val['n_set_pre']))

                        self.input_params[model]+=[{'rates':[ self.par['node'][model]['rate'],
                                                              self.par['node'][model]['rate']*p,
                                                              self.par['node'][model]['rate']], 
                                                   'times':[1., self.start_rec+i*tt, self.start_rec+(i+1)*tt],
                                                   'idx':idx1}] 

        self.time_inputed=int(time.time()-t)
        print 'Inputed', self.time_inputed  
        if not self.verbose: self.stop_stdout(False)       
    def build(self):
        '''
        Creates units representing populations and their spread and stuctures
        holding the connections between populations. This is saved to disk and
        loaded next simulations if not deleted by hand.
        
        Then create all nodes, used in the model
        '''
        
        
        
        if self.built: return
        if not self.inputed: self.inputs() 
        if not self.verbose: self.stop_stdout(True)
        
        print 'Building...'
        t=time.time()

        my_nest.ResetKernel(threads=self.threads, print_time=False)  
        #my_nest.SetKernelStatus({'resolution':0.05})
        
        # Create input units
        for k,v in self.par['node'].iteritems(): 
            if not v['lesion']:     
                dic={}
                self.units_list.append(v['unit_class'](k, dic, self.par))
                self.units_dic[k]=self.units_list[-1]
                
                print self.units_dic[k].n, k
                assert self.units_dic[k].n>=1.0, "Unit %s needs to have atleast one node"%(k)
                # Set input units
        

        setup_structure_list=[]
        for k, v in self.par['conn'].iteritems(): 
   
            if not v['lesion']:
   
                s=k.split('_')
                keys=self.units_dic.keys()
                if (s[0] in keys) and (s[1] in keys):
                    # Add units to dictionary
                    dic={}
                    dic['source']=self.units_dic[s[0]]
                    dic['target']=self.units_dic[s[1]]
                    dic['save_at']='/'.join(self.path_data.split('/')[0:-2])+'/'
                    setup_structure_list.append([k, dic, self.par])

        self.structures=Structure_list(setup_structure_list)
    
        for s in sorted(self.structures,key=lambda x:x.name):
            s.set_connections(save_mode=self.save_conn)

        #! Create input nodes
        for u in self.units_list:
            # Load model
            my_nest.MyLoadModels( self.par.get_nest_setup(u.model), [u.model] )
            
            if u.type == 'input':
                inp=MyPoissonInput( u.model, u.n )     
                
                for p in self.input_params[u.name]:  
                    if not len(p['idx']):
                        p['idx']=range(u.n)
                    inp.set_spike_times(p['rates'], p['times'], self.sim_time, idx=p['idx'])
                #inp.add_spike_recorder() # this eats memory!
            
                self.units_dic[u.name].set_population(inp) 
                                    
            elif u.type == 'network':          
                sd_params={'start':self.start_rec,  'stop':self.stop_rec}
                sd_params.update(self.par['simu']['sd_params'])
                
                group=MyGroup(model = u.model, n=u.n, params = {'I_e':u.I_vivo}, sd=True, 
                              sd_params=sd_params)          
            
                self.units_dic[u.name].set_population(group)

        
        self.built=True
        self.time_built=int(time.time()-t)
        print 'Built', self.time_built
        
        if not self.verbose: self.stop_stdout(False)
        
    def randomize_params(self, params):
        if not self.built: self.build()
        
        for u in self.units_dic.values():
            if u.type =='network':
                u.randomize_params(params)        

    def connect(self):
        '''Connect all nodes in the model'''
        if self.connected: return
        if not self.calibrated: self.calibrate()
        if not self.built: self.build()
        if not self.verbose: self.stop_stdout(True)
        t=time.time()

        for s in sorted(self.structures, key=lambda x:x.name):
            # Load model

            print 'Connecting '+str(s)
            
            my_nest.MyLoadModels( self.par.get_nest_setup(s.syn), [s.syn] )
                        
            sr_ids=numpy.array(s.source.population.ids)
            tr_ids=numpy.array(s.target.population.ids)

            weights=list(s.get_weights())
            delays=list(s.get_delays())
            pre=list(sr_ids[s.conn_pre])
            post=list(tr_ids[s.conn_post])

            my_nest.Connect(pre, post , weights, delays, model=s.syn)
      
            #Clear connection in structure to lower mem consumption. Probably non necessary
            #s.conn_pre=None
            #s.conn_post=None
            
        self.connected=True
        self.time_connected=int(time.time()-t)
        print 'Connected', self.time_connected
        if not self.verbose: self.stop_stdout(False)
    def run(self, print_time=False):
        
        if not self.connected: self.connect()
        if not self.verbose: self.stop_stdout(True)

        if not os.path.isdir(self.path_nest):
            data_to_disk.mkdir(self.path_nest)
        
        my_nest.SetKernelStatus({'print_time':self.par['simu']['print_time'], 'data_path':self.path_nest, 'overwrite_files': True})
        
        for filename in os.listdir(self.path_nest):
            if filename.endswith(".gdf"):
                print 'Deleting: ' +self.path_nest+'/'+filename
                os.remove(self.path_nest+'/'+filename)
        
        t=time.time()
        
        #my_nest.Simulate(self.sim_time)
        print my_nest.GetKernelStatus()
        print self.par
        print self.structures
        my_nest.Simulate(self.sim_time)       
        self.time_run=int(time.time()-t)
        print '{0:10} {1}'.format('Simulated', self.time_run)
        
        if not self.verbose: self.stop_stdout(False)
        
    def get_ids(self, models):
        
        ids_dic={}
        for model in models:
            self.get_spikes(model)
            pop=self.units_dic[model].population            
            ids_dic[model]=pop.ids
        return ids_dic
    
    def get_isis(self, models):
        if isinstance(models,str):
            models=[models]
        dic_isis={}
        for model in models:
            self.get_spikes(model)
            pop=self.units_dic[model].population
            isis=pop.signals['spikes'].isi()
            dic_isis[model]=[pop.signals['spikes'].id_list, isis]
            
        return dic_isis    

    
        
    def get_firing_rate(self, models):
        if isinstance(models,str):
            models=[models]
        dic_rates={}
        for model in models:
            self.get_spikes(model)
            pop=self.units_dic[model].population
            fr=pop.signals['spikes'].firing_rate(1, average=True)
            dic_rates[model]=[numpy.arange(self.start_rec, len(fr)+self.start_rec, 1 ), fr]
            
        return dic_rates

    def get_firing_rate_sets(self, models):
        if isinstance(models,str):
            models=[models]
        dic_rates={}
        
        
        for model in models:
            u=self.units_dic[model]
            self.get_spikes(model)
            
            pop=self.units_dic[model].population
            ids=numpy.array(self.units_dic[model].population.ids)
            fr=[]
            for se in u.sets: 
                
                fr.append(pop.signals['spikes'].id_slice(ids[se]).firing_rate(1, average=True))
            fr=numpy.array(fr)
            dic_rates[model]=[numpy.arange(self.start_rec, fr.shape[1]+self.start_rec, 1 ),fr ]
            
        return dic_rates

    def get_mean_rates(self, models):
        if isinstance(models,str):
            models=[models]
        dic_rates={}
        for model in models:
            
            self.get_spikes(model)
            pop=self.units_dic[model].population
            mrs=pop.signals['spikes'].mean_rates()

            dic_rates[model]=[pop.ids, mrs]
            
        return dic_rates
    
    
    def get_rasters(self, models):
        
        dic_rasters={}
        for model in models:
            self.get_spikes(model)
            pop=self.units_dic[model].population    
            spk_times, spk_ids, ids= numpy.array(pop.signals['spikes'].my_raster())        
            dic_rasters[model]=[numpy.array([spk_times, spk_ids]), ids]
        return dic_rasters
        
    def get_rasters_sets(self, models):
        
        dic_rasters={}
        for model in models:
            u=self.units_dic[model]
            self.get_spikes(model)
            pop=self.units_dic[model].population
            ids0=numpy.array(self.units_dic[model].population.ids)
            n_ids_cum=0
            d=[]
            for se in u.sets: 
                spk_times, spk_ids, ids=numpy.array(pop.signals['spikes'].id_slice(ids0[se]).my_raster()) 
                n_ids=len(ids)
                d.append([numpy.array([spk_times, spk_ids+n_ids_cum]), ids+n_ids_cum])
                n_ids_cum+=n_ids
                      
            dic_rasters[model]=d
        return dic_rasters


    
    def get_simtime_data(self):
        t_total=self.time_calibrated+self.time_built+self.time_connected+self.time_run
        s='{0:5} total:{1} (sec) cali/built/conn/run (%) {2}/{3}/{4}/{5} '.format('Time', t_total, 
                                                                int(100*self.time_calibrated/t_total),
                                                                int(100*self.time_built/t_total),
                                                                int(100*self.time_connected/t_total),
                                                                int(100*self.time_run/t_total))
        return s
      
    def get_spikes_binned(self, models, res, clip=0):
        rdb={} 

        for model in models:
            self.get_spikes(model)
            pop=self.units_dic[model].population
            times, rdb[model]=pop.signals['spikes'].raw_data_binned(self.start_rec, self.sim_time, res, clip)
            
        return rdb
    
    def get_spikes(self, model):
        u=self.units_dic[model]
        pop=u.population
        
        if not u.collected_spikes:
            pop.get_signal( 's', recordable='spikes', start=self.start_rec, stop=self.sim_time )
        u.collected_spikes=True
        
    def get_voltage_trace(self, model):
        u=self.units_dic[model]
        pop=u.population
        
        if not u.collected_spikes:
            self.get_spikes(self, model)
        
        if not u.collected_votage_traces:
            pop.get_signal( 'v', recordable='V_m', start=self.start_rec, stop=self.sim_time )    
            pop.signals['V_m'].my_set_spike_peak( 15, spkSignal= pop.signals['spikes'], start=self.start_rec ) 

        u.collected_votage_traces=True
        
        voltages=[]
        times=[]
        for analog_signal in pop.signals['V_m']:
            voltages.append(analog_signal.signal)
            times.append(numpy.arange(0, len(voltages[-1]))*pop.mm_params['interval'])
        
        if len(voltages)==1:
            voltages=voltages[0]
            times=times[0]
            
        voltages=numpy.array(voltages)
        times=numpy.array(times)
        
        return times, voltages       
    
    def get_mm(self, model, mm_types):
        pop=self.units_dic[self.study_name].population
        recs=my_nest.GetStatus(pop.ids, ['recordables'])[0][0]
        recs=[rec for rec in recs if rec[0] in mm_types]
        
        for rec in recs:
            if 'I' in mm_types:
                pop.get_signal( 'c', recordable=rec, start=self.start_rec, stop=self.sim_time )    
            if 'g' in mm_types:
                pop.get_signal( 'g', recordable=rec, start=self.start_rec, stop=self.sim_time )    
        signal={}
        times={}
        for rec in recs:
            for analog_signal in pop.signals[rec]:
                if not rec in signal.keys():
                    signal[rec]=[]
                    times[rec]=[]
                signal[rec].append(analog_signal.signal)
                times[rec].append(numpy.arange(0, len(signal[rec][-1]))*pop.mm_params['interval'])    
                
            if len(signal[rec])==1:
                signal[rec]=signal[rec][0]
                times[rec]=times[rec][0]
        return times, signal         

class Inhibition_no_parrot(Inhibition_base):    
    
    def __init__(self, threads=1, start_rec=1, sim_time=1000, **kwargs):
        super( Inhibition_no_parrot, self ).__init__(threads, start_rec, sim_time, **kwargs)       
        # In order to be able to convert super class object to subclass object        
        
        self.input_class=MyInput
              
class Slow_wave(Inhibition_base):  
    def __init__(self, par_rep={}, perturbation=None, **kwargs):
        super( Slow_wave, self ).__init__(par_rep, perturbation, **kwargs)       
        # In order to be able to convert super class object to subclass object   
        self.class_default_params=Par_slow_wave

   
class Single_units_activity(Inhibition_base):    
    
    def __init__(self, threads=1, start_rec=1, sim_time=1000, **kwargs):
        super( Single_units_activity, self ).__init__(threads, start_rec, sim_time, **kwargs)       
        # In order to be able to convert super class object to subclass object        
        
        if 'study_name' in kwargs.keys(): 
            self.study_name=kwargs['study_name']
            
        if 'included_models' in kwargs.keys(): 
            self.included_models=kwargs['included_models']
            
    def calibrate(self):           
        '''
        Compute all dependent variables, it is done by creating unit
        and stucuture object using model and network params from 
        model_params.py
        '''
        if self.calibrated: return
        
        self.par['netw']['size']=100.0 #To get correct mask for MSN and FSN this need to be small enough 
          
        name_dic={'M1':'M1p', 'M2':'M2p', 'FS':'FSp', 'ST':'STp',
                  'GA':'GAp', 'GI':'GIp', 'SN':'SNp',  
                  'C1':'C1p', 'C2':'C2p', 'CF':'CFp', 'CS':'CSp',
                  'EA':'EAp', 'EI':'EIp', 'ES':'ESp'}
           
        # Create poisson inuts       
        for old, new in name_dic.iteritems():         
            v=deepcopy(self.par['node'][old])
            v['model']  = 'poisson_generator'
        
            s=old+'_'+self.study_name
            for key, val in self.par['conn'].iteritems() :   
                if key[:len(s)]==s:
                    v['n']=val['fan_in']
            
                    if 'target_rate' in v.keys(): v['rate']   = v['target_rate'] 
                    if 'rate' in v.keys(): v['rate']   = v['rate'] 
                    v['extent'] = [-0.5, 0.5]
                    v['type']        = 'input'
                    v['unit_class']  = Units_input
                    self.par['node'][new]=v
                   
        
        #Create connections
        for old, new in name_dic.iteritems():     
            s=old+'_'+self.study_name
            for key, val in deepcopy(self.par['conn']).iteritems():
                if key[:len(s)]==s:
                    
                    v=deepcopy(self.par['conn'][key])
                    
                    # Add units to dictionary
                    v['rule']    = 'all'
                    v['delay_setup']  = {'constant':v['delay_val']}                                 
                    v['weight_setup'] = {'constant':v['weight_val']}
                    
                    self.par['conn'][new+'_'+self.study_name+ key[len(s):]]=v

        im=deepcopy(self.included_models)
        #Remove unused nodes and connections
        for key in self.par['node'].keys():
            if not key in im: 
                self.par['node'][key]['lesion']=True

         
        #Remove unused nodes and connections
        im.remove(self.study_name)
        for key in self.par['conn'].keys():
            
            if not key.split('_')[0]in im and not key.split('_')[1]!=self.study_name :
                self.par['conn'][key]['lesion']=True

                 
        
        self.calibrated=True     
    
    def inputs(self):
        if not self.calibrated: self.calibrate()
        
        
        for key, val in self.par['node'].iteritems():
            if val['type'] =='input':
                print key, val['rate']
                self.input_params[key]=[{'rates':[val['rate']], 
                                         'times':[1.],
                                         'idx':[]}]    
                
    def record_voltage(self, p):
        if not self.built: self.build()
        if not self.calibrated: self.calibrate()
        
        if p:
            pop=self.units_dic[self.study_name].population
            mm_params={'interval':0.1, 'start':self.start_rec,  'stop':self.sim_time, 
                   'record_from':['V_m']} 
            pop.set_mm(True, mm_params)
    
    def record_mm(self, p):
        if not self.built: self.build()
        if not self.calibrated: self.calibrate()
         
        if p:
            pop=self.units_dic[self.study_name].population
            recs=my_nest.GetStatus(pop.ids, ['recordables'])[0][0]
            recs=[rec for rec in recs if rec[0] in p]
            
            mm_params={'interval':0.1, 'start':self.start_rec,  'stop':self.sim_time, 
                   'record_from':recs} 
            pop.set_mm(True, mm_params)
           
    def get_mean_rate(self, model):
        pop=self.units_dic[model].population
        if not pop.signaled['spikes']:
            self.get_spikes(model)
        
        return pop.signals['spikes'].time_slice(self.start_rec,self.sim_time).mean_rate()   
    
    def get_mean_rates(self, model):
        pop=self.units_dic[model].population
        if not pop.signaled['spikes']:
            self.get_spikes(model)
        
        return pop.signals['spikes'].time_slice(self.start_rec,self.sim_time).mean_rates()


        
    def get_spike_statistics(self, model):

        pop=self.units_dic[model].population
        
        if not pop.signaled['spikes']:
            self.get_spikes(model)
        
        spk=pop.signals['spikes'].time_slice(self.start_rec,self.sim_time).raw_data()[:,0]

        mean_rates=len(spk)/len(pop.ids)/(self.sim_time-self.start_rec)*1000.0
        
        mean_isi=numpy.mean(numpy.diff(spk,axis=0))
        std_isi=numpy.std(numpy.diff(spk,axis=0))
        CV_isi=std_isi/mean_isi

        return mean_rates, mean_isi, std_isi, CV_isi

    def get_connectivity(self, to_model):
        conns={}
        target_id=self.units_dic[to_model].population.ids[0]
        for model, u in self.units_dic.iteritems():
            if not to_model==model:
                pre_id=self.units_dic[model].population.ids
                conn=nest.GetStatus(nest.FindConnections(pre_id))

                targets = [ c['source'] for c in conn if c['target']==target_id]
                #status=nest.GetStatus(targets) 
                weights = [ c['weight'] for c in conn if c['target']==target_id]
                conns[model]=[len(targets), numpy.mean(weights)]#, targets]
           

    def plot(self, ax,  model):
        
        mean_rates, mean_isi, std_isi, CV_isi=self.get_spike_statistics(model)
        
        stats='Rate: {0} (Hz) ISI CV: {1}'.format(round(mean_rates,2),round(CV_isi,1))
               
        pop=self.units_dic[model].population     
        if not pop.signaled['V_m']:
            self.get_voltage_trace(model) 
            
        pop.signals['V_m'].my_set_spike_peak( 15, spkSignal= pop.signals['spikes'], start=self.start_rec ) 
                         
        pylab.rcParams.update( {'path.simplify':False}    )
            
        pop.signals['V_m'].plot(display=ax)
        ax.set_title(stats)
        ax.set_xlim([self.start_rec, self.sim_time])
        
class Single_units_in_vitro(Inhibition_base):
    
    def __init__(self, threads=1, start_rec=1, sim_time=1000, **kwargs):
        super( Single_units_in_vitro, self ).__init__(threads, start_rec, sim_time, **kwargs)       
        # In order to be able to convert super class object to subclass object        
        
        if 'model_name' in kwargs.keys(): 
            self.model_name=kwargs['model_name']
        
        if 'n' in kwargs.keys():
            self.n=kwargs['n']
        else:
            self.n=1
        
        if 'par_rep' in kwargs.keys(): 
            self.par.update(kwargs['par_rep'])   
        

            
    def calibrate(self):           
        '''
        Compute all dependent variables, it is done by creating unit
        and structure object using model and network params from 
        model_params.py
        '''

        v=self.network_params['units_neuron'][self.model_name]
        v['n']=self.n
        self.units_list.append(Units_neuron(self.model_name,v))
        self.units_dic[self.model_name]=self.units_list[-1]
        
        self.calibrated=True                  
                

    def build(self):
        '''Create all nodes, used in the model'''
        if self.built: return
        if not self.calibrated: self.calibrate()
    
        u=self.units_dic[self.model_name]
        
        my_nest.MyLoadModels( self.model_dict, [u.model] )
        group=MyGroup(model = u.model, n=u.n, params = {'I_e':u.I_vitro},
                      sd=True, sd_params={'start':self.start_rec,  'stop':self.sim_time})     
           
        self.units_dic[u.name].set_population(group)           
        self.built=True       


    def IF_curve(self,  currents, tStim):    
        if not self.built: self.build()
        if not self.calibrated: self.calibrate()

        pop=self.units_dic[self.model_name].population
        data=[]
        for pop_id in sorted(pop.ids):
            data.append(pop.IF(currents, tStim=tStim, id=pop_id))   
        
        data=numpy.array(data)
        
        if self.n==1:idx=0
        else:idx=slice(0,self.n)
        
        
        curr=data[:,0][idx]
        fIsi=data[:,1][idx]
        mIsi=data[:,2][idx]
        lIsi=data[:,3][idx]
        
        return curr, fIsi, mIsi, lIsi
    
    def IF_variation(self,currents, tStim, randoization=['C_m']):
        if not self.built: self.build()
        if not self.calibrated: self.calibrate()
        self.randomize_params(randoization)
        return self.IF_curve(currents, tStim)    
    
    def IV_curve(self,  currents, tStim):    
        if not self.built: self.build()
        if not self.calibrated: self.calibrate()
  
        pop=self.units_dic[self.model_name].population
        mm_params={'interval':0.1, 'start':self.start_rec,  'stop':self.sim_time, 
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
        if not self.built: self.build()
        if not self.calibrated: self.calibrate()
        
        pop=self.units_dic[self.model_name].population
        mm_params={'interval':0.1, 'start':self.start_rec,  'stop':self.sim_time, 
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
           
class Bcpnn_h0(Inhibition_base):    
    
    def __init__(self,  par_rep={}, perturbation=None, **kwargs):
        super( Bcpnn_h0, self ).__init__(par_rep, perturbation, **kwargs)       
        # In order to be able to convert super class object to subclass object   
        self.class_default_params=Par_bcpnn_h0
        
    @property
    def path_data(self):        
        return '/afs/nada.kth.se/home/w/u1yxbcfw/results/papers/bcpnn'+'/'+self.name +'/'
    
    @property
    def path_pictures(self):
        return '/afs/nada.kth.se/home/w/u1yxbcfw/projects/papers/bcpnnbg/pictures'+'/'+self.name +'-'
        
class Bcpnn_h1(Bcpnn_h0):    
    
    def __init__(self,  par_rep={}, perturbation=None, **kwargs):
        super( Bcpnn_h1, self ).__init__(par_rep, perturbation, **kwargs)       
        # In order to be able to convert super class object to subclass object   
        self.class_default_params=Par_bcpnn_h1
    
class TestStructureInhibition_base(unittest.TestCase):
    
    kwargs={'save_conn':False, 'sub_folder':'unit_testing', 'verbose':False}
    par_rec={'simu':{'start_rec':1.0, 'stop':100.0,'sim_time':100., 
                     'sd_params':{'to_file':True, 'to_memory':False},
                     'threads':4, 'print_time':False},
             'netw':{'size':500.0}}
    
    nest.sr("M_WARNING setverbosity") #silence nest output
    
    def setUp(self):
        
        self.class_network_construction=Inhibition_base
        self.fileName=self.class_network_construction().path_data+'network'
        self.model_list=['M1','M2', 'FS', 'GA', 'GI', 'ST', 'SN']
            
    def test_inputs(self):
        network=self.class_network_construction(self.par_rec, **self.kwargs)
        network.inputs()
        nodes=[]
        for val in network.par['netw']['input'].values():
            nodes+=val['nodes']
            
        self.assertListEqual(sorted(nodes), sorted(network.input_params.keys()))
    
    def test_build_connect_run(self):     
        network=self.class_network_construction(self.par_rec, **self.kwargs)
        
        for val in network.par['conn'].values():
            if val['delay_setup']['type']=='uniform':
                val['delay_setup']['type']='constant'
                val['delay_setup']['params']=(val['delay_setup']['params']['max']+val['delay_setup']['params']['min'])/2.
        
        network.build()    
        
        for s in network.structures:
            fan_in=float(s.n_conn)/s.target.n   
            fan_in0=network.par['conn'][s.name]['fan_in']
            #print s.name, s.n_conn, fan_in/100.,fan_in0/100.
            self.assertAlmostEqual(fan_in/100.,fan_in0/100., 1)  
        
        network.connect()
        network.run()
        data_to_disk.pickle_save(network, self.fileName)
    
    def test_get_firing_rate(self):
        
        network=data_to_disk.pickle_load(self.fileName)
        my_nest.SetKernelStatus({'data_path':network.path_nest})
        fr=network.get_firing_rate(self.model_list)
        
        for k in self.model_list:
            self.assertEqual(len(fr[k][0]), network.par['simu']['sim_time']-1)
            self.assertEqual(len(fr[k][1]), network.par['simu']['sim_time']-1)

    def test_get_firing_rate_sets(self):
        
        network=data_to_disk.pickle_load(self.fileName)
        my_nest.SetKernelStatus({'data_path':network.path_nest})
        d=network.get_firing_rate_sets(self.model_list)
        
        for k in self.model_list:
            self.assertEqual(len(d[k][0]), network.par['simu']['sim_time']-1)
            self.assertEqual(d[k][1].shape[1], network.par['simu']['sim_time']-1)
            self.assertEqual(d[k][1].shape[0], network.par['node'][k]['n_sets'])

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
                

class TestStructureSlow_wave(TestStructureInhibition_base):
    
    def setUp(self):
        self.par_rec['netw'].update({'sub_sampling':{'M1':10.0,'M2':10.0}})     
        self.class_network_construction=Slow_wave
        
        self.fileName=self.class_network_construction().path_data+'network'
        self.model_list=['M1','M2', 'FS', 'GA', 'GI', 'ST', 'SN']
        
class TestStructureBcpnn_h0(TestStructureInhibition_base):
    
    def setUp(self):
        
        self.par_rec['netw'].update({'sub_sampling':{'M1':20.0,'M2':20.0, 'CO':20.0}})     
        self.class_network_construction=Bcpnn_h0
        
        self.fileName=self.class_network_construction().path_data+'network'
        self.model_list=['CO', 'M1','M2', 'FS', 'GA', 'GI', 'ST', 'SN']
            
class TestStructureBcpnn_h1(TestStructureBcpnn_h0):

    def setUp(self):
        self.class_network_construction=Bcpnn_h1        
        self.fileName=self.class_network_construction().path_data+'network'
        self.model_list=['CO', 'M1','M2', 'F1', 'F2', 'GA', 'GI', 'ST', 'SN']
          
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestStructureSlow_wave)
    unittest.TextTestRunner(verbosity=2).run(suite)
    #unittest.main()             
        