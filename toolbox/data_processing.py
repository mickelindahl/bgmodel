'''
Created on Aug 7, 2013

@author: lindahlm
'''
import copy
import numpy
import pylab
import os
import random
from toolbox import misc, signal_processing, data_to_disk

class Data_unit(object):
    '''
    classdocs
    Class that neural data produced
    '''
    
    def __init__(self, name):
        '''
        Constructor
        '''
        self.ids=None
        self.isis=None
        self.isis_ids=None
        self.name=name
        self.sets=None
        self.pds=None
        self.rasters=None
        self.raster_ids=None #Ids starting at 0
        self.firing_rate=None
        self.rasters_sets=[]
        self.rasters_sets_ids=[]
        self.start=None
        self.stop=None
        self.times=None
        self.phase_spk=[]
        self.phase_spk_conv=[]
        self.phase_band_pass=[]
        self.phase_hilbert=[]
        
        
        
    def merge(self, other):
        
        new_obj=copy.deepcopy(self)
        new_other=copy.deepcopy(other)
        if new_obj.rasters is not None and new_other.rasters is not None:
            
            # Shift ids of other raster
            new_other.rasters[1]+=len(self.ids)
            new_obj.rasters=numpy.append(new_obj.rasters, new_other.rasters, 1)
        if new_obj.firing_rate is not None and new_other.firing_rate is not None:
            new_obj.firing_rate=(new_obj.firing_rate+ new_other.firing_rate)/2.
            
        return new_obj
        
    
    def sample(self, n_sample, seed=1):
        random.seed(seed)
        ids=numpy.unique(self.rasters[1])
        if len(ids)<n_sample:
            sample=ids
        else:
            sample=random.sample(ids, n_sample)
        #sample=random.sample(self.raster_ids, n_sample)
    
        truth_val=numpy.zeros(len(self.rasters[1] ))==1
        for s in sample:
            truth_val+=self.rasters[1] == s           
        
        self.truth_val_sample=truth_val
        self.idx_sample=self.rasters[1,truth_val]
        
        if len(self.truth_val_sample):
            return self.rasters[:, self.truth_val_sample ]
        else: 
            return []

    def set_firing_rate(self, d):
        self.firing_rate_time=d[0]
        self.firing_rate=d[1]   

    def set_firing_rate_sets(self, d):
        self.firing_rate_sets_time=d[0]
        self.firing_rate_sets=d[1]  
        
    def set_isis(self, d):
        self.isis_ids=d[0]
        self.isis=d[1]   

    def set_ids(self,ids):
        self.ids=ids
    
    def set_mean_rates(self, d):
        self.mean_rates_ids=d[0]
        self.mean_rates=d[1]   
    
    def set_rasters(self, d):
        self.rasters=d[0]
        self.raster_ids=d[1]
        
    def set_rasters_sets(self, d):
        
        for dd in d:
            self.rasters_sets.append(dd[0])
            self.rasters_sets_ids.append(dd[1])
        

    def set_times(self, times):
        self.times=times     
    
    def convert2bin(self, start, stop, n_sample, seed=1):
        sample=self.sample(n_sample, seed)
        if len(sample):
            return misc.convert2bin(sample, start, stop, False, 1) 
        else:
            return []
    
    def convolve(self, data, kernel_extent, kernel_type, kernel_params):
        return misc.convolve(data, kernel_extent, kernel_type, 
                           axis=0, single=False, params=kernel_params, no_mean=True)
               
    def get_power_density_spectrum(self, load, save_at, start, stop, NFFT=256, kernel_extent=20.0, kernel_type='gaussian',
                 kernel_params={'std_ms':10, 'fs':1000.0}, fs=1000.0):
        
        if not load:
            pds=[[0],[0]]
            firing_rate=self.firing_rate[(self.firing_rate_time>start) + (self.firing_rate_time<stop)]
            if numpy.mean(firing_rate) >0:
                firing_rate/=numpy.mean(firing_rate)
            
            #firing_rate=self.convolve(firing_rate, kernel_extent, kernel_type, kernel_params)
            
                Pxx, f =signal_processing.psd(firing_rate, NFFT, fs, noverlap=NFFT/2)
                pds=numpy.array([f, Pxx])
                data_to_disk.pickle_save(pds, save_at)
        else:
            pds=data_to_disk.pickle_load(save_at)
            
        self.pds=pds

    def get_phase(self, load, save_at, start, stop, lowcut, highcut, order, fs, kernel_extent, kernel_type, kernel_params, seed=1):       
        '''
        Returns the phase of the population firing rate filters in the band
        lowcut to highcut. 
        '''
        
        if not load:
            signal_raw=self.firing_rate
            signal_con=self.convolve(signal_raw, kernel_extent, kernel_type, kernel_params)
        
            signal_bdp=signal_processing.butter_bandpass_lfilter(signal_con, lowcut, highcut, fs, order=order)
            signal_bdp_filtfilt=signal_processing.butter_bandpass_filtfilt(signal_con, lowcut, highcut, fs, order=order)
            signal_phase=signal_processing.my_hilbert(signal_bdp_filtfilt)
            signal_phase_con=signal_processing.my_hilbert(signal_con)
            #except:
            #signal_phase=[]
            
            pylab.subplot(511).plot(signal_raw)    
            pylab.subplot(512).plot(signal_con)
            pylab.subplot(513).plot(signal_bdp)
            pylab.subplot(513).plot(signal_bdp_filtfilt)
            pylab.subplot(514).plot(numpy.angle(signal_phase))
            pylab.subplot(515).plot(numpy.angle(signal_phase_con))
            pylab.show() 
                
            data_to_disk.pickle_save(signal_phase, save_at)     
        else:
            signal_phase=data_to_disk.pickle_load(save_at)
        self.signal_phase=signal_phase
        
    def get_phases(self, load, save_at, start, stop, lowcut, highcut, order, fs, kernel_extent, kernel_type, kernel_params, n_sample=10, seed=1):       
        '''
        Returns the phase of the population firing rate filters in the band
        lowcut to highcut. 
        '''
        signals_binned=self.convert2bin(start, stop, n_sample, seed)
        
        
        if not load:
            for signal_raw in signals_binned:
                signal_con=self.convolve(signal_raw, kernel_extent, kernel_type, kernel_params)
                signal_bdp=signal_processing.butter_bandpass_lfilter(signal_con, lowcut, highcut, fs, order=order)
                signal_bdp_filtfilt=signal_processing.butter_bandpass_filtfilt(signal_con, lowcut, highcut, fs, order=order)
                signal_phase=signal_processing.my_hilbert(signal_bdp_filtfilt)
                signal_phase_con=signal_processing.my_hilbert(signal_con)
            #except:
            #signal_phase=[]
            
                pylab.subplot(511).plot(signal_raw)    
                pylab.subplot(512).plot(signal_con)
                pylab.subplot(513).plot(signal_bdp)
                pylab.subplot(513).plot(signal_bdp_filtfilt)
                pylab.subplot(514).plot(numpy.angle(signal_phase))
                pylab.subplot(515).plot(numpy.angle(signal_phase_con))
                pylab.show() 
                
            data_to_disk.pickle_save(signal_phase, save_at)     
        else:
            signal_phase=data_to_disk.pickle_load(save_at)
        self.signal_phase=signal_phase        
             
        
       

class Data_units_dic(object):
    
    def __init__(self, models):
        
        self.dic={}
        for model in models:
            self.dic[model]=Data_unit(model)
    
        self.model_list=copy.deepcopy(models)
        
        
    def __getitem__(self, model):
        if model in self.model_list:
            return self.dic[model]
        else:
            raise Exception("Model %d is not present in the Data_units_dic. See model_list()" %model)

    def __setitem__(self, model, val):
        assert isinstance(val, Data_unit), "An Data_units_dic object can only contain Data_unit objects"
        self.dic[model] = val        
        if not model in self.model_list:
            self.model_list.append(model)
        
    def set_firing_rate(self, d):
        for model, val in d.iteritems():
            self.dic[model].set_firing_rate(val)     
               
    def set_firing_rate_sets(self, d):
        for model, val in d.iteritems():
            self.dic[model].set_firing_rate_sets(val)     
            
    def set_ids(self, d):
        for model, val in d.iteritems():
            self.dic[model].set_ids(val)

    def set_isis(self, d):
        for model, val in d.iteritems():
            self.dic[model].set_isis(val)        

    def set_mean_rates(self, d):
        for model, val in d.iteritems():
            self.dic[model].set_mean_rates(val)

    def set_rasters(self, d):
        for model, val in d.iteritems():
            self.dic[model].set_rasters(val)        

    def set_rasters_sets(self, d):
        for model, val in d.iteritems():
            self.dic[model].set_rasters_sets(val)          
                
class Data_units_relation(object):
    '''
    classdocs
    Class that represent coherence between two data units (can be same data units)
    '''
    
        
    def __init__(self, label, **kwargs):
        '''
        Constructor
        '''
        self.cohere=[]
        self.label=label
         
        self.start=None
        self.stop=None

        
    def get_coherence(self, load, save_at, binned_data1, binned_data2, start, stop, NFFT, 
                      kernel_extent, kernel_type, kernel_params, fs=1000.0):
        
        if load==0 or ((load==2) and os.path.exists(save_at)):
            if len(binned_data1)>1 and len(binned_data2)>1: 
                conv_data1=misc.convolve(binned_data1, kernel_extent, kernel_type, params=kernel_params)
                conv_data2=misc.convolve(binned_data2, kernel_extent, kernel_type, params=kernel_params)

                f, mean_Cxy =signal_processing.get_coherence(conv_data1, conv_data2, fs, NFFT, noverlap=NFFT/2)

            else:
                f, mean_Cxy =numpy.array([[0],[0]])   
            if not len(binned_data1):
                binned_data1=[[]]
            
            L=len(binned_data1[0])/NFFT
            if L!=1: 
                p_conf95=numpy.ones(len(f))*(1-0.05**(1/(L-1)))    
            else:
                p_conf95=1.0
            data_to_disk.pickle_save([f, mean_Cxy, p_conf95], save_at)
        else:
            f, mean_Cxy, p_conf95=data_to_disk.pickle_load(save_at)

        
        self.cohere=numpy.array([ f, mean_Cxy, p_conf95]) 
    #def get_cross_spectrum(self, data_unit_1, data_unit_2, start, stop):
        
class Data_units_relation_dic(object):    
    def __init__(self, models_list, **kwargs):
        

        self.dic={}
        for model in models_list:
            self.dic[model]=Data_units_relation(model, **kwargs)
    
        self.relation_list=copy.deepcopy(models_list)
    
        
    def __getitem__(self, relation):
        if relation in self.relation_list:
            return self.dic[relation]
        else:
            raise Exception("Model %d is not present in the Data_units_dic. See model_list()" %relation)

    def __setitem__(self, i, val):
        assert isinstance(val, Data_unit), "An Data_units_dic object can only contain Data_unit objects"
        self.dic[i] = val       