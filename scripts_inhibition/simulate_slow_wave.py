'''
Created on Aug 12, 2013

@author: lindahlm
'''

from simulate_network import Network_model, Network_models_dic
from network_classes import Slow_wave
from toolbox import plot_settings 
from toolbox.my_axes import MyAxes 
import numpy
import pylab

from simulate_network_investigate import  check_setup, get_setup

class Network_model_slow_wave(Network_model):
    
    def __init__(self,  label, flag, size, start, stop, Use_class, threads, **kwargs):
        
        super( Network_model_slow_wave, self ).__init__( label, flag, size, start, stop, Use_class, threads,  **kwargs) 
        
        self.cycles=1
        self.rate_up=None
        self.rate_down=None
        
        if 'cycles' in kwargs.keys(): self.cycles=kwargs['cycles']
        if 'p_mod_rates' in kwargs.keys(): self.p_mod_rates=kwargs['p_mod_rates']
        
        rates={}
        self.rates_up={}
        self.rates_down={}
        for key, val in Slow_wave().par['node'].iteritems():
            if val['type']=='input': 
                if key[0]=='C':
                
                    rates[key]=val['rate'] 
        
                    self.rates_up[key]=val['rate']*(2-self.p_mod_rates) 
                    self.rates_down[key]=val['rate']*(self.p_mod_rates)           
                else:
                    self.rates_up[key]=val['rate']
                    self.rates_down[key]=val['rate']
        self.stop=self.cycles*1000.0
    
    def simulate(self, model_list, **kwargs):
    
        print '\n*******************\nSimulatation setup\n*******************\n'
        self.update_parms()
        kwargs.update({'par_in':self.params_in, 'perturbation':self.perturbation })
        inh=self.use_class(self.threads, self.start, self.stop, **kwargs)
        
        if self.perturbation: inh.par.apply_perturbations()
        inh.par.update_dependable_par()
        inh.calibrate()
        inh.inputs(self.rates_down, self.rates_up, self.cycles)
        inh.build()
        inh.randomize_params(['V_m', 'C_m'])
        inh.connect()
        inh.run(print_time=True)
        
        s=inh.get_simtime_data()
        s+=' Network size='+str(inh.par['netw']['size'])
        s+=' '+self.type_dopamine
        
        fr_dic=inh.get_firing_rate(model_list)   
        ids_dic=inh.get_ids(model_list)      
        isis_dic=inh.get_isis(model_list)          
        mrs_dic=inh.get_mean_rates(model_list)       
        raster_dic=inh.get_rasters(model_list)
                                  
        return s, fr_dic, ids_dic, isis_dic, mrs_dic, raster_dic
    

class Network_model_dic_slow_wave(Network_models_dic):
    
    def __init__(self,  threads, lesion_setup, setup_list_models, Network_model_class):
        
        super( Network_model_dic_slow_wave, self ).__init__( threads, lesion_setup, setup_list_models, Network_model_class) 
    
    
    def show_phase_processing_example(self):
        plot_settings.set_mode(pylab, mode='by_fontsize', w = 800.0, h = 800, fontsize=12)
        fig = pylab.figure( facecolor = 'w' )
        ax_list = []
        n_rows=4
        n_col=2
        ypoints=numpy.linspace(0.1, 0.75, n_rows)
        xpoints=numpy.linspace(0.1, 0.6, n_col)
        for x in xpoints:
            for y in ypoints:
                ax_list.append( MyAxes(fig, [ x,  y,  .8/(n_col+0.5), 0.8/(n_rows+1.5) ] ) )  
        
        xlim=[0,1000]
        model='GPE_I'
        ax=ax_list[0]
        ax.plot(self.data[model].phase_spk[0:2].transpose())   
        ax.set_xlim(xlim)                
        ax.set_title('Raw spike trains')
        ax.legend(['Neuron 1', 'Neuron 2'])
                
        ax=ax_list[1]
        ax.plot(self.data[model].get_setupphase_spk_conv[0:2].transpose())
        ax.set_xlim(xlim)
        ax.set_title('Convolved '+self.kernel_type+' '+str(self.kernel_extent)+' '+str(self.kernel_params))
        ax.legend(['Neuron 1', 'Neuron 2'])        
        
        ax=ax_list[2]
        ax.plot(self.data[model].phase_spk_conv[0:2].transpose())
        ax.set_xlim(xlim)
        ax.set_title('Bandpass low/high/order '+str(self.lowcut)+'/'+str(self.highcut)+'/'+str(self.order))
        ax.legend(['Neuron 1', 'Neuron 2'])        


def main():

    p_mode_rates=[0.8, 0.8, 0.95, 0.95]
    cycles=40
    pds_setup=[1024*4, 10., 'gaussian',{'std_ms':5, 'fs':1000.0}]
    cohere_setup=[1024*4, 10., 'gaussian',{'std_ms':5, 'fs':1000.0}, 40]
    pds_models=['GP', 'GA', 'GI', 'ST', 'SN']
    cohere_relations=['GA_GA', 'GI_GI', 'GA_GI','ST_GA', 'ST_GA']
        
    record_from_models=['M1', 'M2', 'FS', 'GA', 'GI', 'ST', 'SN']
    plot_models=pds_models
    plot_relations=cohere_relations
    setup_list=[]
    size=40000.0
    start=1000.0
    stop=41000.0
    use_class=Slow_wave

    check_setup()

    setup=get_setup()
    for s in setup:
        kwargs={}
        kwargs['perturbation']=s[1]
        kwargs.update({'cycles':cycles, 'p_mod_rates':0.9})
        setup_list.append([s[0]+'-dop', 'dop', size, start, stop, use_class, kwargs])
        setup_list.append([s[0]+'-no_dop', 'no_dop', size, start, stop, use_class, kwargs])
        
    labels=[sl[0] for sl in setup_list]
    kwargs_simulation={'sd_params':{'to_file':True, 'to_memory':False}}#, 'record_to': ['file']}}
    #, 'record_to': ['file']}}
    #'flush_records': True,
    nms=Network_models_dic(8, setup_list, Network_model_slow_wave)
    nms.simulate_example([1]*len(labels), labels, record_from_models, **kwargs_simulation)
    nms.signal_pds([1]*len(labels), labels, pds_models, pds_setup)
    nms.signal_coherence([1]*len(labels), labels, cohere_relations, cohere_setup)
    #nms.signal_phase([0]*2, [labels[3]], plot_models[5:8], phase_setup)
    
    fig=nms.show_compact(labels, plot_models, plot_relations, band=[0.5,1.5])
    nms.show_exclude_rasters(labels[0:4]+labels[16:18], plot_models, plot_relations, xlim=[5000.0,7000.0], xlim_pds=[0,5], xlim_coher=[0,5])
    pylab.show()
    
    
    
main()    





    