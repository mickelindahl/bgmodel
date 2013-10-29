
import numpy
import pylab
import os
import sys
import random

# Get directory where model and code resides 
model_dir=   '/'.join(os.getcwd().split('/')[0:-1])    
code_dir=  '/'.join(os.getcwd().split('/')[0:-2])  

# Add model, code and current directories to python path
sys.path.append(os.getcwd())  
sys.path.append(model_dir)
sys.path.append(code_dir+'/nest_toolbox') 

          
# Imports dependent on adding code model and nest_toolbox path
from model_params import models, network                                  
from src import my_nest, misc
from src.my_population import MyGroup, MyPoissonInput, MyLayerGroup,MyLayerPoissonInput 
from multiprocessing import Pool
from numpy.random import random_integers
import nest.topology as tp
import nest
numpy.random.seed(0)
N_GPE=30
N_MSN=500
N_STN=100

BASE_RATE_GPE=25
BASE_RATE_MSN=0.1
BASE_RATE_STN=10.0

SYNAPSE_MODEL_BACKROUND_GPE=['GPE_SNR_gaba_p']
SYNAPSE_MODEL_BACKROUND_MSN=['MSN_SNR_gaba_p1']
SYNAPSE_MODEL_BACKROUND_STN=['STN_SNR_ampa_s']

SNR_INJECTED_CURRENT=400.0


def simulate_basa_line_SNr(msn_rate, gpe_rate, stn_rate, n_msn, n_gpe, n_stn, neuron_model, snr_current, sim_time=1000, threads=8, stn_syn='STN_SNR_ampa_s'):

    SNR_INJECTED_CURRENT=snr_current
    SYNAPSE_MODEL_BACKROUND_STN=[stn_syn]
    
    model_list, model_dict=models()
    my_nest.ResetKernel(threads=threads)       
    my_nest.MyLoadModels( model_dict, neuron_model )
    my_nest.MyLoadModels( model_dict, SYNAPSE_MODEL_BACKROUND_MSN)       
    my_nest.MyLoadModels( model_dict, SYNAPSE_MODEL_BACKROUND_GPE)      
    my_nest.MyLoadModels( model_dict, SYNAPSE_MODEL_BACKROUND_STN)      
    
    SNR_list=[] # List with SNR groups for synapse. 
    
    if n_msn>0: MSN_base=MyPoissonInput(n=n_msn, sd=True)
    if n_gpe>0: GPE=MyPoissonInput(n=n_gpe, sd=False)
    if n_stn>0: STN=MyPoissonInput(n=n_stn, sd=False)
    
    if n_msn>0: MSN_base.set_spike_times(rates=[ msn_rate], times=[1], t_stop=sim_time, seed=0)    
    if n_gpe>0: GPE.set_spike_times(rates=[gpe_rate], times=[1], t_stop=sim_time, seed=0)     
    if n_stn>0: STN.set_spike_times(rates=[stn_rate], times=[1], t_stop=sim_time, seed=0)     
           
    I_e=my_nest.GetDefaults(neuron_model[0])['I_e']+SNR_INJECTED_CURRENT

    SNR=MyGroup( neuron_model[0], n=1, sd=True, params={'I_e':I_e}, mm_dt=.1, mm=True)    

        
    if n_msn>0: my_nest.Connect(MSN_base[:], SNR[:]*len(MSN_base[:]), model=SYNAPSE_MODEL_BACKROUND_MSN[0])
    if n_gpe>0: my_nest.Connect(GPE[:],SNR[:]*len(GPE[:]), model=SYNAPSE_MODEL_BACKROUND_GPE[0])   
    if n_stn>0: my_nest.Connect(STN[:],SNR[:]*len(STN[:]), model=SYNAPSE_MODEL_BACKROUND_STN[0])
                    
    my_nest.MySimulate(sim_time)    
    
    SNR.get_signal( 's', start=0, stop=sim_time )
    
    meanRate=round(SNR.signals['spikes'].mean_rate(1000,sim_time),1)
    spk=SNR.signals['spikes'].time_slice(1000,sim_time).raw_data()
    CV=numpy.std(numpy.diff(spk[:,0],axis=0))/numpy.mean(numpy.diff(spk[:,0],axis=0))

    SNR.get_signal( 'v',recordable='V_m', start=0, stop=sim_time )  
    SNR.signals['V_m'].my_set_spike_peak( 15, spkSignal= SNR.signals['spikes'] ) 
    pylab.rcParams.update( {'path.simplify':False}    )
    SNR.signals['V_m'].plot()
    pylab.title(str(meanRate)+ 'Hz, CV='+str(CV))
    pylab.show()
    
   
    return 

def simulate_basa_line_STN(ctx_rate, gpe_rate, n_ctx, n_gpe, neuron_model, syn_models, stn_current, sim_time=1000, threads=8):
    
    
    model_list, model_dict=models()
    my_nest.ResetKernel(threads=threads)       
    my_nest.MyLoadModels( model_dict, neuron_model )
    my_nest.MyLoadModels( model_dict, syn_models)       
    
    SNR_list=[] # List with SNR groups for synapse. 
    
    if n_ctx>0: CTX=MyPoissonInput(n=n_ctx, sd=True)
    if n_gpe>0: GPE=MyPoissonInput(n=n_gpe, sd=False)
    
    if n_ctx>0: CTX.set_spike_times(rates=[ ctx_rate], times=[1], t_stop=sim_time, seed=0)    
    if n_gpe>0: GPE.set_spike_times(rates=[gpe_rate], times=[1], t_stop=sim_time, seed=0)     
           
    I_e=my_nest.GetDefaults(neuron_model[0])['I_e']+stn_current

    STN=MyGroup( neuron_model[0], n=1, sd=True, params={'I_e':I_e}, 
                 mm_dt=.1, mm=True)    
        
    if n_ctx>0: my_nest.Connect(CTX[:], STN[:]*len(CTX[:]), model=syn_models[0])
    if n_gpe>0: my_nest.Connect(GPE[:],STN[:]*len(GPE[:]), model=syn_models[1])   

                    
    my_nest.MySimulate(sim_time)    
    
    STN.get_signal( 's', start=0, stop=sim_time )
    
    meanRate=round(STN.signals['spikes'].mean_rate(1000,sim_time),1)
    spk=STN.signals['spikes'].time_slice(1000,sim_time).raw_data()
    CV=numpy.std(numpy.diff(spk[:,0],axis=0))/numpy.mean(numpy.diff(spk[:,0],axis=0))

    STN.get_signal( 'v',recordable='V_m', start=0, stop=sim_time )   
    STN.signals['V_m'].my_set_spike_peak( 15, spkSignal= STN.signals['spikes'] ) 
    
             
    pylab.rcParams.update( {'path.simplify':False}    )
    
    STN.signals['V_m'].plot()
    pylab.title(str(meanRate)+ 'Hz, CV='+str(CV))
    pylab.show()
    
   
    return        
 
def simulate_basa_line_GPe(msn_rate, stn_rate, gpe_rate,  n_msn, n_stn, n_gpe, neuron_model, syn_models, gpe_current, sim_time=1000, threads=8):
    
    
    model_list, model_dict=models()
    my_nest.ResetKernel(threads=threads)       
    my_nest.MyLoadModels( model_dict, neuron_model )
    my_nest.MyLoadModels( model_dict, syn_models)       
    
    SNR_list=[] # List with SNR groups for synapse. 
    
    if n_msn>0: MSN=MyPoissonInput(n=n_msn, sd=False)
    if n_stn>0: STN=MyPoissonInput(n=n_stn, sd=False)
    if n_gpe>0: GPE=MyPoissonInput(n=n_gpe, sd=False)
    
    if n_msn>0: MSN.set_spike_times(rates=[msn_rate], times=[1], t_stop=sim_time, seed=0)    
    if n_stn>0: STN.set_spike_times(rates=[stn_rate], times=[1], t_stop=sim_time, seed=0)     
    if n_gpe>0: GPE.set_spike_times(rates=[gpe_rate], times=[1], t_stop=sim_time, seed=0)  
               
    I_e=my_nest.GetDefaults(neuron_model[0])['I_e']+gpe_current

    GPE_target=MyGroup( neuron_model[0], n=1, sd=True, params={'I_e':I_e}, 
                 mm_dt=.1, mm=True)    
        
    if n_msn>0: my_nest.Connect(MSN[:], GPE_target[:]*len(MSN[:]), model=syn_models[0])
    if n_stn>0: my_nest.Connect(STN[:],GPE_target[:]*len(STN[:]), model=syn_models[1])   
    if n_gpe>0: my_nest.Connect(GPE[:],GPE_target[:]*len(GPE[:]), model=syn_models[2])   
                    
    my_nest.MySimulate(sim_time)    
    
    GPE_target.get_signal( 's', start=0, stop=sim_time )
    
    meanRate=round(GPE_target.signals['spikes'].mean_rate(1000,sim_time),1)
    spk=GPE_target.signals['spikes'].time_slice(1000,sim_time).raw_data()
    CV=numpy.std(numpy.diff(spk[:,0],axis=0))/numpy.mean(numpy.diff(spk[:,0],axis=0))

    GPE_target.get_signal( 'v',recordable='V_m', start=0, stop=sim_time )   
    GPE_target.signals['V_m'].my_set_spike_peak( 15, spkSignal= GPE_target.signals['spikes'] ) 
    
             
    pylab.rcParams.update( {'path.simplify':False}    )
    
    GPE_target.signals['V_m'].plot()
    pylab.title(str(meanRate)+ 'Hz, CV='+str(CV))
    pylab.show()
    
   
    return        
    
def simulate_network(params_msn_d1, params_msn_d2, params_stn,
                     synapse_models, sim_time, seed, I_e_add, threads=1, 
                     start_rec=0, model_params={}):    
    '''
        params_msn_d1 - dictionary with timing and burst freq setup for msn
                     {'base_rates':[0.1, 0.1, ..., 0.1], #Size number of actions 
                      'mod_rates': [[20,0,...,0],
                                    [0,20,...,0],...[0,0,...,20]] #size number of actions times number of events   
                      'mod_times':[[500,1000],[1500,2000],[9500,10000]] # size number of events 
                      'n_neurons':500}
                      
        params_msn_d2 - dictionary with timing and burst freq setup for gpe
        params_stn    - dictionary {'rate':50}
                     same as params_msn
        neuron_model - string, the neuron model to use 
        synapse_models - dict, {'MSN':'...', 'GPE':,'...', 'STN':'...'}
        sim_time - simulation time
        seed - seed for random generator
        I_e_add - diabled
        start_rec - start recording from
        model_params - general model paramters
    '''

    
    I_e_add={'SNR':300, 'STN':0,'GPE':30}
    f=0.01#0.01#0.5
    
    I_e_variation={'GPE':25*f,'SNR':100*f,'STN':10*f}
    
    my_nest.ResetKernel(threads=8) 
    numpy.random.seed(seed)
    
    params = {'conns':{'MSN_D1_SNR':{'syn':synapse_models[0]},   
                       'GPE_SNR':{'syn':synapse_models[1]}}}   
    
    params=misc.dict_merge(model_params, params)
               
    model_list, model_dict = models()
    group_list, group_dict, connect_list, connect_params = network(model_dict, params)
    print connect_params
    

    
    
    groups={}
    for name, model, setup in group_list:
        
        # Update input current
        my_nest.MyLoadModels( model_dict, [model] )
        if name in I_e_add.keys():
            I_e=my_nest.GetDefaults(model)['I_e']+I_e_add[name]
            my_nest.SetDefaults(model,{'I_e':I_e})
            
        groups[name]=[]
        for action in range(connect_params['misc']['n_actions']):
            if model in ['MSN_D1_spk_gen','MSN_D2_spk_gen']:
                group=MyPoissonInput(params=setup, sd=True, sd_params={'start':start_rec, 'stop':sim_time})
            else:
                group=MyGroup(params=setup, sd=True, mm=False, mm_dt = 0.1,
                               sd_params={'start':start_rec, 'stop':sim_time} )
                
            groups[name].append(group)
    
    for action in range(connect_params['misc']['n_actions']):
        groups['MSN_D1'][action].set_spike_times(list(params_msn_d1['mod_rates'][action]), 
                                        list(params_msn_d1['mod_times']), sim_time, 
                                        ids=groups['MSN_D1'][action].ids)
        groups['MSN_D2'][action].set_spike_times(params_msn_d2['mod_rates'][action], 
                                        params_msn_d2['mod_times'], sim_time, 
                                        ids=groups['MSN_D2'][action].ids)
      
    # Create neurons and synapses
    for source, target, props  in connect_list:
        my_nest.MyLoadModels( model_dict, [props['model']] )
        
        
        for action in range(connect_params['misc']['n_actions']):
            
            pre=list(groups[source][action].ids)
            post=list(groups[target][action].ids)
            my_nest.MyRandomConvergentConnect(pre, post, params=props)
           
    STN_CTX_input_base=my_nest.Create('poisson_generator',params={'rate':params_stn['rate'], 'start':0., 'stop':sim_time})
    my_nest.MyLoadModels( model_dict, ['CTX_STN_ampa_s'] )
    
    for action in range(connect_params['misc']['n_actions']): 
        my_nest.DivergentConnect(STN_CTX_input_base, groups['STN'][action].ids, 
                                 model='CTX_STN_ampa_s')
    
    
    my_nest.MySimulate(sim_time)    
    
    for action in range(connect_params['misc']['n_actions']):     
        groups['MSN_D1'][action].get_signal( 's', start=start_rec, stop=sim_time )
        groups['MSN_D2'][action].get_signal( 's', start=start_rec, stop=sim_time )
        groups['GPE'][action].get_signal( 's', start=start_rec, stop=sim_time )    
        groups['SNR'][action].get_signal( 's', start=start_rec, stop=sim_time )    
        groups['STN'][action].get_signal( 's', start=start_rec, stop=sim_time )    
    
    return groups



def inspect_network():    
    model_list, model_dict = models()
    layer_list, connect_list = network(model_dict)
    
    
    # Create neurons and synapses
    layer_dic={}  
    for name, model, props  in layer_list:
        my_nest.MyLoadModels( model_dict, [model[1]] )

    
        #! Create layer, retrieve neurons ids per elements and p
        if model[0]=='spike_generator':
            layer=MyLayerPoissonInput(layer_props=props, sd=False)
        else:  
            layer=MyLayerGroup(layer_props=props, sd=True, mm=True, mm_dt = 0.1 )
        layer_dic[name]=layer
    
    
    # Connect populations
    for conn in connect_list:
        my_nest.MyLoadModels( model_dict, [conn[2]['synapse_model']] )
        name=conn[0] + '_' + conn[1]+'_'+conn[3]    
        tp.ConnectLayers(layer_dic[conn[0]].layer_id, layer_dic[conn[1]].layer_id, conn[2])
        layer_dic[conn[1]].add_connection(source=layer_dic[conn[0]], type=conn[3], props=conn[2])
        
    return layer_dic
    
    
    
     