'''
Created on 4 Feb 2013

@author: pierre berthet
'''
# export LD_LIBRARY_PATH=/lib/:/usr/local/lib/nest/
#lamboot


#####################
# 1 module of 30 neurons connected to 2 pools of 30 neurons. These 2 pools don't have any recurrent connection but have inhibition to all the neurons in the other pool.

import nest, numpy as np, pylab as pl, time
import nest.raster_plot, nest.voltage_trace

start_time = time.time()

nest.ResetKernel()
# load bcpnn synapse module and iaf neuron with bias
if (not 'bcpnn_synapse' in nest.Models('synapses')):
    nest.Install('ml_module')
nest.SetKernelStatus({"overwrite_files": True})

# --------------------------------- SETTING UP NETWORK --------------------------------------

poisson_per_neuron_in_minic = 30   # no. poisson_1 generators to feed neurons in 'A' and 'B'
static_poisson_weight = 15.0 # 10.75 #12.10 #10.75 #13.0 #       # instantaneous epsp slope
static_inhib_weight = -1.00          # inhibitory connections from D2 neurons to their respective action neuron
static_excit_weight = 5.0          # excitatory connections between cortex and the minicolumns
static_excit_reward_weight = 1.5          # excitatory connections between reward and the minicolumns
neurons_per_input_pool = 60         # no. neurons in the input pools
neurons_per_reinforcement_pool = 30         # no. neurons in the reinforcement input pools
neurons_per_minicolumn = 20        # no. neurons in 'A' and 'B'
active_poisson_rate = 20.0        # Hz, fmax
inactive_poisson_rate = 0.0       # Hz, fmin

# synapse params
tau_j = 10.0
tau_i = 10.0
tau_e = 100.0
tau_p = 100000.0
epsilon = 0.001
gain = 0.0 # initial, do not inject any current during training. This is just to make life easier

newfontsize = 16.0 # font size for the figures text
labels_fontsize = 16.0 #font for the axis text

voltmeter = nest.Create("voltmeter")
nest.SetStatus(voltmeter,[{"to_file": True, "withtime": True}])
voltmeter2 = nest.Create("voltmeter")
nest.SetStatus(voltmeter2,[{"to_file": True, "withtime": True}])
voltmeterA = nest.Create("voltmeter")
nest.SetStatus(voltmeterA,[{"to_file": True, "withtime": True}])
voltmeterB = nest.Create("voltmeter")
nest.SetStatus(voltmeterB,[{"to_file": True, "withtime": True}])
voltmeter_in1 = nest.Create("voltmeter")
nest.SetStatus(voltmeter_in1,[{"to_file": True, "withtime": True}])
voltmeter_in2 = nest.Create("voltmeter")
nest.SetStatus(voltmeter_in2,[{"to_file": True, "withtime": True}])
voltmeter_rw1 = nest.Create("voltmeter")
nest.SetStatus(voltmeter_rw1,[{"to_file": True, "withtime": True}])
voltmeter_rw2 = nest.Create("voltmeter")
nest.SetStatus(voltmeter_rw2,[{"to_file": True, "withtime": True}])
bw_1 = nest.Create("multimeter", params={'record_from': ['bias','weight'], 'interval' :0.1})
bw_2 = nest.Create("multimeter", params={'record_from': ['bias','weight'], 'interval' :0.1})
#weight_1 = nest.Create("multimeter", params={'record_from': ['weight'], 'interval' :0.1})
#weight_2 = nest.Create("multimeter", params={'record_from': ['weight'], 'interval' :0.1})

# presynaptic neurons
cortical_input_1 = nest.Create('iaf_cond_alpha_bias', neurons_per_input_pool, params = {'fmax':20.0, 'tau_j': tau_j,'tau_e': tau_e,'tau_p':tau_p, 'epsilon': epsilon, 't_ref': 2.0, 'gain': gain})
cortical_input_2 = nest.Create('iaf_cond_alpha_bias', neurons_per_input_pool, params = {'fmax':20.0, 'tau_j': tau_j,'tau_e': tau_e,'tau_p':tau_p, 'epsilon': epsilon, 't_ref': 2.0, 'gain': gain})

# Reward neurons ## used to stimulate the pools
reward_first_input = nest.Create('iaf_cond_alpha_bias', neurons_per_reinforcement_pool, params = {'fmax':20.0, 'tau_j': tau_j,'tau_e': tau_e,'tau_p':tau_p, 'epsilon': epsilon, 't_ref': 2.0, 'gain': gain})
reward_second_input = nest.Create('iaf_cond_alpha_bias', neurons_per_reinforcement_pool, params = {'fmax':20.0, 'tau_j': tau_j,'tau_e': tau_e,'tau_p':tau_p, 'epsilon': epsilon, 't_ref': 2.0, 'gain': gain})

# postsynaptic neurons
A_D1_minicolumn = nest.Create('iaf_cond_alpha_bias', neurons_per_minicolumn, params = {'fmax':20.0, 'tau_j': tau_j,'tau_e': tau_e,'tau_p':tau_p, 'epsilon': epsilon, 't_ref': 2.0, 'gain': gain})
A_D2_minicolumn = nest.Create('iaf_cond_alpha_bias', neurons_per_minicolumn, params = {'fmax':20.0, 'tau_j': tau_j,'tau_e': tau_e,'tau_p':tau_p, 'epsilon': epsilon, 't_ref': 2.0, 'gain': gain})
B_D1_minicolumn = nest.Create('iaf_cond_alpha_bias', neurons_per_minicolumn, params = {'fmax':20.0, 'tau_j': tau_j,'tau_e': tau_e,'tau_p':tau_p, 'epsilon': epsilon, 't_ref': 2.0, 'gain': gain})
B_D2_minicolumn = nest.Create('iaf_cond_alpha_bias', neurons_per_minicolumn, params = {'fmax':20.0, 'tau_j': tau_j,'tau_e': tau_e,'tau_p':tau_p, 'epsilon': epsilon, 't_ref': 2.0, 'gain': gain})

# output neurons, used to check activity
actionA = nest.Create('iaf_cond_alpha_bias', 1, params = {'fmax':20.0, 'tau_j': tau_j,'tau_e': tau_e,'tau_p':tau_p, 'epsilon': epsilon, 't_ref': 2.0, 'gain': gain})
actionB = nest.Create('iaf_cond_alpha_bias', 1, params = {'fmax':20.0, 'tau_j': tau_j,'tau_e': tau_e,'tau_p':tau_p, 'epsilon': epsilon, 't_ref': 2.0, 'gain': gain})

# poisson generators
poisson_1 = nest.Create('poisson_generator', poisson_per_neuron_in_minic * neurons_per_input_pool, params = {'rate': active_poisson_rate})
poisson_2 = nest.Create('poisson_generator', poisson_per_neuron_in_minic * neurons_per_input_pool, params = {'rate': active_poisson_rate})
poisson_first_reward = nest.Create('poisson_generator', poisson_per_neuron_in_minic * neurons_per_reinforcement_pool, params = {'rate': active_poisson_rate})
poisson_second_reward = nest.Create('poisson_generator', poisson_per_neuron_in_minic * neurons_per_reinforcement_pool, params = {'rate': active_poisson_rate})

# spike detectors
first_detect = nest.Create("spike_recorder", params = {"withgid":True,"withtime":True})
second_detect = nest.Create("spike_recorder", params = {"withgid":True,"withtime":True})
detect_a = nest.Create("spike_recorder", params = {"withgid":True,"withtime":True})
detect_b = nest.Create("spike_recorder", params = {"withgid":True,"withtime":True})
detect_c = nest.Create("spike_recorder", params = {"withgid":True,"withtime":True})
detect_d = nest.Create("spike_recorder", params = {"withgid":True,"withtime":True})
detect_e = nest.Create("spike_recorder", params = {"withgid":True,"withtime":True})
detect_f = nest.Create("spike_recorder", params = {"withgid":True,"withtime":True})

#first_bias = nest.Create("multimeter", params={'record_from': ['bias','p_j'], 'interval' :0.1})
#nest.ConvergentConnect(A_D1_minicolumn, first_bias)
#second_bias = nest.Create("multimeter", params={'record_from': ['bias','p_j'], 'interval' :0.1})
#nest.ConvergentConnect(B_D1_minicolumn, second_bias)

# ------------------------------- CONNECTIONS ------------------------------------#

# connect recording devices

nest.ConvergentConnect(cortical_input_1, detect_a)
nest.ConvergentConnect(cortical_input_2, detect_b)
nest.ConvergentConnect(A_D1_minicolumn, detect_c)
nest.ConvergentConnect(B_D1_minicolumn, detect_d)
nest.ConvergentConnect(reward_first_input, detect_e)
nest.ConvergentConnect(reward_second_input, detect_f)
nest.Connect(actionA, first_detect)
nest.Connect(actionB, second_detect)
#nest.DivergentConnect(voltmeter, A_D1_minicolumn )
#nest.DivergentConnect(voltmeter2, B_D1_minicolumn )
#nest.DivergentConnect(voltmeter_in1, cortical_input_1 )
#nest.DivergentConnect(voltmeter_in2, cortical_input_2 )
#nest.ConvergentConnect( voltmeterA, actionA)
#nest.ConvergentConnect( voltmeterB, actionB)
nest.Connect(voltmeter, [A_D1_minicolumn[0]] )
nest.Connect(voltmeter2, [B_D1_minicolumn[0]] )
nest.Connect(voltmeter_in1, [cortical_input_1[0]] )
nest.Connect(voltmeter_in2, [cortical_input_2[0]] )
nest.Connect(voltmeter_rw1, [reward_first_input[0]] )
nest.Connect(voltmeter_rw2, [reward_second_input[0]] )
nest.Connect( voltmeterA, actionA)
nest.Connect( voltmeterB, actionB)
#nest.Connect(bw_1, [A_D1_minicolumn[0]])
#nest.Connect(bw_2, [B_D1_minicolumn[0]])

# connect neurons in one minicolumn to all neurons in the other minicolumn using static inhibitory connections
# connect neurons from poisson_1 generator to neurons in cortical_input_1 with static excitatory connections


#for neur_pois in cortical_input_1:
#    nest.ConvergentConnect(poisson_1[(neur_pois - 1) * poisson_per_neuron_in_minic: neur_pois * poisson_per_neuron_in_minic], [neur_pois], weight = static_poisson_weight, delay = 1.0)
for i in range(1,len(cortical_input_1)):
    nest.ConvergentConnect(poisson_1[(i-1)*poisson_per_neuron_in_minic:i*poisson_per_neuron_in_minic], [cortical_input_1[i-1]], weight=static_poisson_weight, delay = 1.0)  


#for neur_pois2 in cortical_input_2:
#    nest.ConvergentConnect(poisson_2[(neur_pois2 - 1) * poisson_per_neuron_in_minic: neur_pois2 * poisson_per_neuron_in_minic], [neur_pois2], weight = static_poisson_weight, delay = 1.0)

for i in range(1,len(cortical_input_2)):
    nest.ConvergentConnect(poisson_2[(i-1)*poisson_per_neuron_in_minic:i*poisson_per_neuron_in_minic], [cortical_input_2[i-1]], weight=static_poisson_weight, delay = 1.0)  

for i in range(1,len(reward_first_input)):
    nest.ConvergentConnect(poisson_first_reward[(i-1)*poisson_per_neuron_in_minic:i*poisson_per_neuron_in_minic], [reward_first_input[i-1]], weight=static_poisson_weight, delay = 1.0)    
    nest.ConvergentConnect(poisson_second_reward[(i-1)*poisson_per_neuron_in_minic:i*poisson_per_neuron_in_minic], [reward_second_input[i-1]], weight=static_poisson_weight, delay = 1.0)

#for neur_pois_r1 in reward_first_input:
#    
#    nest.ConvergentConnect(poisson_first_reward[(neur_pois_r1 - 1) * poisson_per_neuron_in_minic: neur_pois_r1 * poisson_per_neuron_in_minic], [neur_pois_r1], weight = static_poisson_weight, delay = 1.0)
#for neur_pois_r2 in reward_second_input:
#    nest.ConvergentConnect(poisson_second_reward[(neur_pois_r2 - 1) * poisson_per_neuron_in_minic: neur_pois_r2 * poisson_per_neuron_in_minic], [neur_pois_r2], weight = static_poisson_weight, delay = 1.0)

#for neuron1 in A_D1_minicolumn:
#    nest.DivergentConnect([neuron1], B_D1_minicolumn, weight = static_inhib_weight, delay = 1.0)
#
#for neuron2 in B_D1_minicolumn:
#    nest.DivergentConnect([neuron2], A_D1_minicolumn, weight = static_inhib_weight, delay = 1.0)
    
for neur_rew1 in reward_first_input:
    nest.DivergentConnect([neur_rew1], A_D1_minicolumn, weight = static_excit_reward_weight, delay = 1.0)
    nest.DivergentConnect([neur_rew1], B_D2_minicolumn, weight = static_excit_reward_weight, delay = 1.0)
for neur_rew2 in reward_second_input:  
    nest.DivergentConnect([neur_rew2], B_D1_minicolumn, weight = static_excit_reward_weight, delay = 1.0)
    nest.DivergentConnect([neur_rew2], A_D2_minicolumn, weight = static_excit_reward_weight, delay = 1.0)

    
# connect neurons in cortical_input_1 to neurons in first and second minicolumns with BCPNN synapses
#            ## /////// WARNING EXACT SAME INPUT TO THE 2 MINICOLUMNS \\\\\\\\ ###


nest.ConvergentConnect(A_D1_minicolumn, actionA, weight=static_excit_weight, delay = 5.0)
nest.ConvergentConnect(B_D1_minicolumn, actionB, weight=static_excit_weight, delay = 5.0)
nest.ConvergentConnect(A_D2_minicolumn, actionA, weight = static_inhib_weight, delay = 10.0)    # Delay D2 >> D1
nest.ConvergentConnect(B_D2_minicolumn, actionB, weight = static_inhib_weight, delay = 10.0)

nest.SetDefaults('bcpnn_synapse', params = {'gain': 0.70, 'K':1.0,'fmax': 20.0,'epsilon': epsilon,'delay':1.0,'tau_i': tau_i,'tau_j': tau_j,'tau_e': tau_e,'tau_p': tau_p})

for neur_pois in cortical_input_1:
    nest.DivergentConnect([neur_pois], A_D1_minicolumn, model='bcpnn_synapse')
    nest.DivergentConnect([neur_pois], B_D1_minicolumn, model='bcpnn_synapse')
    nest.DivergentConnect([neur_pois], A_D2_minicolumn, model='bcpnn_synapse')
    nest.DivergentConnect([neur_pois], B_D2_minicolumn, model='bcpnn_synapse')
    
for neur_pois in cortical_input_2:
    nest.DivergentConnect([neur_pois], A_D1_minicolumn, model='bcpnn_synapse')
    nest.DivergentConnect([neur_pois], B_D1_minicolumn, model='bcpnn_synapse')
    nest.DivergentConnect([neur_pois], A_D2_minicolumn, model='bcpnn_synapse')
    nest.DivergentConnect([neur_pois], B_D2_minicolumn, model='bcpnn_synapse')
    
    
# connect minicolumn to one output neuron
#for neur_mini in A_D1_minicolumn:
#    nest.ConvergentConnect(neur_mini, actionA)
#for neur_mini2 in B_D1_minicolumn:
#    nest.ConvergentConnect(neur_mini2, actionB)


# connect poisson_1 generators to output neurons
#nest.ConvergentConnect(first_post_poisson, first_neuron, weight = static_poisson_weight, delay = 1.0)
#nest.ConvergentConnect(second_post_poisson, second_neuron, weight = static_poisson_weight, delay = 1.0)

# ------------------------------- TRAINING SIMULATION -------------------------------------
total_simulation_time = 13000 #100000 #32000 # 50000
simulation_intervals = 500 # 300

# Record the weights during simulation. multiply by two because half connected to first, half connected to second
w_A_D1 = np.zeros(total_simulation_time/simulation_intervals)
w_B_D1 = np.zeros(total_simulation_time/simulation_intervals)
w_A_D2 = np.zeros(total_simulation_time/simulation_intervals)
w_B_D2 = np.zeros(total_simulation_time/simulation_intervals)
b_A_D1 = np.zeros(total_simulation_time/simulation_intervals)
b_B_D1 = np.zeros(total_simulation_time/simulation_intervals)
b_A_D2 = np.zeros(total_simulation_time/simulation_intervals)
b_B_D2 = np.zeros(total_simulation_time/simulation_intervals)

change_interval = 10 # how often to change the active/inactive to zero, lower = more often keep at 10 for %2,4,6,8,0
#plot_counter = 1

#weight_track = pl.figure(8008)
#neuron_1a = nest.GetStatus(nest.FindConnections([cortical_input_1[8]], target = [A_D1_minicolumn[8]]))
#neuron_2a = nest.GetStatus(nest.FindConnections([cortical_input_1[8]], target = [B_D1_minicolumn[8]]))
# Show that cortical stimulation only is not sufficient to elicit spikes in the minicolumns
print 'initial stimulation'
nest.SetStatus(poisson_1,{'rate': active_poisson_rate})
nest.SetStatus(poisson_first_reward,{'rate': inactive_poisson_rate})
nest.SetStatus(poisson_second_reward,{'rate': inactive_poisson_rate})
nest.Simulate(simulation_intervals)


#for i in range(0,total_simulation_time/simulation_intervals):
#    if i%6==0 or i%6==2 or i%6==4:
#        nest.SetStatus(poisson_1,{'rate': inactive_poisson_rate})
#        nest.SetStatus(poisson_2,{'rate': inactive_poisson_rate})
#        nest.SetStatus(poisson_first_reward,{'rate': inactive_poisson_rate})
#        nest.SetStatus(poisson_second_reward,{'rate': inactive_poisson_rate})
#        print 'inactive period'
#        nest.Simulate(simulation_intervals)
#    if i%6 ==1 or i%6==5:
#        nest.SetStatus(poisson_1,{'rate': active_poisson_rate})
#        nest.SetStatus(poisson_first_reward,{'rate': active_poisson_rate})
#        nest.SetStatus(poisson_second_reward,{'rate': inactive_poisson_rate})
#        print 'active CORTEX 1 + A D1 + B D2'
#        nest.Simulate(simulation_intervals)
#    if i%6 ==3:
#        nest.SetStatus(poisson_2,{'rate': active_poisson_rate})
#        nest.SetStatus(poisson_first_reward,{'rate': inactive_poisson_rate})
#        nest.SetStatus(poisson_second_reward,{'rate': active_poisson_rate})
#        print 'active CORTEX 2 + B D1 + A D2'The Higher Education Ordinance (SFS ) - Swedish Code of Statutes 1993:100
#        nest.Simulate(simulation_intervals)


number_of_cycles = 12
for i in range(0, number_of_cycles):
    if i%4==0 or i%4==2:
        nest.SetStatus(poisson_1,{'rate': inactive_poisson_rate})
        nest.SetStatus(poisson_2,{'rate': inactive_poisson_rate})
        nest.SetStatus(poisson_first_reward,{'rate': inactive_poisson_rate})
        nest.SetStatus(poisson_second_reward,{'rate': inactive_poisson_rate})
        print 'inactive period'
        nest.Simulate(simulation_intervals)
    if i%4 ==1:
        nest.SetStatus(poisson_1,{'rate': active_poisson_rate})
        nest.SetStatus(poisson_first_reward,{'rate': active_poisson_rate})
        nest.SetStatus(poisson_second_reward,{'rate': inactive_poisson_rate})
        print 'active CORTEX 1 + A D1 + B D2'
        nest.Simulate(simulation_intervals)
    if i%4 ==3:
        nest.SetStatus(poisson_2,{'rate': active_poisson_rate})
        nest.SetStatus(poisson_first_reward,{'rate': inactive_poisson_rate})
        nest.SetStatus(poisson_second_reward,{'rate': active_poisson_rate})
        print 'active CORTEX 2 + B D1 + A D2'
        nest.Simulate(simulation_intervals)
        
    
    print 'mem pot', nest.GetStatus(voltmeterA, 'events')[0].get('V_m')[len(nest.GetStatus(voltmeterA, 'events')[0].get('V_m'))-1]
    #### Recordings
    neuron_1a = nest.GetStatus(nest.FindConnections([cortical_input_1[8]], target = [A_D1_minicolumn[8]]))
    neuron_2a = nest.GetStatus(nest.FindConnections([cortical_input_1[8]], target = [B_D1_minicolumn[8]]))
    
    
    # D1 data collection
    w_A_D1[i] = [neuron_1a[0]['weight']][0]
    w_B_D1[i] = [neuron_2a[0]['weight']][0]
    b_A_D1[i] = [neuron_1a[0]['bias']][0]
    b_B_D1[i] = [neuron_2a[0]['bias']][0]
    
    # D2 data collection
    w_A_D2[i] = [neuron_1a[0]['weight']][0]
    w_B_D2[i] = [neuron_2a[0]['weight']][0]
    b_A_D2[i] = [neuron_1a[0]['bias']][0]
    b_B_D2[i] = [neuron_2a[0]['bias']][0]
    
    print 'PROBA:', np.exp(w_A_D1[i]+b_A_D1[i])
    
# Test if cortical stimulation only is now sufficient to elicit spike in the minicolumns    
print 'final phase'
nest.SetStatus(poisson_1,{'rate': inactive_poisson_rate})
nest.SetStatus(poisson_2,{'rate': inactive_poisson_rate})
nest.SetStatus(poisson_first_reward,{'rate': inactive_poisson_rate})
nest.SetStatus(poisson_second_reward,{'rate': inactive_poisson_rate})
nest.Simulate(simulation_intervals)
nest.SetStatus(poisson_1,{'rate': active_poisson_rate})
# D1 data collection
w_A_D1[number_of_cycles] = [neuron_1a[0]['weight']][0]
w_B_D1[number_of_cycles] = [neuron_2a[0]['weight']][0]
b_A_D1[number_of_cycles] = [neuron_1a[0]['bias']][0]
b_B_D1[number_of_cycles] = [neuron_2a[0]['bias']][0]

# D2 data collection
w_A_D2[number_of_cycles] = [neuron_1a[0]['weight']][0]
w_B_D2[number_of_cycles] = [neuron_2a[0]['weight']][0]
b_A_D2[number_of_cycles] = [neuron_1a[0]['bias']][0]
b_B_D2[number_of_cycles] = [neuron_2a[0]['bias']][0]
#nest.SetStatus(poisson_first_reward,{'rate': inactive_poisson_rate})
#nest.SetStatus(poisson_second_reward,{'rate': inactive_poisson_rate})
nest.Simulate( 2 * simulation_intervals )

# D1 data collection
w_A_D1[number_of_cycles+1] = [neuron_1a[0]['weight']][0]
w_B_D1[number_of_cycles+1] = [neuron_2a[0]['weight']][0]
b_A_D1[number_of_cycles+1] = [neuron_1a[0]['bias']][0]
b_B_D1[number_of_cycles+1] = [neuron_2a[0]['bias']][0]

# D2 data collection
w_A_D2[number_of_cycles+1] = [neuron_1a[0]['weight']][0]
w_B_D2[number_of_cycles+1] = [neuron_2a[0]['weight']][0]
b_A_D2[number_of_cycles+1] = [neuron_1a[0]['bias']][0]
b_B_D2[number_of_cycles+1] = [neuron_2a[0]['bias']][0]   
nest.SetStatus(poisson_1,{'rate': inactive_poisson_rate})
nest.SetStatus(poisson_2,{'rate': active_poisson_rate})
nest.Simulate( 2 * simulation_intervals )
# D1 data collection
w_A_D1[number_of_cycles+2] = [neuron_1a[0]['weight']][0]
w_B_D1[number_of_cycles+2] = [neuron_2a[0]['weight']][0]
b_A_D1[number_of_cycles+2] = [neuron_1a[0]['bias']][0]
b_B_D1[number_of_cycles+2] = [neuron_2a[0]['bias']][0]

# D2 data collection
w_A_D2[number_of_cycles+2] = [neuron_1a[0]['weight']][0]
w_B_D2[number_of_cycles+2] = [neuron_2a[0]['weight']][0]
b_A_D2[number_of_cycles+2] = [neuron_1a[0]['bias']][0]
b_B_D2[number_of_cycles+2] = [neuron_2a[0]['bias']][0]

nest.SetStatus(poisson_1,{'rate': inactive_poisson_rate})
nest.SetStatus(poisson_2,{'rate': inactive_poisson_rate})
nest.SetStatus(poisson_first_reward,{'rate': inactive_poisson_rate})
nest.SetStatus(poisson_second_reward,{'rate': inactive_poisson_rate})
nest.Simulate(simulation_intervals)
# D1 data collection
w_A_D1[number_of_cycles+3] = [neuron_1a[0]['weight']][0]
w_B_D1[number_of_cycles+3] = [neuron_2a[0]['weight']][0]
b_A_D1[number_of_cycles+3] = [neuron_1a[0]['bias']][0]
b_B_D1[number_of_cycles+3] = [neuron_2a[0]['bias']][0]

# D2 data collection
w_A_D2[number_of_cycles+3] = [neuron_1a[0]['weight']][0]
w_B_D2[number_of_cycles+3] = [neuron_2a[0]['weight']][0]
b_A_D2[number_of_cycles+3] = [neuron_1a[0]['bias']][0]
b_B_D2[number_of_cycles+3] = [neuron_2a[0]['bias']][0]
nest.SetStatus(poisson_1,{'rate': active_poisson_rate})
nest.Simulate(simulation_intervals)
# D1 data collection
w_A_D1[number_of_cycles+4] = [neuron_1a[0]['weight']][0]
w_B_D1[number_of_cycles+4] = [neuron_2a[0]['weight']][0]
b_A_D1[number_of_cycles+4] = [neuron_1a[0]['bias']][0]
b_B_D1[number_of_cycles+4] = [neuron_2a[0]['bias']][0]

# D2 data collection
w_A_D2[number_of_cycles+4] = [neuron_1a[0]['weight']][0]
w_B_D2[number_of_cycles+4] = [neuron_2a[0]['weight']][0]
b_A_D2[number_of_cycles+4] = [neuron_1a[0]['bias']][0]
b_B_D2[number_of_cycles+4] = [neuron_2a[0]['bias']][0]
nest.SetStatus(poisson_1,{'rate': inactive_poisson_rate})
nest.SetStatus(poisson_2,{'rate': active_poisson_rate})
nest.Simulate(simulation_intervals)
# D1 data collection
w_A_D1[number_of_cycles+5] = [neuron_1a[0]['weight']][0]
w_B_D1[number_of_cycles+5] = [neuron_2a[0]['weight']][0]
b_A_D1[number_of_cycles+5] = [neuron_1a[0]['bias']][0]
b_B_D1[number_of_cycles+5] = [neuron_2a[0]['bias']][0]

# D2 data collection
w_A_D2[number_of_cycles+5] = [neuron_1a[0]['weight']][0]
w_B_D2[number_of_cycles+5] = [neuron_2a[0]['weight']][0]
b_A_D2[number_of_cycles+5] = [neuron_1a[0]['bias']][0]
b_B_D2[number_of_cycles+5] = [neuron_2a[0]['bias']][0]

print 'simulation time', time.time()-start_time


################################################################################################################
##                                          FIGURES                                                           ##
################################################################################################################
pl.figure(99)
pl.rc("font", size=newfontsize)
pl.plot(w_A_D1, label="weights A D1")
pl.plot(w_B_D1, '-.', label="weights B D1")
pl.plot(w_A_D2, label="weights A D2")
pl.plot(w_B_D2, '-.', label="weights B D2")
pl.plot(b_A_D1, label="bias A D1")
pl.plot(b_B_D1, '-.', label="bias B D1")
pl.plot(b_A_D2, label="bias A D2")
pl.plot(b_B_D2, '-.', label="bias B D2")
pl.legend(loc='lower right', ncol=2, shadow=True)
pl.xlabel("time "+ r"$ms$")
pl.show()

pl.figure(88)
pl.rc("font", size=newfontsize)
offset = 0
addspike = 25   #artificially added voltage in order to display spikes in the membrane potential plot
listname_volt = ["State 1", "State 2", "reward A (AD1+BD2)", "reward B (AD2+BD1)","A D1", "B D1",  "action A", "action B"]
for voltm in [voltmeter_in1, voltmeter_in2,  voltmeter_rw1, voltmeter_rw2, voltmeter,voltmeter2, voltmeterA,  voltmeterB]:
    datavolt = nest.GetStatus(voltm)[0]
    vm = datavolt['events']['V_m']/2
    spikedetec = nest.GetStatus([nest.GetStatus(nest.FindConnections([datavolt['events']['senders'][0]]))[0]['target']])[0]
    for i in range(0,spikedetec['n_events']):
        if spikedetec['events']['senders'][i]==datavolt['events']['senders'][0]:
            #datavolt['events']['V_m'][int(spikedetec['events']['times'][i])] += addspike
            vm[int(spikedetec['events']['times'][i])] += addspike
    pl.plot(vm + 40*offset, label = listname_volt[offset])
    offset +=1
#pl.legend(loc='lower right')
ylabels = ('[neuron x]\nstate 1', '[neuron y]\nstate 2', '[neuron m]\nreward A D1 + B D2', '[neuron n]\nreward B D1 + A D2','[neuron i]\nSTR A D1', '[neuron j]\nSTR B D1', 'neuron\naction A', 'neuron\naction B')
ylocs = [-30,10,50,90,130,170,210, 250]
pl.yticks(ylocs, ylabels, fontsize=labels_fontsize)
pl.axis([ 0, nest.GetStatus(voltmeter_in1)[0]['n_events']+1, min(ylocs)-10, max(ylocs)+addspike+10 ])
frame88 = pl.gca()
frame88.xaxis.label.set_fontsize(labels_fontsize)
frame88.yaxis.label.set_fontsize(labels_fontsize)
#frame88.axes.get_yaxis().set_ticks([])
pl.xlabel("time "+ r"$ms$")
pl.show()




pl.figure(808)
pl.rc("font", size=newfontsize)
pl.imshow([nest.GetStatus(voltmeterB)[0]['events']['V_m'],nest.GetStatus(voltmeterA)[0]['events']['V_m'],nest.GetStatus(voltmeter2)[0]['events']['V_m'],nest.GetStatus(voltmeter)[0]['events']['V_m'],nest.GetStatus(voltmeter_rw2)[0]['events']['V_m'],nest.GetStatus(voltmeter_rw1)[0]['events']['V_m'], nest.GetStatus(voltmeter_in2)[0]['events']['V_m'],nest.GetStatus(voltmeter_in1)[0]['events']['V_m']],  aspect=6499/5, interpolation = 'nearest')
#pl.axis([ 0, 6500 , -1.0, 6.0 ])
#Order of the recording devices might be handset, to be improved
ylabels = ( 'neuron\naction B' ,'neuron\naction A','[neuron j]\nSTR B D1','[neuron i]\nSTR A D1','[neuron n]\nreward B D1 + A D2','[neuron m]\nreward A D1 + B D2','[neuron y]\nstate 2','[neuron x]\nstate 1')
ylocs = np.arange(len(ylabels))
pl.yticks(ylocs, ylabels)
frame808 = pl.gca()
frame808.xaxis.label.set_fontsize(labels_fontsize)
frame808.yaxis.label.set_fontsize(labels_fontsize)
#pl.ylabel(r"$y$",fontsize=labels_fontsize)
pl.xlabel("time "+ r"$ms$")
pl.colorbar()


pl.show()
#nest.voltage_trace.from_device(voltmeter_in1)   #blue
#nest.voltage_trace.from_device(voltmeter_in2)   #green
#nest.voltage_trace.from_device(voltmeter)       #red
#nest.voltage_trace.from_device(voltmeter2)      #cyan
#nest.voltage_trace.from_device(voltmeterA)      #purple
#nest.voltage_trace.from_device(voltmeterB)      #yellow
#nest.voltage_trace.pylab.legend(loc='lower right')
#nest.voltage_trace.show()
#nest.raster_plot.from_device(detect_b, hist=True)
#nest.raster_plot.from_device(detect_a, hist=True)
#nest.raster_plot.show()
#nest.raster_plot.from_device(first_detect, hist=True)
#nest.raster_plot.from_device(second_detect, hist=True)
#nest.raster_plot.from_device(detect_a, hist=True)
#nest.raster_plot.from_device(detect_b, hist=True)
#nest.raster_plot.from_device(detect_c, hist=True)
#nest.raster_plot.from_device(detect_d, hist=True)
#
#nest.raster_plot.show()

# ensure weights are updated
#pl.figure(1)
#pl.plot([a['weight'] for a in nest.GetStatus(nest.FindConnections(A_D1_minicolumn, target = B_D1_minicolumn))], 'r')
##pl.plot([a['weight'] for a in nest.GetStatus(nest.FindConnections(A_D1_minicolumn, target = second_neuron))], 'b')
#pl.plot([a['weight'] for a in nest.GetStatus(nest.FindConnections(B_D1_minicolumn, target = A_D1_minicolumn))], 'b')
##pl.plot([a['weight'] for a in nest.GetStatus(nest.FindConnections(B_D1_minicolumn, target = second_neuron))], 'r')
#pl.show()
#
## Get results from recall simulation - has bayes inference been successful?
#result_vec_first = []
#result_vec_second = []
#prev_time = 0
#intervals = 1000 # 1000
#for calc_rates in range(0, total_simulation_time):
#    result_vec_first.append( len([a for a in nest.GetStatus(first_detect)[0]['events']['times'] if (a > (prev_time) and a < calc_rates)]) * (1000.0 / (intervals )))
#    result_vec_second.append( len([a for a in nest.GetStatus(second_detect)[0]['events']['times'] if (a > (prev_time) and a < calc_rates)]) * (1000.0 / (intervals )))
#    prev_time = calc_rates
#
#pl.figure(5)
#pl.plot(result_vec_first)
#pl.plot(result_vec_second)
#pl.xlabel('time [s]')
#pl.ylabel('rate [Hz]')
#pl.show()
pl.figure(51)
#all = len(nest.GetStatus(voltmeter_in2)[0]['events']['V_m'])
pl.rc("font", size=newfontsize)
z = 0
color = ['b','g', 'r', 'c', 'm', 'y', 'k']
for a in [detect_a, detect_b, detect_c, detect_d,detect_e, detect_f, first_detect, second_detect]:
    cl = color[z%len(color)]
    pl.scatter(nest.GetStatus(a)[0]['events']['times'], nest.GetStatus(a)[0]['events']['senders'], c=cl, marker='.')
    z+=1
pl.axis([ 0, nest.GetStatus(voltmeter_in1)[0]['n_events'], 0, 300 ])
frame51 = pl.gca()
frame51.xaxis.label.set_fontsize(labels_fontsize)
frame51.yaxis.label.set_fontsize(labels_fontsize)
pl.xlabel("time "+ r"$ms$")
pl.show()

