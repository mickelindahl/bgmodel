''' Created by mmohaghegh on 10.04.17
This .py file contains the parameters used for simulating eNeuro_fig_01_and_02_sim_sw.py
This will help me understand what is going on in bgmodel. Therefore, it is necessary in
the first step to have one sample parameter set with which the sw simulation can be run.
'''

import pickle
import nest

data_dir = '/Users/Mohammad/Documents/PhD/Projects/BGmodel/bgmodel/python/scripts_mohammad/temp-data/'

# Reading data to variables

with open(data_dir+'nestpar.pickle','rb') as hd1:
    params_nest = pickle.load(hd1)
hd1.close()

with open(data_dir+'poppar.pickle','rb') as hd2:
    params_pop = pickle.load(hd2)
hd1.close()

# Other parameters which might not be in the variables above

# Generating parrot neurons and a dynamic poisson generator for CX to MSN D1
C1_n = params_pop['C1']['n']
C1_model = 'parrot_neuron'
C1_params = params_pop['C1']['params']

PC1_n = 1
PC1_model = 'poisson_generator_dynamic'
PC1_times = params_pop['C1']['spike_setup'][0]['times']
PC1_rates = params_pop['C1']['spike_setup'][0]['rates']

# Generating parrot neurons and a dynamic poisson generator for CX to MSN D2
C2_n = params_pop['C2']['n']
C2_model = 'parrot_neuron'
C2_params = params_pop['C2']['params']

PC2_n = 1
PC2_model = 'poisson_generator_dynamic'
PC2_times = params_pop['C2']['spike_setup'][0]['times']
PC2_rates = params_pop['C2']['spike_setup'][0]['rates']

# Generating parrot neurons and a dynamic poisson generator for CX to FSI
CF_n = params_pop['CF']['n']
CF_model = 'parrot_neuron'
CF_params = params_pop['CF']['params']

PCF_n = 1
PCF_model = 'poisson_generator_dynamic'
PCF_times = params_pop['CF']['spike_setup'][0]['times']
PCF_rates = params_pop['CF']['spike_setup'][0]['rates']

# Generating parrot neurons and a dynamic poisson generator for CX to STN
CS_n = params_pop['CS']['n']
CS_model = 'parrot_neuron'
CS_params = params_pop['CS']['params']

PCS_n = 1
PCS_model = 'poisson_generator_dynamic'
PCS_times = params_pop['CS']['spike_setup'][0]['times']
PCS_rates = params_pop['CS']['spike_setup'][0]['rates']

# Generating parrot neurons and a poisson generator for external input to GP arky
EA_n = params_pop['EA']['n']
EA_model = 'parrot_neuron'
EA_params = params_pop['EA']['params']

PEA_n = 1
PEA_model = 'poisson_generator'
PEA_tstop = params_pop['EA']['spike_setup'][0]['t_stop']
PEA_tstart = params_pop['EA']['spike_setup'][0]['times'][0]
PEA_rates = params_pop['EA']['spike_setup'][0]['rates'][0]

# Generating parrot neurons and a poisson generator for external input to FSI??
EF_n = params_pop['EF']['n']
EF_model = 'parrot_neuron'
EF_params = params_pop['EF']['params']

PEF_n = 1
PEF_model = 'poisson_generator'
PEF_tstop = params_pop['EF']['spike_setup'][0]['t_stop']
PEF_tstart = params_pop['EF']['spike_setup'][0]['times'][0]
PEF_rates = params_pop['EF']['spike_setup'][0]['rates'][0]

# Generating parrot neurons and a poisson generator for external input to GP Proto
EI_n = params_pop['EI']['n']
EI_model = 'parrot_neuron'
EI_params = params_pop['EI']['params']

PEI_n = 1
PEI_model = 'poisson_generator'
PEI_tstop = params_pop['EI']['spike_setup'][0]['t_stop']
PEI_tstart = params_pop['EI']['spike_setup'][0]['times'][0]
PEI_rates = params_pop['EI']['spike_setup'][0]['rates'][0]

# Creating FSI neurons
FS_n = params_pop['FS']['n']


## Nest building networks

# Installing ml_module
nest.Install('ml_module')

# Creating cortical inputs to MSN D1s
C1 = nest.Create(C1_model,C1_n,C1_params)
PC1 = nest.Create(PC1_model,PC1_n,{'timings':PC1_times,'rates':PC1_rates})
nest.Connect(PC1,C1)

# Creating cortical inputs to MSN D2s
C2 = nest.Create(C2_model,C2_n,C2_params)
PC2 = nest.Create(PC2_model,PC2_n,{'timings':PC1_times,'rates':PC1_rates})
nest.Connect(PC1,C1)

# Creating cortical inputs to FSIs
CF = nest.Create(CF_model,CF_n,CF_params)
PCF = nest.Create(PCF_model,PCF_n,{'timings':PCF_times,'rates':PCF_rates})
nest.Connect(PCF,CF)

# Creating cortical inputs to STNs
CS = nest.Create(CS_model,CS_n,CS_params)
PCS = nest.Create(PCS_model,PCS_n,{'timings':PCS_times,'rates':PCS_rates})
nest.Connect(PCS,CS)

# Creating external inputs to GP Arky
EA = nest.Create(EA_model,EA_n,EA_params)
PEA = nest.Create(PEA_model,PEA_n,{'rate':PEA_rates,'start':PEA_tstart,'stop':PEA_tstop})
nest.Connect(PEA,EA)

# Creating external inputs to FSI ???
EF = nest.Create(EF_model,EF_n,EF_params)
PEF = nest.Create(PEF_model,PEF_n,{'rate':PEF_rates,'start':PEF_tstart,'stop':PEF_tstop})
nest.Connect(PEF,EF)

# Creating external inputs to GP Proto
EI = nest.Create(EI_model,EI_n,EI_params)
PEI = nest.Create(PEI_model,PEI_n,{'rate':PEI_rates,'start':PEI_tstart,'stop':PEI_tstop})
nest.Connect(PEI,EI)
