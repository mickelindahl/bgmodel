''' Created by mmohaghegh on 10.04.17
This .py file contains the parameters used for simulating eNeuro_fig_01_and_02_sim_sw.py
This will help me understand what is going on in bgmodel. Therefore, it is necessary in
the first step to have one sample parameter set with which the sw simulation can be run.
'''

# import pickle
import nest

nest.Install('ml_module')


from core.network.parameters.eneuro import EneuroPar

eneuro = EneuroPar()
params_dic = eneuro.dic
# print params_dic

'''
data_dir = '/Users/Mohammad/Documents/PhD/Projects/BGmodel/bgmodel/python/scripts_mohammad/temp-data/'

# Reading data to variables

with open(data_dir+'nestpar.pickle','rb') as hd1:
    params_nest = pickle.load(hd1)
hd1.close()

with open(data_dir+'poppar.pickle','rb') as hd2:
    params_pop = pickle.load(hd2)
hd1.close()
'''

# Generating cortical inputs to MSN D1

# C1_n = params_dic['node']['C1']['n']
C1_n = 1
C1_model = params_dic['nest']['C1']['type_id']
C1_rate = params_dic['node']['C1']['rate']
C1_start = 1.0
C1_stop = params_dic['node']['C1']['spike_setup'][0]['t_stop']

C1 = nest.Create(C1_model, C1_n, params = {'rate':C1_rate,
                                         # 'origin':C1_start,
                                         'stop':C1_stop})

# Generating cortical inputs to MSN D2

# C1_n = params_dic['node']['C1']['n']
C2_n = 1
C2_model = params_dic['nest']['C2']['type_id']
C2_rate = params_dic['node']['C2']['rate']
C2_start = 1.0
C2_stop = params_dic['node']['C2']['spike_setup'][0]['t_stop']

C2 = nest.Create(C2_model, C2_n, params = {'rate':C2_rate,
                                         # 'origin':C1_start,
                                         'stop':C2_stop})

# Generating cortical inputs to FSI

# C1_n = params_dic['node']['C1']['n']
CF_n = 1
CF_model = params_dic['nest']['CF']['type_id']
CF_rate = params_dic['node']['CF']['rate']
CF_start = 1.0
CF_stop = params_dic['node']['CF']['spike_setup'][0]['t_stop']

CF = nest.Create(CF_model, CF_n, params = {'rate':CF_rate,
                                         # 'origin':C1_start,
                                         'stop':CF_stop})

# Generating cortical inputs to STN

# C1_n = params_dic['node']['C1']['n']
CS_n = 1
CS_model = params_dic['nest']['CS']['type_id']
CS_rate = params_dic['node']['CS']['rate']
CS_start = 1.0
CS_stop = params_dic['node']['CS']['spike_setup'][0]['t_stop']

CS = nest.Create(CS_model, CS_n, params = {'rate':CS_rate,
                                         # 'origin':C1_start,
                                         'stop':CS_stop})

# Generating external excitatory inputs to GP Arkypallidal

# C1_n = params_dic['node']['C1']['n']
EA_n = 1
EA_model = params_dic['nest']['EA']['type_id']
EA_rate = params_dic['node']['EA']['rate']
EA_start = 1.0
EA_stop = params_dic['node']['EA']['spike_setup'][0]['t_stop']

EA = nest.Create(EA_model, EA_n, params = {'rate':EA_rate,
                                         # 'origin':C1_start,
                                         'stop':EA_stop})

# Generating external excitatory inputs to GP Proto

# C1_n = params_dic['node']['C1']['n']
EI_n = 1
EI_model = params_dic['nest']['EI']['type_id']
EI_rate = params_dic['node']['EI']['rate']
EI_start = 1.0
EI_stop = params_dic['node']['EI']['spike_setup'][0]['t_stop']

EI = nest.Create(EI_model, EI_n, params = {'rate':EI_rate,
                                         # 'origin':C1_start,
                                         'stop':EI_stop})

# Generating external excitatory inputs to SNr

# C1_n = params_dic['node']['C1']['n']
ES_n = 1
ES_model = params_dic['nest']['ES']['type_id']
ES_rate = params_dic['node']['ES']['rate']
ES_start = 1.0
ES_stop = params_dic['node']['ES']['spike_setup'][0]['t_stop']

ES = nest.Create(ES_model, ES_n, params = {'rate':ES_rate,
                                         # 'origin':C1_start,
                                         'stop':ES_stop})

# Creating neuron objects for simulations MSN D1

M1_n = params_dic['node']['M1']['n']
M1_model = params_dic['nest']['M1']['type_id']
M1_params = params_dic['nest']['M1']
M1_params.pop('type_id')

nest.CopyModel(M1_model, 'MSND1', params = M1_params)

M1 = nest.Create('MSND1',M1_n)

# Creating neuron objects for simulations MSN D2

M2_n = params_dic['node']['M2']['n']
M2_model = params_dic['nest']['M2']['type_id']
M2_params = params_dic['nest']['M2']
M2_params.pop('type_id')

nest.CopyModel(M2_model, 'MSND2', params = M2_params)

M2 = nest.Create('MSND2', M2_n)

# Creating neuron objects for simulations FSI

FS_n = params_dic['node']['FS']['n']
FS_model = params_dic['nest']['FS']['type_id']
FS_params = params_dic['nest']['FS']
FS_params.pop('type_id')
FS_params.update({'GABAA_2_Tau_decay':FS_params['GABAA_2_Tau_decay']*1.0})

nest.CopyModel(FS_model, 'FSI', params = FS_params)

FS = nest.Create('FSI', FS_n)

# Creating neuron objects for simulations STN

ST_n = params_dic['node']['ST']['n']
ST_model = params_dic['nest']['ST']['type_id']
ST_params = params_dic['nest']['ST']
ST_params.pop('type_id')

nest.CopyModel(ST_model, 'STN', params = ST_params)

ST = nest.Create('STN', ST_n)

# Creating neuron objects for simulations GPe Arkypallidal

GA_n = params_dic['node']['GA']['n']
GA_model = params_dic['nest']['GA']['type_id']
GA_params = params_dic['nest']['GA']
GA_params.pop('type_id')

nest.CopyModel(GA_model, 'GPArky', params = GA_params)

GA = nest.Create('GPArky', GA_n)

# Creating neuron objects for simulations GPe Proto

GI_n = params_dic['node']['GI']['n']
GI_model = params_dic['nest']['GI']['type_id']
GI_params = params_dic['nest']['GI']
GI_params.pop('type_id')

nest.CopyModel(GI_model, 'GPProto', params = GI_params)

GI = nest.Create('GPProto', GI_n)

# Creating neuron objects for simulations SNr

SN_n = params_dic['node']['SN']['n']
SN_model = params_dic['nest']['SN']['type_id']
SN_params = params_dic['nest']['SN']
SN_params.pop('type_id')

nest.CopyModel(SN_model, 'SNr', params = SN_params)

SN = nest.Create('SNr', SN_n)

# Creating neuron objects for simulations GP FSI?

GF_n = params_dic['node']['GF']['n']
GF_model = params_dic['nest']['GF']['type_id']
GF_params = params_dic['nest']['GF']
GF_params.pop('type_id')

nest.CopyModel(GF_model, 'GPFS', params = GF_params)

GF = nest.Create('GPFS', GF_n)

# Connecting populations together

# Connecting cortical input to MSN D1

synmodel = params_dic['nest']['C1_M1_ampa']['type_id']
weight = params_dic['nest']['C1_M1_ampa']['weight']
delay = params_dic['nest']['C1_M1_ampa']['delay']
rule = 'all-to-all'
syn = params_dic['conn']['C1_M1_ampa']['syn']

nest.CopyModel(synmodel,syn,{'delay':delay,
                          'weight':weight})

nest.Connect(C1, M1, syn_spec=syn)

'''
# Generating parrot neurons and a dynamic poisson generator for CX to MSN D1

C1_n = params_dic[]['C1']['n']
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
nest.Connect(PEI,EI)'''
