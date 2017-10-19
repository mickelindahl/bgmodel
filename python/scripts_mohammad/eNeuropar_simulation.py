''' Created by mmohaghegh on 10.04.17
This .py file contains the parameters used for simulating eNeuro_fig_01_and_02_sim_sw.py
This will help me understand what is going on in bgmodel. Therefore, it is necessary in
the first step to have one sample parameter set with which the sw simulation can be run.
'''

# import pickle
import nest
import nest.raster_plot as raster
'''
import matplotlib
matplotlib.use('Agg')
'''
import matplotlib.pyplot as plt
import os

res_dir = os.getenv('BG_MODEL_PYTHON') + '/results-tr-sims/'

if not os.path.isdir(res_dir):
    os.mkdir(res_dir)



nest.SetKernelStatus({'local_num_threads':4})
nest.Install('ml_module')


from core.network.parameters.eneuro import EneuroPar

eneuro = EneuroPar()
eneuro.set({'simu':{'sim_stop':4000.}
            })
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
                                         'origin':CF_start,
                                         'stop':CF_stop})

# Generating cortical inputs to MSN D1 (NMDA)

# C1_n = params_dic['node']['C1']['n']
C1_n = 1
C1_model = params_dic['nest']['C1']['type_id']
C1_rate = params_dic['node']['C1']['rate']
C1_start = 1.0
C1_stop = params_dic['node']['C1']['spike_setup'][0]['t_stop']

C1N = nest.Create(C1_model, C1_n, params = {'rate':C1_rate,
                                         # 'origin':C1_start,
                                         'stop':C1_stop})

# Generating cortical inputs to MSN D2 (NMDA)

# C1_n = params_dic['node']['C1']['n']
C2_n = 1
C2_model = params_dic['nest']['C2']['type_id']
C2_rate = params_dic['node']['C2']['rate']
C2_start = 1.0
C2_stop = params_dic['node']['C2']['spike_setup'][0]['t_stop']

C2N = nest.Create(C2_model, C2_n, params = {'rate':C2_rate,
                                         # 'origin':C1_start,
                                         'stop':C2_stop})

# Generating cortical inputs to FSI (NMDA)

# C1_n = params_dic['node']['C1']['n']
CF_n = 1
CF_model = params_dic['nest']['CF']['type_id']
CF_rate = params_dic['node']['CF']['rate']
CF_start = 1.0
CF_stop = params_dic['node']['CF']['spike_setup'][0]['t_stop']

CFN = nest.Create(CF_model, CF_n, params = {'rate':CF_rate,
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

# Generating cortical inputs to STN (NMDA)

# C1_n = params_dic['node']['C1']['n']
CS_n = 1
CS_model = params_dic['nest']['CS']['type_id']
CS_rate = params_dic['node']['CS']['rate']
CS_start = 1.0
CS_stop = params_dic['node']['CS']['spike_setup'][0]['t_stop']

CSN = nest.Create(CS_model, CS_n, params = {'rate':CS_rate,
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

# Generating external excitatory inputs to GP Proto

# C1_n = params_dic['node']['C1']['n']
EF_n = 1
EF_model = params_dic['nest']['EF']['type_id']
EF_rate = params_dic['node']['EF']['rate']
EF_start = 1.0
EF_stop = params_dic['node']['EF']['spike_setup'][0]['t_stop']

EF = nest.Create(EF_model, EF_n, params = {'rate':EF_rate,
                                         # 'origin':C1_start,
                                         'stop':EF_stop})

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

# Creating neuron objects for simulations GP projecting to FS?

GF_n = params_dic['node']['GF']['n']
GF_model = params_dic['nest']['GF']['type_id']
GF_params = params_dic['nest']['GF']
GF_params.pop('type_id')

nest.CopyModel(GF_model, 'GPFS', params = GF_params)

GF = nest.Create('GPFS', GF_n)

# Connecting populations together

# Connecting cortical input to MSN D1 (AMPA)

conn_name = 'C1_M1_ampa'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)

    nest.Connect(C1, M1, syn_spec=syn)

# Connecting cortical input to MSN D1 (NMDA)

conn_name = 'C1_M1_nmda'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(C1N, M1, syn_spec=syn)

# Connecting cortical input to MSN D2 (AMPA)

conn_name = 'C2_M2_ampa'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(C2, M2, syn_spec=syn)

# Connecting cortical input to MSN D2 (NMDA)

conn_name = 'C2_M2_nmda'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(C2N, M2, syn_spec=syn)

# Connecting cortical input to FSI

conn_name = 'CF_FS_ampa'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(CF, FS, syn_spec=syn)

# Connecting cortical input to STN (AMPA)

conn_name = 'CS_ST_ampa'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(CS, ST, syn_spec=syn)

# Connecting cortical input to STN (NMDA)

conn_name = 'CS_ST_nmda'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(CSN, ST, syn_spec=syn)

# Connecting external input to SNr

conn_name = 'ES_SN_ampa'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(ES, SN, syn_spec=syn)

# Connecting external input to GP Arky

conn_name = 'EA_GA_ampa'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(EA, GA, syn_spec=syn)

# Connecting external input to GP Proto

conn_name = 'EI_GI_ampa'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(EI, GI, syn_spec=syn)

# Connecting external input to GP-projecting FSI

conn_name = 'EF_GF_ampa'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(EF, GF, syn_spec=syn)


# Connecting MSN D1 to MSN D1

conn_name = 'M1_M1_gaba'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(M1, M1, conn_spec=conn, syn_spec=syn)

# Connecting MSN D1 to MSN D2

conn_name = 'M1_M2_gaba'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(M1, M2, conn_spec=conn, syn_spec=syn)

# Connecting MSN D2 to MSN D2

conn_name = 'M2_M2_gaba'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(M2, M2, conn_spec=conn, syn_spec=syn)

# Connecting MSN D2 to MSN D1

conn_name = 'M2_M1_gaba'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(M2, M1, conn_spec=conn, syn_spec=syn)

# Connecting FSI to MSN D1

conn_name = 'FS_M1_gaba'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(FS, M1, conn_spec=conn, syn_spec=syn)

# Connecting FSI to MSN D2

conn_name = 'FS_M2_gaba'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(FS, M2, conn_spec=conn, syn_spec=syn)

# Connecting FSI to FSI

conn_name = 'FS_FS_gaba'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(FS, FS, conn_spec=conn, syn_spec=syn)

# Connecting MSN D1 to SNr

conn_name = 'M1_SN_gaba'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(M1, SN, conn_spec=conn, syn_spec=syn)

# Connecting MSN D2 to GP Arky

conn_name = 'M2_GA_gaba'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(M2, GA, conn_spec=conn, syn_spec=syn)

# Connecting GP Arky to MSN D2

conn_name = 'GA_M2_gaba'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(GA, M2, conn_spec=conn, syn_spec=syn)

# Connecting GP Arky to MSN D1

conn_name = 'GA_M1_gaba'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(GA, M1, conn_spec=conn, syn_spec=syn)

# Connecting MSN D2 to GP Proto

conn_name = 'M2_GI_gaba'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(M2, GI, conn_spec=conn, syn_spec=syn)

# Connecting GP Proto to MSN D2

conn_name = 'GI_M2_gaba'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(GI, M2, conn_spec=conn, syn_spec=syn)

# Connecting GP Proto to MSN D1

conn_name = 'GI_M1_gaba'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(GI, M1, conn_spec=conn, syn_spec=syn)

# Connecting MSN D2 to GP FSI projecting

conn_name = 'M2_GF_gaba'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(M2, GF, conn_spec=conn, syn_spec=syn)

# Connecting GP FSI projecting to MSN D2

conn_name = 'GF_M2_gaba'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(GF, M2, conn_spec=conn, syn_spec=syn)

# Connecting GP FSI projecting to MSN D1

conn_name = 'GF_M1_gaba'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(GF, M1, conn_spec=conn, syn_spec=syn)

# Connecting GP special neurons to FSI projecting. Maybe they are not really special but rather are add to tailor the
# need for reviewers

conn_name = 'GF_FS_gaba'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(GF, FS, conn_spec=conn, syn_spec=syn)

# Connecting GPe Proto to FSI

conn_name = 'GI_FS_gaba'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(GI, FS, conn_spec=conn, syn_spec=syn)

# Connecting GPe Arky to FSI

conn_name = 'GA_FS_gaba'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(GA, FS, conn_spec=conn, syn_spec=syn)

# Connecting GP special neurons to GP Proto.

conn_name = 'GF_GI_gaba'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(GF, GI, conn_spec=conn, syn_spec=syn)

# Connecting GP Proto to GP special neurons.

conn_name = 'GI_GF_gaba'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(GI, GF, conn_spec=conn, syn_spec=syn)

# Connecting GP special neurons to GP Arky.

conn_name = 'GF_GA_gaba'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(GF, GA, conn_spec=conn, syn_spec=syn)

# Connecting GP Arky to GP special neurons.

conn_name = 'GA_GF_gaba'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(GA, GF, conn_spec=conn, syn_spec=syn)

# Connecting GP special neurons to themselves.

conn_name = 'GF_GF_gaba'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(GF, GF, conn_spec=conn, syn_spec=syn)

# Connecting GP Arky to themselves.

conn_name = 'GA_GA_gaba'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(GA, GA, conn_spec=conn, syn_spec=syn)

# Connecting GP Proto to themselves.

conn_name = 'GI_GI_gaba'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(GI, GI, conn_spec=conn, syn_spec=syn)

# Connecting GP Proto to GP Arky.

conn_name = 'GI_GA_gaba'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(GI, GA, conn_spec=conn, syn_spec=syn)

# Connecting GP Arky to GP Proto.

conn_name = 'GA_GI_gaba'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(GA, GI, conn_spec=conn, syn_spec=syn)

# Connecting GP Proto to STN.

conn_name = 'GI_ST_gaba'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(GI, ST, conn_spec=conn, syn_spec=syn)

# Connecting STN to GP Proto.

conn_name = 'ST_GI_ampa'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(ST, GI, conn_spec=conn, syn_spec=syn)

# Connecting GP FS to STN.

conn_name = 'GF_ST_gaba'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(GF, ST, conn_spec=conn, syn_spec=syn)

# Connecting STN to GP FS.

conn_name = 'ST_GF_ampa'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(ST, GF, conn_spec=conn, syn_spec=syn)

# Connecting STN to GP Arky

conn_name = 'ST_GA_ampa'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(ST, GA, conn_spec=conn, syn_spec=syn)

# Connecting STN to STN

conn_name = 'ST_ST_ampa'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(ST, ST, conn_spec=conn, syn_spec=syn)

# Connecting STN to SNr

conn_name = 'ST_SN_ampa'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(ST, SN, conn_spec=conn, syn_spec=syn)

# Connecting GP FS to SNr

conn_name = 'GF_SN_gaba'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(GF, SN, conn_spec=conn, syn_spec=syn)

# Connecting GP Proto to SNr

conn_name = 'GI_SN_gaba'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']
indeg = params_dic['conn'][conn_name]['fan_in'].as_integer_ratio()[0]
conn = {'rule': 'fixed_indegree', 'indegree':indeg}
isconn = params_dic['conn'][conn_name]['lesion']

if not isconn:
    nest.CopyModel(synmodel, syn, syndic)
    nest.Connect(GI, SN, conn_spec=conn, syn_spec=syn)

# Connecting spike detectors to the populations

SN_spks = nest.Create('spike_detector',params={'start':1000.})
nest.Connect(SN,SN_spks)

ST_spks = nest.Create('spike_detector',params={'start':1000.})
nest.Connect(ST,ST_spks)

M1_spks = nest.Create('spike_detector',params={'start':1000.})
nest.Connect(M1,M1_spks)

M2_spks = nest.Create('spike_detector',params={'start':1000.})
nest.Connect(M2,M2_spks)

FS_spks = nest.Create('spike_detector',params={'start':1000.})
nest.Connect(FS,FS_spks)

GA_spks = nest.Create('spike_detector',params={'start':1000.})
nest.Connect(GA,GA_spks)

GI_spks = nest.Create('spike_detector',params={'start':1000.})
nest.Connect(GI,GI_spks)

GF_spks = nest.Create('spike_detector',params={'start':1000.})
nest.Connect(GF,GF_spks)

nest.Simulate(5000.)

raster.from_device(SN_spks,hist=True)
#plt.savefig(res_dir+'SNr.svg',format='svg')
plt.savefig(res_dir+'SNr.pdf',format='pdf')

raster.from_device(ST_spks,hist=True)
#plt.savefig(res_dir+'STN.svg',format='svg')
plt.savefig(res_dir+'STN.pdf',format='pdf')

raster.from_device(M1_spks,hist=True)
plt.savefig(res_dir+'MSND1.pdf',format='pdf')

#raster.from_device(M2_spks,hist=True)
#plt.savefig(res_dir+'MSND2.pdf',format='pdf')


raster.from_device(FS_spks,hist=True)
plt.savefig(res_dir+'FSI.pdf',format='pdf')

raster.from_device(GA_spks,hist=True)
#plt.savefig(res_dir+'GPeArky.svg',format='svg')
plt.savefig(res_dir+'GPeArky.pdf',format='pdf')

raster.from_device(GI_spks,hist=True)
#plt.savefig(res_dir+'GPeProto.svg',format='svg')
plt.savefig(res_dir+'GPeProto.pdf',format='pdf')

raster.from_device(GF_spks,hist=True)
#plt.savefig(res_dir+'GPeFSI.svg',format='svg')
plt.savefig(res_dir+'GPeFSI.pdf',format='pdf')


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
