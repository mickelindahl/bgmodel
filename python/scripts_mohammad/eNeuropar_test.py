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
eneuro.set({'simu':{'sim_stop':4000.},
            'node':{'CF':{'rate':700.}}
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

# Creating neuron objects for simulations MSN D1

M1_n = params_dic['node']['M1']['n']
M1_model = params_dic['nest']['M1']['type_id']
M1_params = params_dic['nest']['M1']
M1_params.pop('type_id')

nest.CopyModel(M1_model, 'MSND1', params = M1_params)

M1 = nest.Create('MSND1',M1_n)

conn_name = 'C1_M1_ampa'
syndic = params_dic['nest'][conn_name]
synmodel = syndic.pop('type_id')
syn = params_dic['conn'][conn_name]['syn']

nest.CopyModel(synmodel, syn, syndic)

nest.Connect(C1, M1, syn_spec=syn)

M1_spks = nest.Create('spike_detector')
nest.Connect(M1,M1_spks)

nest.Simulate(5000.)

raster.from_device(M1_spks,hist=True)
plt.savefig(res_dir+'MSND1.pdf',format='pdf')

plt.show()
