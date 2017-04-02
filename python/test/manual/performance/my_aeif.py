# Create by Mikael Lindahl on 3/12/17.

import nest
import pprint
import time
import util
import os
# import os
pp=pprint.pprint

if nest.version()=='NEST 2.2.2':
    os.environ['LD_LIBRARY_PATH']='/home/mikael/git/bgmodel/nest/dist/install/nest-2.2.2/lib/nest'

nest.Install('ml_module')

def getPramsSTN():
    dic = {}

    dic['tau_w'] = 333.0  # I-V relation, spike frequency adaptation
    dic['a_1'] = 0.3  # I-V relation
    dic['a_2'] = 0.0  # I-V relation
    dic['b'] = 0.05  # 0.1 #0.1#200./5.
    dic['C_m'] = 60.0  # t_m/R_in
    dic['Delta_T'] = 16.2
    dic['g_L'] = 10.0
    dic['E_L'] = -80.2
    dic['I_e'] = 6.0
    dic['V_peak'] = 15.0
    dic['V_reset'] = -70.0  # I-V relation
    dic['V_a'] = -64.0
    dic['V_th'] = -64.0

    dic['AMPA_1_Tau_decay']=3.0

    return dic

par={
    'period_first':500.0,
    'period_second':500.0,
    'rate_first':500.,
    'rate_second':250.
}

df =nest.GetDefaults("my_aeif_cond_exp")
pp(df)

nest.CopyModel("my_aeif_cond_exp","STN",
               getPramsSTN())

rec=nest.GetDefaults("my_aeif_cond_exp")["receptor_types"]
pp(rec)
n=nest.Create(**{
    'model':'poisson_generator_periodic',
    'n':1000,
    'params':par})


# n=nest.Create(**{
#     'model':'poisson_generator',
#     'n':1000,
#     'params':{'rate':900.}})

m=nest.Create('STN', 1000)
# m=nest.Create('my_aeif_cond_exp', 100)
# m=nest.Create('izhik_cond_exp', 100)
#

multimeter = nest.Create("multimeter")
nest.SetStatus(multimeter, {"withtime":True, "record_from":["V_m"]})

spikedetector = nest.Create("spike_detector",
                params={"withgid": True, "withtime": True})

nest.CopyModel("static_synapse", "excitatory",
               {
                   "weight": 0.25,
                   "delay": 0.5,
                   "receptor_type": rec["AMPA_1"]
                   # "receptor_type": 0
               })

nest.Connect(multimeter, [m[0]])

if nest.version()=='NEST 2.12.0':
    syn_dict = {
        "model": 'excitatory',
    }
    conn_dict = {
        "rule": "one_to_one",
    }

    nest.Connect(n, m, conn_dict, syn_dict)
    nest.Connect(m, spikedetector)
    # print 'Connecting ' + ' my_nest.GetConnections ', len(nest.GetConnections(n)), len(n)

if nest.version()=='NEST 2.2.2':

    nest.Connect(n, m, **{
        "model":"excitatory"
    })

    nest.ConvergentConnect(m, spikedetector)

with util.Stopwatch('Speed test'):
    nest.SetKernelStatus({'print_time':True})
    nest.Simulate(1000)

util.show(multimeter, spikedetector)
