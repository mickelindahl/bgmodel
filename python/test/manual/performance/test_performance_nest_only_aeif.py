# Create by Mikael Lindahl on 3/12/17.

import nest
import pprint
import time
# import os
pp=pprint.pprint

# os.environ]'LD'
#
nest.Install('ml_module')

class Stopwatch():
    def __init__(self, *args, **kwargs):
        self.msg = args[0]
        self.args = args
        self.time = None

    def __enter__(self):

        self.time = time.time()
        print self.msg,


    def __exit__(self, type, value, traceback):
        t = round(time.time() - self.time, )
        msg_out = '... finnish {} {} sec '.format(self.msg, t)
        print msg_out

def getPramsSTN():
    dic = {}

    dic['tau_w'] = 333.0  # I-V relation, spike frequency adaptation
    dic['a'] = 0.3  # I-V relation
    dic['b'] = 0.05  # 0.1 #0.1#200./5.
    dic['C_m'] = 60.0  # t_m/R_in
    dic['Delta_T'] = 16.2
    dic['g_L'] = 10.0
    dic['E_L'] = -80.2
    dic['I_e'] = 6.0
    dic['V_peak'] = 15.0
    dic['V_reset'] = -70.0  # I-V relation
    dic['V_th'] = -64.0

    return dic

par={
    'period_first':500.0,
    'period_second':500.0,
    'rate_first':1000.,
    'rate_second':800.
}

df =nest.GetDefaults("aeif_cond_exp")
# pp(df)

nest.CopyModel("aeif_cond_exp","STN",
               getPramsSTN())

# rec=nest.GetDefaults("my_aeif_cond_exp")["receptor_types"]
# pp(rec)
n=nest.Create(**{
    'model':'poisson_generator_periodic',
    'n':1000,
    'params':par})


# n=nest.Create(**{
#     'model':'poisson_generator',
#     'n':1000,
#     'params':{'rate':900.}})

m=nest.Create('aeif_cond_exp', 1000)
# m=nest.Create('my_aeif_cond_exp', 100)
# m=nest.Create('izhik_cond_exp', 100)
#
nest.CopyModel("static_synapse","excitatory",
               {
                "weight":2.5,
                "delay":0.5,
                # "receptor_type": rec["AMPA_1"]
                "receptor_type": 0
                })

syn_dict = {
    "model": 'excitatory',
    "weight": [0.5]*len(n),
    "delay": [2.]*len(n),
}
conn_dict = {
    "rule": "one_to_one",
}

if nest.version()=='NEST 2.12.0':
    nest.Connect(n, m, conn_dict, syn_dict)
    # print 'Connecting ' + ' my_nest.GetConnections ', len(nest.GetConnections(n)), len(n)

if nest.version()=='NEST 2.2.2':
    nest.Connect(n, m, **{
        "model":"excitatory"
    })

with Stopwatch('Speed test'):
    nest.SetKernelStatus({'print_time':True})
    nest.Simulate(1000)