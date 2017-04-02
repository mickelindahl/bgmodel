# Create by Mikael Lindahl on 3/12/17.

import nest
import pprint
pp=pprint.pprint

nest.Install('ml_module')

par={
    'period_first':500.0,
    'period_second':500.0,
    'rate_first':1000.,
    'rate_second':800.
}

df =nest.GetDefaults("my_aeif_cond_exp")
pp(df)

nest.CopyModel("my_aeif_cond_exp","FS",
               {"b":2.5})

rec=nest.GetDefaults("my_aeif_cond_exp")["receptor_types"]
pp(rec)
n=nest.Create(**{
    'model':'poisson_generator_periodic',
    'n':100,
    'params':par})
# m=nest.Create('aeif_cond_exp', 1000)
# m=nest.Create('my_aeif_cond_exp', 100)
m=nest.Create('izhik_cond_exp', 100)
#
nest.CopyModel("static_synapse","excitatory",
               {"weight":2.5,
                "delay":0.5,
                "receptor_type": rec["AMPA_1"]
                # "receptor_type": 0
                })

syn_dict = {
    "model": 'excitatory',
    "weight": [0.5]*len(n),
    "delay": [2.]*len(n),
}
conn_dict = {
    "rule": "one_to_one",
}

nest.Connect(n, m, conn_dict, syn_dict)
print 'Connecting ' + ' my_nest.GetConnections ', len(nest.GetConnections(n)), len(n)

#

# nest.Connect(n, m, **{
#     "model":"excitatory"
# })

nest.SetKernelStatus({'print_time':True})
nest.Simulate(1000)