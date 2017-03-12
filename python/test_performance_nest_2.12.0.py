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
m=nest.Create('my_aeif_cond_exp', 1)
#
nest.CopyModel("static_synapse","excitatory",
               {"weight":2.5,
                "delay":0.5,
                "receptor_type": rec["AMPA_1"]})
#
nest.Connect(n, m, **{
    "model":"excitatory"
})

nest.SetKernelStatus({'print_time':True})
nest.Simulate(100000)