import nest
nest.Install('ml_module')

from python.core.network.default_params import Inhibition
import fig_01_and_02_pert as op
import pprint

pp = pprint.pprint


par = Inhibition(**{'perturbations': op.get()[0]})
pp(par.dic['nest'].keys())

print('')
for m in sorted(nest.node_models):
    if m in ['izhik_cond_exp', 'pif_psc_alpha']:
        print('Model', m, 'exists')

import nest.voltage_trace
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

nest.set_verbosity("M_WARNING")
nest.ResetKernel()

p = par.dic['nest']['M1']
del p["type_id"]

# pp(p)

neuron = nest.Create("izhik_cond_exp", params=p)
voltmeter = nest.Create("voltmeter")

neuron.I_e = 300.0

nest.Connect(voltmeter, neuron)

nest.Simulate(1000.0)

nest.voltage_trace.from_device(voltmeter)
plt.show()
