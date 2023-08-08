import nest

nest.Install('ml_module')
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

neuron = nest.Create("izhik_cond_exp")
voltmeter = nest.Create("voltmeter")

neuron.I_e = 20.0

nest.Connect(voltmeter, neuron)

nest.Simulate(1000.0)

nest.voltage_trace.from_device(voltmeter)
plt.show()