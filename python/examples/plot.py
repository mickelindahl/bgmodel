# Create by Mikael Lindahl on 4/27/17.

from core import data_to_disk
import pylab

path = '/Users/Mohammad/Documents/PhD/Projects/BGmodel/bgmodel/results/example/simulate/data'

sd = data_to_disk.Storage_dic.load(path, ['Net_0'])

for key in ['FS', 'M1', 'M2', 'ST', 'GI', 'GA',  'SN']:

    fig=pylab.figure()
    spk=sd['Net_0'][key]['spike_signal'].load_data().wrap.as_spike_list()

    fr=spk.Factory_firing_rate()
    ax=pylab.subplot(211)
    fr.plot(ax)
    ax.set_title(key)

    ax=pylab.subplot(212)
    ax.scatter(spk.get_raster()['x'], spk.get_raster()['y'])
    ax.set_xlim([0,2000])
pylab.show()

print sd

