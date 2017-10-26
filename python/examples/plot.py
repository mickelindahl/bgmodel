# Create by Mikael Lindahl on 4/27/17.

from core import data_to_disk
import matplotlib.pyplot as plt

path = '/Users/Mohammad/Documents/PhD/Projects/BGmodel/bgmodel/results/example/simulate/data'
path_res = '/Users/Mohammad/Documents/PhD/Projects/BGmodel/bgmodel/results/example/simulate/Fig'

sd = data_to_disk.Storage_dic.load(path, ['Net_0'])

for key in ['FS', 'M1', 'M2', 'ST', 'GI', 'GA',  'SN']:

    fig=plt.figure()
    spk=sd['Net_0'][key]['spike_signal'].load_data().wrap.as_spike_list()

    fr=spk.Factory_firing_rate()
    ax=plt.subplot(211)
    fr.plot(ax)
    ax.set_title(key)

    ax=plt.subplot(212)
    ax.scatter(spk.get_raster()['x'], spk.get_raster()['y'])
    #ax.set_xlim([0,2000])
    plt.savefig(path_res+'/ramp-slope5th-'+key+'.pdf',format='pdf')
#plt.show()

print sd

