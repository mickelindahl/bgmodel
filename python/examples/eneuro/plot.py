# Create by Mikael Lindahl on 4/27/17.

from core import data_to_disk
import matplotlib.pyplot as plt
import os



def main(mode, size):

    base = os.path.join(os.getenv('BGMODEL_HOME'), 'results/example/eneuro', str(size), mode)
    #
    # path = '/Users/Mohammad/Documents/PhD/Projects/BGmodel/bgmodel/results/example/simulate/data'
    # path_res = '/Users/Mohammad/Documents/PhD/Projects/BGmodel/bgmodel/results/example/simulate/Fig'

    path = os.path.join(base, 'data')
    path_fr = os.path.join(base, 'firing-rate')
    path_scatter = os.path.join(base, 'scatter')

    if not os.path.isdir(path_fr):
        data_to_disk.mkdir(path_fr)

    if not os.path.isdir(path_scatter):
        data_to_disk.mkdir(path_scatter)

    # path_scatter = os.path.join(base, 'scatter.png')

    sd = data_to_disk.Storage_dic.load(path, ['Net_0'])


    for i, key in enumerate(['FS', 'M1', 'M2', 'ST', 'GI', 'GF', 'GA', 'SN']):

        spk=sd['Net_0'][key]['spike_signal'].load_data().wrap.as_spike_list()

        st = spk.Factory_spike_stat()

        fr=spk.Factory_firing_rate()

        fig = plt.figure( tight_layout=True)
        ax=fig.add_subplot(111)
        fr.plot(ax)
        ax.set_title(key+' | '+str(st.rates['mean'])[:4]+' Hz | '+mode+ ' | # '+ str(len(spk.ids))  +' ('+str(size)+')')

        plt.savefig(os.path.join(path_fr,key+'.png'), format='png')


        fig = plt.figure(tight_layout=True)
        ax=fig.add_subplot(111)
        ax.scatter(spk.get_raster()['x'], spk.get_raster()['y'])
        ax.set_title(key+' | '+str(st.rates['mean'])[:4]+' Hz | '+mode+ ' | # '+ str(len(spk.ids))  +' ('+str(size)+')')

        plt.savefig(os.path.join(path_scatter, key + '.png'), format='png')


        #ax.set_xlim([0,2000])
        # plt.savefig(path_res+'/ramp-slope5th-'+key+'.pdf',format='pdf')

    # plt.tight_layout()

    #
    #plt.show()

    print sd


if __name__ == '__main__':
    mode = 'activation-dopamine-depleted'
    # mode = 'slow-wave-control'
    size = 3000

    main(mode, size)