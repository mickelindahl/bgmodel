# Create by Mikael Lindahl on 4/27/17.

from core import data_to_disk
import matplotlib.pyplot as plt
import os
import pprint
import json

pp=pprint.pprint

def main(mode,size):
    # mode = 'activation-control'

    base = os.path.join(os.getenv('BGMODEL_HOME'), 'results/example/eneuro', str(size), mode )
    path = os.path.join( base, 'data')
    path_out = os.path.join(base, 'statistic.json')


    print(path)

    sd = data_to_disk.Storage_dic.load(path, ['Net_0'])

    print(sd)

    data={}

    for node in ['FS', 'M1', 'M2', 'ST', 'GI', 'GA', 'GF',  'SN']:

        fig=plt.figure()
        spk=sd['Net_0'][node]['spike_signal'].load_data().wrap.as_spike_list()

        st = spk.Factory_spike_stat()
        isi = st.isi
        del isi['raw']

        for key in st.rates.keys():
            st.rates[key] = round(st.rates[key],5)

        for key in isi.keys():
            isi[key] = round(isi[key],5)

        data[node]={
            'rate':st.rates,
            'isi':isi
        }
        # r=st.rates

    pp(data)

    json.dump(data, open(path_out, 'w'))

if __name__ == '__main__':

    mode = 'activation-dopamine-depleted'
    size = 20000
    main(mode, size)

# print sd

