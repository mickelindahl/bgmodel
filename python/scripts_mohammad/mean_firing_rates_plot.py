# Create by Mikael Lindahl on 4/27/17.
# -*- coding: utf-8 -*-
from core import data_to_disk
import matplotlib.pyplot as plt
import os
import pprint
import json

pp = pprint.pprint
import math

def main(mode, size):

    base = os.path.join(os.getenv('BGMODEL_HOME'), 'results/example/eneuro', str(size), mode)
    # path = os.path.join( base, 'data')
    path = os.path.join(base, 'statistic.json')
    path_out = os.path.join(base, 'statistic.png')

    if not os.path.exists(path):
        print 'Missing ' + path + '. Need to run mean_firing_rates.py'
        exit(0)

    data = json.load(open(path, 'r'))

    pp(data)

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                    '%s' % str(height)[:5],
                    ha='center', va='bottom')

    plt.figure(figsize=(12, len(data.keys()) / 2 * 2))

    lables = []
    rates = []
    CVs = []
    x = list(range(8))
    width = 0.8
    for i, key in enumerate(sorted(data.keys())):
        rate = data[key]['rate']
        isi = data[key]['isi']

        rm = float(rate['mean'])
        rm = rm if math.isnan(rm) else 0

        CV = float(isi['CV'])
        CV = CV if math.isnan(CV) else 0

        rates.append(rm)
        CVs.append(CV)
        lables.append(key)

    print(i, key)
    ax = plt.subplot(2, 1, 1)
    rects1 = ax.bar(x, rates, width, color='b')
    # add some text for labels, title and axes ticks
    ax.set_ylabel('Rate (Hz)')
    ax.set_title('Nuclei rates')
    ax.set_xticks([v + width / 2 for v in x])
    ax.set_xticklabels(lables)
    autolabel(rects1)

    ax = plt.subplot(2, 1, 2)
    rects2 = ax.bar(x, CVs, width, color='r')
    # add some text for labels, title and axes ticks

    ax.set_title('Nuclei CV')
    ax.set_xticks([v + width / 2 for v in x])
    ax.set_xticklabels(lables)

    autolabel(rects2)

    title =  mode + ' | netw size ' + str(size)

    plt.suptitle(title, fontsize=16)
    # plt.show()
    plt.savefig(path_out, format='png', dpi=70)


if __name__ == '__main__':
    mode = 'activation-dopamine-depleted'
    size = 20000

    main(mode, size)
