# Create by Mikael Lindahl on 4/27/17.
# -*- coding: utf-8 -*-
from core import data_to_disk
import matplotlib.pyplot as plt
import os
import pprint
import json

pp=pprint.pprint


def main(mode, size):

    base = os.path.join(os.getenv('BGMODEL_HOME'), 'results/example/eneuro', str(size), mode)
    # path = os.path.join( base, 'data')
    path = os.path.join(base, 'randomized-params.json')
    path_out = os.path.join(base, 'randomized-params.png')

    if not os.path.exists(path):
        print 'Missing '+path+'. Need to run simulate.py'
        exit(0)

    data = json.load( open(path, 'r') )
    pp(data)


    plt.figure(figsize=(18, len(data.keys())*2))
    #
    # lables=[]
    # rates=[]
    # CVs=[]
    # x =list(range(8))
    # width = 0.8
    for i, key in enumerate(sorted(data.keys())):

        C_m=data[key]['C_m']
        V_th=data[key]['V_th']
        V_m = data[key]['V_m']
        VE = data[key]['V_th-E_L']

        ax=plt.subplot(8,4,1+i*4)
        ax.hist(C_m, 10, facecolor='green', alpha=0.75)
        ax.set_ylabel('Neurons (#)')
        ax.set_title(key+' #'+str(len(C_m))+ ' C_m mean '+str(sum(C_m)/len(C_m))[:4])

        ax=plt.subplot(8,4,2+i*4)
        ax.hist(V_th, 10, facecolor='blue', alpha=0.75)
        ax.set_ylabel('Neurons (#)')
        ax.set_title(key+' #'+str(len(C_m))+ ' V_th mean '+str(sum(V_th)/len(V_th))[:4])

        ax=plt.subplot(8,4,3+i*4)
        ax.hist(V_m, 10, facecolor='red', alpha=0.75)
        ax.set_ylabel('Neurons (#)')
        ax.set_title(key+' #'+str(len(C_m))+ ' V_m mean '+str(sum(V_m)/len(V_th))[:4])

        ax = plt.subplot(8, 4, 4+ i * 4)
        ax.hist(VE, 10, facecolor='magenta', alpha=0.75)
        ax.set_ylabel('Neurons (#)')
        ax.set_title(key + ' #' + str(len(C_m)) + ' V_th-E_L mean ' + str(sum(VE) / len(VE))[:4])

        # ax=plt.subplot(8,3,3+i*3)
        # ax.hist(C_m, 10, facecolor='blue', alpha=0.75)
        # ax.set_ylabel('Neurons (#)')
        # ax.set_title(key+' #'+str(len(C_m))+ '(V_th)')


    # plt.tight_layout()
    plt.subplots_adjust( wspace=0.2, hspace=0.4)
    title = mode + ' | netw size ' + str(size)

    plt.suptitle(title, fontsize=16)


    plt.savefig(path_out, format='png', dpi=70)

if __name__ == '__main__':

    # mode = 'activation-control'
    mode = 'activation-dopamine-depleted'
    size = 20000

    main(mode, size)

#


