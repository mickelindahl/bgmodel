# -*- coding: utf-8 -*-
"""
nest_toolbox
========

A collection of tools for modeling neuron networks, plotting data, saving data.


Modules
------
misc                 - contains miscellaneous functions. I.e thus not at the 
                       moment fit in any of the other modules
my_populations.py    - contains class MuGroup representing a neuron population, 
                       with recorders, data handling (conductance, current, voltage, 
                       spikes)
my_signals           - Contains classes, MyConductanceList, MyCurrentList,
                       MyVmList and MySpikeList which has NeuroTools classes as
                       base classes. That is inherites all methods from 
                       NeuroTools objects. Here additional functions can be 
                       added which one thinks are missing in neurotools.
                   
my_nest              - module for creating your own nest functions. Inherits 
                       all nest functions. 
plot_settings        - contains functions setting nice plot settings

"""

import misc
import data_to_disk
import network
import my_axes
import my_nest
import my_population
import my_signals
import my_topology
import plot_settings


def get_data_root_path(flag):
    
    if flag=='unittest':
        p='/afs/nada.kth.se/home/w/u1yxbcfw/results/papers/unittest'
     
    if flag=='inhibition':
        p='/afs/nada.kth.se/home/w/u1yxbcfw/results/papers/inhibition/'
     
    if flag in ['bcpnn_h0', 'bcpnn_h1']:
        p='/afs/nada.kth.se/home/w/u1yxbcfw/results/papers/bcpnn'
     
    return p 
def get_figure_root_path(flag):
    
    if flag=='unittest':
        p=('/afs/nada.kth.se/home/w/u1yxbcfw/projects/papers/unittest'
            +'/figures')
    
    if flag=='inhibition':
        p=('/afs/nada.kth.se/home/w/u1yxbcfw/projects/papers/inhibition'
           +'/figures/')
    
    if flag in ['bcpnn_h0', 'bcpnn_h1']:
        p=('/afs/nada.kth.se/home/w/u1yxbcfw/projects/papers/bcpnnbg'
           +'/figures')
    return p
