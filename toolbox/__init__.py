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

import network_connectivity
import network_construction
import misc
import my_axes
import my_nest
import my_population
import my_signals
import my_topology
import plot_settings
