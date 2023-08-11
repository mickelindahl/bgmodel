# Bgmodel
This is a basal ganglia model currently including striatum, 
globus pallidus, substantia nigra reticulata and subthalamus.
The model includes dopamine modulated neurons and synapses, 
 synapses with short-term plasticity and hybrid neuron models from 
 [Izhikevich](http://www.izhikevich.org/publications/spikes.htm) and
 [Brette](http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model).

This basal ganglia model (tag 1.0.0 and nest-2.2.2) were used in the paper 
["Untangling basal ganglia network dynamics and function – role of dopamine depletion and inhibition investigated in a spiking network model"](http://eneuro.org/content/early/2016/12/22/ENEURO.0156-16.2016).
It build using [pyNEST](http://www.nest-simulator.org/introduction-to-pynest/) that under the 
hood utilize the [NEST simulator](http://www.nest-simulator.org/). The model have been run Nest 2.6 see [nest download](http://www.nest-simulator.org/download/). 
Script used for the paper are located in `python/scripts_inhibition`. 

## Installation

This installation depends on that conda (anaconda or miniconda) is installed

Clone the repository

```
git clone https://github.com/mickelindahl/bgmodel.git
```

Copy `sample.env` to `.env`
```
cp sample.env .env
```

Open `.env` and set environment variables

Run install 

```
./install.sh
```

Done!

## Usage


## Dependencies
* conda







