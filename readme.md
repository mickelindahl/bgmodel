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

Make sure you have installed all 
[dependencies](https://github.com/mickelindahl/bgmodel#dependencies)

clone the repository

```
git clone https://github.com/mickelindahl/bgmodel.git
```

Got to `dist` director within model
```
cd {path to model}/dist
```

These install scripts are available in `nest/`. 
Choose one for your system and run it.

* `install_2.12.0_linux.sh`
* `install_2.6.0_linux.sh`
* `install_2.2.2_linux.sh`
* `install_2.12.0_mac.sh`


Then copy `sample.env` to `.env`
```
cp sample.env .env
```
Open `.env` and set data result path

Done!

## Install neurotools on beskow

Stand in the root of the model and run
```sh
module		 load python/2.7.13
TMP=$(pwd)

mkdir tmp
cp resources/NeuroTools-0.3.1.tar.gz tmp
cd tmp
tar -xzf NeuroTools-0.3.1.tar.gz
cd NeuroTools-0.3.1
python setup.py install --prefix=$TMP/local
cd ..
cd ..
rm -r tmp
```

Then add `{model root}/local/lib64/python2.6/site-packages` to your python path



## Dependencies

* python: numpy, scipy, mpi4py, NeuroTools (0.2.0), psycopg2
others: openmpi, libncurses-dev, libreadline-dev, libopenmpi-dev, libgsl, gsl (gnu scitific library, solver, neccesary for module) 

Dependencies module
* autoconf, automake

Dependences python
* cython



```
suod apt-get install cython
sudo pip install python-dotenv
sudo pip install NeuroTools
sudo pip install mpi4py
sudo pip install psycopg2
```






