# Bgmodel
This basal ganglia model (tag 1.0.0) were used in the paper 
["Untangling basal ganglia network dynamics and function – role of dopamine depletion and inhibition investigated in a spiking network model"](http://eneuro.org/content/early/2016/12/22/ENEURO.0156-16.2016.article-info).
It build using [pyNEST](http://www.nest-simulator.org/introduction-to-pynest/) that under the 
hood utilize the [NEST simulator](http://www.nest-simulator.org/). The model have been run Nest 2.6 see [nest download](http://www.nest-simulator.org/download/).

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






