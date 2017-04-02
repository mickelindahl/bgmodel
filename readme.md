# Bgmodel
This is the basal ganglia model used in the paper 
["Untangling basal ganglia network dynamics and function – role of dopamine depletion and inhibition investigated in a spiking network model"](http://eneuro.org/content/early/2016/12/22/ENEURO.0156-16.2016.article-info),
It build using [pyNEST](http://www.nest-simulator.org/introduction-to-pynest/) that under the 
hood utilize the [NEST simulator](http://www.nest-simulator.org/). The model have been run Nest 2.6 see [nest download](http://www.nest-simulator.org/download/).

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

## Installation

Make sure you have installed all dependencies

clone the repository

```
git clone https://github.com/mickelindahl/bgmodel.git
```

Got to `dist` director within model
```
cd {path to model}/dist
```

Install nest with module either for 
[linux](https://github.com/mickelindahl/bgmodel#linux) 
or [mac](https://github.com/mickelindahl/bgmodel#mac) 

Else to only install the model with previous nest installtion
```
cd {path to model}/module/compile-module-2.12.0.sh {path to nest installation}
```
Int `{path to model}/` Copy `sample.env` to `.env`
```
cp sample.env .env
```
Open `.env` and set data result path

Done!

## Install nest and module linux
### Nest 2.12.0 
Run
```
./install_2.12.0.sh
```

#### Nest 2.2.2
Run
```
./install_2.2.2.sh
```

### Install nest and module Mac
#### Nest 2.12.0
Run
```
./install_2.12.0_mac.sh
```


















