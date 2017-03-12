# Bgmodel
This is the basal ganglia model used in the paper 
["Untangling basal ganglia network dynamics and function – role of dopamine depletion and inhibition investigated in a spiking network model"](http://eneuro.org/content/early/2016/12/22/ENEURO.0156-16.2016.article-info),
It build using [pyNEST](http://www.nest-simulator.org/introduction-to-pynest/) that under the 
hood utilize the [NEST simulator](http://www.nest-simulator.org/). The model have been run Nest 2.6 see [nest download](http://www.nest-simulator.org/download/).

##Installation

Dependencies:
* python: numpy, scipy, mpi4py, NeuroTools (0.2.0), psycopg2
others: openmpi, libncurses-dev, libreadline-dev, libopenmpi-dev, libgsl, gsl (gnu scitific library, solver, neccesary for module) 

Dependencies module
* autoconf, automake


clone the repository

```
git clone https://github.com/mickelindahl/bgmodel.git
```

To install both nest and the model ooen terminal and enter 
```
cd {path to model}/dist
./install_nest_2.12_and_module_2.12.sh
```

Else to only install the model with prevoius nest installtion
```
cd {path to model}/module/compile-module-2.12.0.sh {path to nest installation}
```
Int `{path to model}/` Copy `sample.env` to `.env`
```
cp sample.env .env
```
Open `.env` adn edit it accordingly your system

If you have not added nest variables to your 
environment do it. E.g add the following to
 to `.bashrc`. That will set all nessecary env variables for
 running nest.

```sh
soruce ./bgmodel/nest/dist/install/nest-simulator-2.12.0/bin/nest_vars.sh
export BG_MODEL_PYTHON="{path to model}/python"
```

















