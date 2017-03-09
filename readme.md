# Bgmodel
This is the basal ganglia model used in the paper 
["Untangling basal ganglia network dynamics and function – role of dopamine depletion and inhibition investigated in a spiking network model"](http://eneuro.org/content/early/2016/12/22/ENEURO.0156-16.2016.article-info),
It build using [pyNEST](http://www.nest-simulator.org/introduction-to-pynest/) that under the 
hood utilize the [NEST simulator](http://www.nest-simulator.org/). The model have been run Nest 2.4 see [nest download](http://www.nest-simulator.org/download/).

##Installation

Dependencies:
* python: numpy, scipy, mpi4py, NeuroTools (0.2.0), psycopg2
others: openmpi, libncurses-dev, libreadline-dev, libopenmpi-dev, libgsl, gsl (gnu scitific library, solver, neccesary for module) 

In `~/git` run git clone `git clone https://github.com/mickelindahl/bgmodel.git`

###Install nest
Download nest to `~/opt/NEST/dist/` (nest-2.4.2)

Then run `ln -s  ~/git/bgmodel/nest/dist/compile-nest-mpi.sh ~/opt/NEST/dist/compile-nest-mpi.sh`

Go to folder `cd ~/opt/NEST/dist` and run `./compile-nest-mpi.sh nest-2.4.2` (OBS make sure you have GSL libaries)

Update  `PYTHONPATH` with `~/opt/NEST/dist/install-nest-2.2.2/lib/python2.7/site-packages/`
in `~/.bashrc`

```sh
PYNEST=/home/mikael/opt/NEST/dist/install-nest-2.2.2/lib/python2.7/site-packages/
export PYTHONPATH=$PYTHONPATH:$PYNEST
```

test if pynest works run `python` and then `import nest`

### Install module

Dependencies
* autoconf, automake

Run
`ln -s ~/git/bgmodel/nest/module/module-130701 /home/mikael/opt/NEST/module/module-130701`
`ln -s /home/mikael/git/bgmodel/nest/module/compile-module.sh /home/mikael/opt/NEST/module/compile-module.sh`

First goto module folder  `cd ~/opt/NEST/module/`
Then run `compile-module.sh {module folder} {nest version} {nest installation directory} {nest model source directory}` 

```
./compile-module.sh module-130701 nest-2.2.2 ~/opt/NEST/dist/install-nest-2.2.2/ ~/opt/NEST/dist/nest-2.2.2/models/
```

Don't forget to add nest installation directory to `LD_LIBRARY_PATH` in bashrc
Add
```
NEST_INSTALL_DIR=/home/mikael/opt/NEST/dist/install-nest-2.2.2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NEST_INSTALL_DIR/lib/nes
```
to .bashrc and then run
```
. ~/.bashrc
```

## Running the model

### Env variables
Add these to environmental variables on your system
`BGMODEL_HOME`, `BGMODEL_HOME_CODE`, `BGMODEL_HOME_DATA`, `BGMODEL_HOME_MODULE`

```
export BGMODEL_HOME=$HOME
export BGMODEL_HOME_CODE="$BGMODEL_HOME/git/bgmodel/python"
export BGMODEL_HOME_DATA="$BGMODEL_HOME/results/papers/inhibition/network/thalamus"
export BGMODEL_HOME_MODULE="$BGMODEL_HOME/opt/NEST/module/install-module-130701-nest-2.2.2"
```
where  
* BGMODEL_HOME - home folder
* BGMODEL_HOME_DATA - folder to store data in. It constitutes of a main path plust the name of the host you are running on. Se moudle my_socket how it is retireed. Host is usally your computer name. 
* BGMODEL_HOME_CODE - folder where python code for project resides
* BGMODEL_HOME_MODULE - folder where nest module is installed

Add your script folder to bgmodel as script_{project name}

Copy config.py to script folder. 

Might need to adjust these methods depending on your system. If you are running on a system that can mpirun then
you should be fine. But if you are using a supercomputer system then 
some adjustment might be needed in order to get the job_handler to work.
There is one set of config classes which works on a system that uses sbatch to initiate scripts with
a aprun command. 

















