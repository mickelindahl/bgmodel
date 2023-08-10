#!/bin/bash -l

#enable modules within the batch system
. /opt/modules/default/etc/modules.sh

#load the modules
module swap PrgEnv-cray PrgEnv-gnu
module add python
module add nest

echo "Swaped PrgEnv-cray PrgEnv-gnu"
echo "added python, nest"


HOME_MILNER=/cfs/milner/scratch/l/lindahlm
NEST_INSTALL=/pdc/vol/nest/2.2.2/lib/nest
NEURO_TOOLS=$HOME_MILNER/local/lib/python2.7/site-packages
PSYCOPG2=$HOME_MILNER/local/lib64/python2.6/site-packages
PYTHON=/pdc/vol/nest/2.2.2/lib/python2.7/site-packages
PYTHON_GNU=/pdc/vol/python/2.7.6-gnu/lib/python2.7/site-packages

export BGMODEL_HOME="$HOME_MILNER/git/bgmodel"
export BGMODEL_HOME_CODE="$HOME_MILNER/git/bgmodel/python"
export BGMODEL_HOME_DATA="$HOME_MILNER/results/papers/inhibition/network/milner"

#SRC=$HOME_MILNER/git/bgmodel/core
SRC=$HOME_MILNER/git/bgmodel/python
#SRC=$SRC:$HOME_MILNER/git/bgmodel/ #For pickle load module
echo "Python path before:"
echo $PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$NEURO_TOOLS:$PSYCOPG2:$PYTHON:$PYTHON_GNU:$SRC

echo "Python path set to:"
echo $PYTHONPATH
