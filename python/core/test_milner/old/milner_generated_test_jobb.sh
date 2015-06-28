#!/bin/bash -l
# The name of the script is myjob
#SBATCH -J dummy_job

# Wall-clock time will be given to this job
#SBATCH -t 00:10:00

# Number of MPI tasks.cd ..
# always ask for complete nodes (i.e. mppwidth should normally
# be a multiple of 20)
#SBATCH -n 20

#SBATCH -e err
#SBATCH -o out

#enable modules within the batch system
. /opt/modules/default/etc/modules.sh

#load the modules
module swap PrgEnv-cray PrgEnv-gnu
module add python
module add nest

HOME_MILNER=/cfs/milner/scratch/l/lindahlm
NEST_INSTALL=/pdc/vol/nest/2.2.2/lib/nest
NEURO_TOOLS=$HOME_MILNER/local/lib/python2.7/site-packages
PYTHON=/pdc/vol/nest/2.2.2/lib/python2.7/site-packages
PYTHON_GNU=/pdc/vol/python/2.7.6-gnu/lib/python2.7/site-packages
SRC=$HOME_MILNER/git/bgmodel/core

export PYTHONPATH=$NEURO_TOOLS:$PYTHON:$PYTHON_GNU:$SRC

aprun -n 20 python milner_test_simulation.py 2>&1 | tee delme_milner_test_simulation.txt
