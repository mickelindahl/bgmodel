#!/bin/bash -l

# The name of the script is myjob
#SBATCH -J dummy_job

# Wall-clock time will be given to this job
#SBATCH -t 00:10:00

# Number of MPI tasks.
#SBATCH -n 2

# Number of nodes
#SBATCH -N 1

# Number of MPI tasks per node
#SBATCH --ntasks-per-node=2

# Number of cores hosting OpenMP threads
#SBATCH -c 10

#SBATCH -e jobb_err
#SBATCH -o jobb_out


#enable modules within the batch system
. /opt/modules/default/etc/modules.sh

#load the modules
module swap PrgEnv-cray PrgEnv-gnu
module add python
module add nest


HOME=/home/mikael
NEURO_TOOLS=$HOME/local/lib/python2.7/site-packages
PYTHON=/pdc/vol/nest/2.2.2/lib/python2.7/site-packages
PYTHON_GNU=/pdc/vol/python/2.7.6-gnu/lib/python2.7/site-packages
SRC=$HOME/git/bgmodel/core

export OMP_NUM_THREADS=20
export PYTHONPATH=$PYTHONPATH:$NEURO_TOOLS:$PYTHON:$PYTHON_GNU:$SRC

aprun -n 2 -N 4 -d 20 python simulation.py 2>&1 | tee delme_simulation
