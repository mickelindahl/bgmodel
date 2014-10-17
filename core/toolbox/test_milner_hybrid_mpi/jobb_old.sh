#!/bin/bash -l

# The name of the script is myjob
#SBATCH -J dummy_job

# Wall-clock time will be given to this job
#SBATCH -t 00:10:00

# Number of MPI tasks.
#SBATCH -n 4

# Number of nodes
#SBATCH -N 1

# Number of MPI tasks per node. For aprun -N pes_per_node, e.i.
# number of processing elements (PEs) per node.
#SBATCH --ntasks-per-node=4 

# Number of cores hosting OpenMP threads
#SBATCH -c 5

#SBATCH -e jobb_err
#SBATCH -o jobb_out

#enable modules within the batch system
. /opt/modules/default/etc/modules.sh

#load the modules
module swap PrgEnv-cray PrgEnv-gnu
module add python
module add nest/2.2.2-wo-music


PYTHON=/pdc/vol/nest/2.2.2-wo-music/lib/python2.7/site-packages
PYTHON_GNU=/pdc/vol/python/2.7.6-gnu/lib/python2.7/site-packages

export OMP_NUM_THREADS=10

export PYTHONPATH=$PYTHONPATH:$PYTHON:$PYTHON_GNU:

aprun -n 4 -N 4 -d 10 python simulation.py 2>&1 | tee delme_simulation
