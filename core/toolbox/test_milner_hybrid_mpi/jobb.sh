#!/bin/bash -l

# The name of the script is myjob
#SBATCH -J dummy_job

# Wall-clock time will be given to this job
#SBATCH -t 00:10:00

# Number of MPI tasks.
#SBATCH -n 8

# Number of nodes
#SBATCH -N 2

# Number of MPI tasks per node
#SBATCH --ntasks-per-node=4

# Number of cores hosting OpenMP threads
#SBATCH -c 4

#SBATCH -e /home/mikael/results/unittest/test_milner_hybrid_mpi/main/jobb_err
#SBATCH -o /home/mikael/results/unittest/test_milner_hybrid_mpi/main/jobb_out


#if [ "foo" = "bar"  ]; then
if [ 0 -eq 1 ]; then
	echo "Inside"

	#enable modules within the batch system
	. /opt/modules/default/etc/modules.sh
	
	#load the modules
	module swap PrgEnv-cray PrgEnv-gnu
	module add python
	module add nest/2.2.2-wo-music
fi

HOME=/home/mikael
NEURO_TOOLS=$HOME/local/lib/python2.7/site-packages
PYTHON=/pdc/vol/nest/2.2.2-wo-music/lib/python2.7/site-packages
PYTHON_GNU=/pdc/vol/python/2.7.6-gnu/lib/python2.7/site-packages
SRC=$HOME/git/bgmodel/core

export PYTHONPATH=$PYTHONPATH:$NEURO_TOOLS:$PYTHON:$PYTHON_GNU:$SRC
export OMP_NUM_THREADS=10


aprun -n 8 -N 4 -d 10 python /home/mikael/git/bgmodel/core/toolbox/test_milner_hybrid_mpi/simulation.py 2>&1 | tee /home/mikael/results/unittest/test_milner_hybrid_mpi/main/delme_simulation

