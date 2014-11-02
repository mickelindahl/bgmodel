#!/bin/bash -l

# The name of the script is myjob
#SBATCH -J lindahl_test_job

# Wall-clock time will be given to this job
#SBATCH -t 00:10:00

# Number of MPI tasks.
#SBATCH -n 4

# Number of nodes
#SBATCH -N 1

# Number of MPI tasks per node
#SBATCH --ntasks-per-node=4

# Number of cores hosting OpenMP threads
#SBATCH -c 4

#SBATCH -e /home/mikael/results/unittest/parallel_excecution/supermicro/sbatch_err
#SBATCH -o /home/mikael/results/unittest/parallel_excecution/supermicro/sbatch_out


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
SRC=$HOME/git/bgmodel/core:HOME/git/bgmodel/

export PYTHONPATH=$PYTHONPATH:$NEURO_TOOLS:$PYTHON:$PYTHON_GNU:$SRC
export OMP_NUM_THREADS=10


mpirun -np 20 python /home/mikael/git/bgmodel/core/toolbox/parallel_excecution/simulation.py /home/mikael/results/unittest/parallel_excecution/supermicro/params_in.pkl 2>&1 | tee /home/mikael/results/unittest/parallel_excecution/supermicro/tee_mpi_out