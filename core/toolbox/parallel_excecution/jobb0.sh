#!/bin/bash -l

# The name of the script is myjob
#SBATCH -J dummy_job

# Wall-clock time will be given to this job
#SBATCH -t 00:10:00

# Number of MPI tasks.
#SBATCH -n {num-mpi-task}

# Number of nodes
#SBATCH -N {num-of-nodes}

# Number of MPI tasks per node
#SBATCH --ntasks-per-node={num-mpi-tasks-per-node}

# Number of cores hosting OpenMP threads
#SBATCH -c {cores_hosting_OpenMP_threads}

#SBATCH -e {path_sbatch_err}
#SBATCH -o {path_sbatch_out}


#if [ "foo" = "bar"  ]; then
if [ {on_milner} -eq 1 ]; then
	echo "Inside"

	#enable modules within the batch system
	. /opt/modules/default/etc/modules.sh
	
	#load the modules
	module swap PrgEnv-cray PrgEnv-gnu
	module add python
	module add nest/2.2.2-wo-music
fi

HOME={home}
NEURO_TOOLS=$HOME/local/lib/python2.7/site-packages
PYTHON=/pdc/vol/nest/2.2.2-wo-music/lib/python2.7/site-packages
PYTHON_GNU=/pdc/vol/python/2.7.6-gnu/lib/python2.7/site-packages
SRC=$HOME/git/bgmodel/core

export PYTHONPATH=$PYTHONPATH:$NEURO_TOOLS:$PYTHON:$PYTHON_GNU:$SRC
export OMP_NUM_THREADS={num-threads-per-mpi-process}


{call}