#!/bin/bash -l

# The name of the script is myjob
#SBATCH -J postgresql

# Wall-clock time will be given to this job
#SBATCH -t 00:05:00

# Number of MPI tasks.
#SBATCH -n 2

# Number of nodes
#SBATCH -N 1

# Number of MPI tasks per node
#SBATCH --ntasks-per-node=2

# Number of cores hosting OpenMP threads
#SBATCH -c 2

#SBATCH -e /cfs/milner/scratch/l/lindahlm/results/unittest/test_postgresql/jobb_err
#SBATCH -o /cfs/milner/scratch/l/lindahlm/results/unittest/test_postgresql/jobb_out

#if [ "foo" = "bar"  ]; then
if [ 1 -eq 1 ]; then
	echo "Inside"

	#enable modules within the batch system
	. /opt/modules/default/etc/modules.sh
	
	#load the modules
	module swap PrgEnv-cray PrgEnv-gnu
	module add python
	module add nest/2.2.2-wo-music
fi

HOME=/cfs/milner/scratch/l/lindahlm
NEURO_TOOLS=$HOME/local/lib/python2.7/site-packages
PSYCOPG2=$HOME_MILNER/local/lib64/python2.6/site-packages
PYTHON=/pdc/vol/nest/2.2.2-wo-music/lib/python2.7/site-packages
PYTHON_GNU=/pdc/vol/python/2.7.6-gnu/lib/python2.7/site-packages
SRC=$HOME/git/bgmodel/core:HOME/git/bgmodel/

export PYTHONPATH=$PYTHONPATH:$NEURO_TOOLS:$PSYCOPG2:$PYTHON:$PYTHON_GNU:$SRC
export OMP_NUM_THREADS=20


aprun -n 2 -N 2 -d 20 -m 16380 python /cfs/milner/scratch/l/lindahlm/git/bgmodel/core/misc_folder/test_postgresql_on_milner/connect_to_database.py