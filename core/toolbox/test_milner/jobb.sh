#!/bin/bash -l

# The name of the script is myjob
#SBATCH -J dummy_job

# Wall-clock time will be given to this job
#SBATCH -t 00:10:00

# Number of MPI tasks.cd ..
# always ask for complete nodes (i.e. mppwidth should normally
# be a multiple of 20)
#SBATCH -n 20

#SBATCH -e /home/mikael/results/unittest/test_milner/main/jobb_err
#SBATCH -o /home/mikael/results/unittest/test_milner/main/jobb_out


#if [ "foo" = "bar"  ]; then
if [ 0 -eq 1 ]; then
	echo "Inside"

	#enable modules within the batch system
	. /opt/modules/default/etc/modules.sh
	
	#load the modules
	module swap PrgEnv-cray PrgEnv-gnu
	module add python
	module add nest
fi

HOME=/home/mikael
NEURO_TOOLS=$HOME/local/lib/python2.7/site-packages
PYTHON=/pdc/vol/nest/2.2.2/lib/python2.7/site-packages
PYTHON_GNU=/pdc/vol/python/2.7.6-gnu/lib/python2.7/site-packages
SRC=$HOME/git/bgmodel/core

export PYTHONPATH=$PYTHONPATH:$NEURO_TOOLS:$PYTHON:$PYTHON_GNU:$SRC

mpirun -np 10 python /home/mikael/git/bgmodel/core/toolbox/test_milner/simulation.py /home/mikael/results/unittest/test_milner/main/nest /home/mikael/results/unittest/test_milner/main/spike_dic 2>&1 | tee /home/mikael/results/unittest/test_milner/main/delme_simulation
#SCRIPT=$3
#ARGV=$4
#OUTPUT=$5
#aprun -n 20 python $SCRIPT $ARGV 2>&1 | tee $OUTPUT