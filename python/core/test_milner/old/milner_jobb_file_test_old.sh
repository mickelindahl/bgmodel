#!/bin/bash -l

# Take jobfile as input

# The name of the script is myjob
#SBATCH -J lindahlm_jobb

# Only 1 hour wall-clock time will be given to this job
#SBATCH -t 0:10:00

# Number of MPI tasks.cd ..
# always ask for complete nodes (i.e. mpp width should normally
# be a multiple of 20)
#SBATCH -n 40

#SBATCH -e error_file.e
#SBATCH -o output_file.o

#enable modules within the batch system
. /opt/modules/default/etc/modules.sh

#load the nest modulecp
module swap PrgEnv-cray PrgEnv-gnu
module add python
module add nest

HOME_MILNER=/cfs/milner/scratch/l/lindahlm
NEURO_TOOLS=$HOME_MILNER/local/lib/python2.7/site-packages
PYTHON=/pdc/vol/nest/2.2.2/lib/python2.7/site-packages
PYTHON_GNU=/pdc/vol/python/2.7.6-gnu/lib/python2.7/site-packages
SRC=$HOME_MILNER/git/bgmodel/core

export PYTHONPATH=$NEURO_TOOLS:$PYTHON:$PYTHON_GNU:$SRC

echo "Starting test 1 at `date`"
# Run and write the output into my_output_file
echo $1.py
aprun -n 2 python $1.py 2>&1 | tee delme_$1
echo "Stopping test 1 at `date`"
