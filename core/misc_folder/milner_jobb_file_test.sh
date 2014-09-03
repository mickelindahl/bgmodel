#!/bin/bash -l

# Take jobfile as input

# The name of the script is myjob
#SBATCH -J lindahlm_jobb

# Only 1 hour wall-clock time will be given to this job
#SBATCH -t 0:10:00

# Number of MPI tasks.cd ..
# always ask for complete nodes (i.e. mppwidth should normally
# be a multiple of 20)
#SBATCH -n 40

#SBATCH -e error_file.e
#SBATCH -o output_file.o


#load the nest modulecp
module swap PrgEnv-cray PrgEnv-gnu
module add python
module add nest

HOME_MILNER=/cfs/milner/scratch/l/lindahlm

MODULE=$HOME_MILNER/opt/NEST/module/install-module-130701-nest-2.2.2-milner/lib/nest
NEST_INSTALL=/pdc/vol/nest/2.2.2/lib/nest
NEURO_TOOLS=$HOME_MILNER/local/lib/python2.7/site-packages
PYTHON=/pdc/vol/nest/2.2.2/lib/python2.7/site-packages
PYTHON2=/pdc/vol/python/2.7.6-gnu/lib/python2.7/site-packages
SRC=$HOME/git/bgmodel/core

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cfs/milner/scratch/b/bkaplan/BCPNN-Module/build-module-100725
#export PYTHONPATH=/pdc/vol/nest/2.2.2/lib/python2.7/site-packages:/pdc/vol/python/2.7.6-gnu/lib/python2.7/site-packages

#LD_LIBRARY_PATH=/pdc/vol/python/2.7.6-gnu/lib:/pdc/vol/openssl/1.0.0l/lib:/pdc/vol/music/multiconn-961/lib:/pdc/vol/gsl/1.16/lib:/pdc/vol/python/2.7.6/lib

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MODULE
#$MODULE:$LD_LIBRARY_PATH
export PYTHONPATH=$NEURO_TOOLS:$PYTHON:$PYTHON2:$SRC

echo $PYTHONPATH
echo ''
echo $LD_LIBRARY_PATH
echo ''
echo "Starting test 1 at `date`"
# Run and write the output into my_output_file
echo $1.py
aprun -n 2 python $1.py 2>&1 | tee delme_$1 
echo "Stopping test 1 at `date`"


