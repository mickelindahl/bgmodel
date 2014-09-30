#!/bin/bash -l

# The name of the script is myjob
#SBATCH -J {job_name}

# Wall-clock time will be given to this job
#SBATCH -t {hours}:{minutes}:{seconds}

# Number of MPI tasks.
# always ask for complete nodes (i.e. mppwidth should normally 
# be a multiple of 20)
#SBATCH -n {nodes_reserved}

#SBATCH -e {path_sbatch_err}
#SBATCH -o {path_sbatch_out}


if [ {on_milner} -eq 1 ]; then

	#enable modules within the batch system
	. /opt/modules/default/etc/modules.sh
	
	#load the modules
	module swap PrgEnv-cray PrgEnv-gnu
	module add python
	module add nest

fi
HOME={home}
NEURO_TOOLS=$HOME/local/lib/python2.7/site-packages
PYTHON=/pdc/vol/nest/2.2.2/lib/python2.7/site-packages
PYTHON_GNU=/pdc/vol/python/2.7.6-gnu/lib/python2.7/site-packages
SRC=$HOME/git/bgmodel/core


export PYTHONPATH=$PYTHONPATH:$NEURO_TOOLS:$PYTHON:$PYTHON_GNU:$SRC

{call}
            