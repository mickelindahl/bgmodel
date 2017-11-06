#!/bin/bash -l

# The name of the script is myjob
#SBATCH -J {job_name}

# Wall-clock time will be given to this job
#SBATCH -t {hours}:{minutes}:{seconds}

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

#enable modules within the batch system
. /opt/modules/default/etc/modules.sh

#module swap PrgEnv-cray PrgEnv-gnu
#module add nest
#module add cmake/2.8.12.2


#load the modules
module swap PrgEnv-cray PrgEnv-gnu
#module add python
#module add nest/2.2.2-wo-music
#module add nest/2.6.0

#prereq	 PrgEnv-gnu
module		 load gsl/2.3
module		 load python/2.7.13
export PATH={nest_installation_path}/bin:$PATH
export MANPATH=/pdc/vol/nest/2.12.0-py27/share/man:$MANPATH
#export PYTHONPATH=/pdc/vol/nest/2.12.0-py27/lib64/python2.7/site-packages:$PYTHONPATH
export CRAY_ROOTFS=DSL
export CRAYPE_LINK_TYPE=dynamic


HOME={root_model}
NEURO_TOOLS={root_model}/local/lib64/python2.6/site-packages
#PYTHON=/pdc/vol/nest/2.2.2-wo-music/lib/python2.7/site-packages
#PYTHON=/pdc/vol/nest/2.6.0-wo-music/lib/python2.7/site-packages
PYTHON=/pdc/vol/nest/2.12.0-py27/lib64/python2.7/site-packages
#PYTHON_GNU=/pdc/vol/python/2.7.6-gnu/lib/python2.7/site-packages

SRC={root_model}/python

export BGMODEL_HOME={root_model}
export BGMODEL_HOME_CODE={root_model}/python
#export BGMODEL_HOME_DATA="$BGMODEL_HOME/results/papers/inhibition/network/milner"
#export BGMODEL_HOME_MODULE="$BGMODEL_HOME/opt/NEST/module/install-module-130701-nest-2.2.2-wo-music"
#export BGMODEL_HOME_MODULE="$BGMODEL_HOME/opt/NEST/module/install-module-150605-2.6.0-nest-2.6.0"
export BGMODEL_HOME_DATA={root_model}/results

export PYTHONPATH=$PYTHONPATH:$NEURO_TOOLS:$PYTHON:$SRC
export OMP_NUM_THREADS={num-threads-per-mpi-process}

aprun -n {num-mpi-task} -N {num-mpi-tasks-per-node} -d {num-threads-per-mpi-process} -m {memory_per_node} python {path_script} {path_params} 2>&1 | tee {path_tee_out}
