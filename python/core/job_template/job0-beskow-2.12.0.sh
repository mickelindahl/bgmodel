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

#load the modules
module swap PrgEnv-cray PrgEnv-gnu
module load gsl/2.3
module load python/2.7.13

export PATH={nest_installation_path}/bin:$PATH
export MANPATH=/pdc/vol/nest/2.12.0-py27/share/man:$MANPATH
export CRAY_ROOTFS=DSL
export CRAYPE_LINK_TYPE=dynamic

HOME={root_model}
NEURO_TOOLS={root_model}/local/lib64/python2.6/site-packages
PYTHON=/pdc/vol/nest/2.12.0-py27/lib64/python2.7/site-packages

SRC={root_model}/python

export BGMODEL_HOME={root_model}
export BGMODEL_HOME_CODE={root_model}/python
export BGMODEL_HOME_DATA={root_model}/results

export PYTHONPATH=$PYTHONPATH:$NEURO_TOOLS:$PYTHON:$SRC
export OMP_NUM_THREADS={num-threads-per-mpi-process}

aprun -n {num-mpi-task} -N {num-mpi-tasks-per-node} -d {num-threads-per-mpi-process} -m {memory_per_node} {cmd} 2>&1 | tee {path_tee_out}
