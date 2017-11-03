#!/bin/sh

# OBS need to run it with source in local shell

#Input: Take nest-install-dir

#module swap PrgEnv-cray PrgEnv-gnu
#module add nest
#module add cmake/2.8.12.2

#prereq	 PrgEnv-gnu
module		 load gsl/2.3
module		 load python/2.7.13
export PATH=${1}/bin:$PATH
export MANPATH=/pdc/vol/nest/2.12.0-py27/share/man:$MANPATH
export PYTHONPATH=/pdc/vol/nest/2.12.0-py27/lib64/python2.7/site-packages:$PYTHONPATH
export		 CRAY_ROOTFS=DSL
export CRAYPE_LINK_TYPE=dynamic