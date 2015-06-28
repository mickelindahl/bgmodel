#!/bin/bash
#PBS -d {path_qsubout}
#PBS -j oe
#PBS -N {job_name}
#PBS -l mem={memsize}
#PBS -l walltime={hours}:{minutes}:{seconds}
python {path_script} {path_params} 2>&1 | tee {path_tee_out} 

