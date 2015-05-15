#!/bin/bash -l

mpirun -np 20 python {path_script} {path_params} 2>&1 | tee {path_tee_out}