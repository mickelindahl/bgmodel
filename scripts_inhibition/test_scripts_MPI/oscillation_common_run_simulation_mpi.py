'''
Created on Sep 26, 2014

@author: mikael
'''
import sys

from toolbox import data_to_disk
from scripts_inhibition.oscillation_common import run_simulation

path_in,path_out=sys.argv[1:]

from_disk, threads=data_to_disk.pickle_load(path_in)

v=run_simulation(from_disk=from_disk,
               threads=threads,
               type_of_run='mpi_supermicro')

data_to_disk.pickle_save(v, path_out)