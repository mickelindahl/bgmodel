'''
Created on Oct 13, 2014

@author: mikael
'''
import sys

from toolbox.my_nest import sim_group

nest_path, np,=sys.argv[1:]

sim_group(nest_path, **{'total_num_virtual_procs':int(np)})
