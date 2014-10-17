'''
Created on Oct 7, 2014

@author: mikael
'''
import subprocess
from toolbox.misc import Stopwatch
    
# os.environ['OMP_sNUM_THREADS'] = '2'   
# fileName = data_path + 'data_in.pkl'
# fileOut = data_path + 'data_out.pkl'


for np in [1,2,4]:
    
    with Stopwatch('Np='+str(np)):
        p = subprocess.Popen(['mpirun', '-np', str(np), 'python', 
                              'simulation.py'], 
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
#         stderr=subprocess.STDOUT
        )
        out, err = p.communicate()
    #         print out
    #         print err