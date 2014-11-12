'''
Created on Sep 30, 2014

@author: mikael
'''
import sys
import time
import subprocess
time.sleep(5)
path,=sys.argv[1:]

subprocess.Popen(['cp', path+'/data0',path+'/data2'])

