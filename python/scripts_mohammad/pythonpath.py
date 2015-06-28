'''
Created on Jun 24, 2015

@author: mikael

'''


import os
import sys
currdir=os.getcwd()
basedir='/'.join(currdir.split('/')[:-1])

# Add directories to python path
sys.path.append(basedir)
sys.path.append(basedir +'/core')
sys.path.append(basedir +'/core/toolbox')  
sys.path.append(basedir +'/scripts_inhibition')
sys.path.append(currdir)
