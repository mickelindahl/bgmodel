"""
Created on May 4, 2015

@author: mikael
"""
import os
import sys

# Set globals
HOME = os.getenv('BGMODEL_HOME')
HOME_CODE = os.getenv('BGMODEL_HOME_CODE')
HOME_DATA = os.getenv('BGMODEL_HOME_DATA')

parent_directory_up_two_levels = os.path.dirname(os.path.dirname(sys.executable))
HOME_MODULE = os.getenv('BGMODEL_HOME_MODULE', parent_directory_up_two_levels)
