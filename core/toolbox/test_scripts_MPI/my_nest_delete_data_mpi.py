'''
Created on Oct 13, 2014

@author: mikael
'''
import sys

from toolbox.my_nest import delete_data

nest_path, np,=sys.argv[1:]

delete_data(nest_path)
