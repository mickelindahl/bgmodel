'''
Created on May 4, 2015

@author: mikael
'''
import os
# HOME=os.getenv('HOME')

from os.path import join, dirname, basename
from dotenv import load_dotenv

print '!!!!!'
# path = dirname(dirname(dirname(__file__)))
#
# # dotenv_path = join(path, '.env')
# # load_dotenv(dotenv_path)

# Set globals
HOME=os.getenv('BGMODEL_HOME')
HOME_CODE=os.getenv('BGMODEL_HOME_CODE')
HOME_DATA=os.getenv('BGMODEL_HOME_DATA')
HOME_MODULE=os.getenv('BGMODEL_HOME_MODULE')

# for items in  sorted(os.environ.items(), key=lambda x: x[0]):
#     print items