'''
Created on Oct 6, 2014

@author: mikael
'''
from distutils.core import setup
from Cython.Build import cythonize


setup(
    ext_modules = cythonize(
#                             "hello_world.pyx",
#                             "nest_speed.pyx",
                                'connect_speed_fun.pyx'
                            )
)

