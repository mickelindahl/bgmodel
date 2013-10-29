'''
Created on Aug 5, 2013

@author: lindahlm
'''

import numpy
import scipy.signal as signal
import pylab

fs=1000.0
std=40.0
bin=100.0

kernel = signal.gaussian(bin, std*fs/1000.0)
time=n = (numpy.arange(0,bin) - (bin-1.0)/2.0)*1000/fs
#numpy.linspace(-bin/2, bin/2, bin)*1000/fs

pylab.plot(time, kernel)


fs=100.0
std=40.0
bin=10.0

kernel = signal.gaussian(bin, std*fs/1000.0)
time=(numpy.arange(0,bin) - (bin-1.0)/2.0)*1000/fs
#numpy.linspace(-bin/2, bin/2, bin)*1000/fs
pylab.plot(time, kernel,'r')

pylab.show()
