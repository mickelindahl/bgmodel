'''
Created on Feb 19, 2015

@author: mikael
'''
import numpy
import pylab
vact=-20
sact=16
f0=lambda v:1/(1+numpy.exp(-0.062*v)/3.57)
f1=lambda v:1/(1+numpy.exp((vact-v)/sact))
v=numpy.arange(-100.,30.)

pylab.plot(v,f0(v))
pylab.plot(v,f1(v))
pylab.show()
