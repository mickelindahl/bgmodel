import numpy
def sigmoid(t, a, b):
    return 1/(1+numpy.exp(-a*(t+b)))



import pylab

x=numpy.linspace(-1, 1, 100)
y=sigmoid(x,20, -0.5)
pylab.plot(x,y)
pylab.show()
