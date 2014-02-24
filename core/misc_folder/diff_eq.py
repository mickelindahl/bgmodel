import numpy
import pylab

tau=100

f=lambda t,b:numpy.exp(-t/tau)+b/tau
f2=lambda t,b, v:v*numpy.exp(-t/tau)+b/tau
t=numpy.arange(2000.)
b=numpy.zeros(2000)

pylab.plot(t,f(t,b))

t0=0

for i in range(len(b)):
    if numpy.random.rand()>0.9:
        b[i]=1.
t_diff=numpy.diff(t)
b=b[0:-1]

l=[]
v=1.
for x,y in zip(t_diff, b):
    v=f2(x,y,v)
    l.append(v) 

pylab.plot(t[0:-1], l)
pylab.show()
