'''
Created on Apr 30, 2015

@author: mikael
'''
import numpy
import pylab
res=13.
interval=999
x=numpy.arange(0.,80*res**2*interval)
y=numpy.sin(x/50*2*numpy.pi)


'''
With 20 Hz oscillation avoid interval and res that is a
multiple of 20, that is 2 or 5 (20=2*2*5

11 and 999 works fine
'''

# pylab.plot(x[0:1000],y[0:1000])

pylab.figure(figsize=(24, 20))
print y[0*interval::interval*res**2]
print x[0*interval::interval*res**2] % 50

for i in range(int(res**2)):
    pylab.subplot(int(res),int(res),i+1).plot(y[i*interval::interval*res**2])
#     pylab.subplot(int(res),int(res),i+1).hist(y[i*interval::interval*res**2])
#     pylab.ylim([-1,1])

pylab.show()
