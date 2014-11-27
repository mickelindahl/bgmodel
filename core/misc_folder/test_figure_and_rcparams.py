'''
Created on Nov 26, 2014

@author: mikael
'''
import pylab

pylab.rcParams['font.size']=20
pylab.figure()
pylab.subplot(111)
pylab.plot([1,2],[1,2])
pylab.title('Title')

pylab.rcParams['font.size']=10
pylab.figure()
pylab.subplot(111)
pylab.plot([1,2],[1,2])
pylab.title('Title2')
pylab.show()