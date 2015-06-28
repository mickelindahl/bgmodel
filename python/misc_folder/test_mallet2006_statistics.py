'''
Created on Jun 4, 2014

@author: mikael
'''
from scipy.stats import chisquare
import numpy, pylab

# Data from fig 2B1
d1=numpy.array([26.4455, 25.6872, 25.3081, 25.782, 25.782, 26.2559,
                26.5403, 25.4976, 24.6445, 25.0237, 24.6445, 24.8341,
                25.4976, 26.7299])

# d1=numpy.array([45.9596,48.0808,48.7879,48.7879,48.7879,48.4343,
#                 44.5455,45.6061, 40.303,43.1313,45.2525,42.0707, 
#                 43.8384, 46.6667])

E_d1=numpy.array(list((d1[:7]+d1[7:])/2)+list((d1[:7]+d1[7:])/2))
print E_d1


# Data from fig 2B2
d2=[16.6825,14.1232, 13.8389, 14.4076, 15.3555, 16.872, 
   20.9479,26.3507,27.2986,27.2986,27.3934,27.2038, 
   25.8768, 21.4218]

print numpy.round(d1,1)
print numpy.round(d2,1)
print numpy.mean(d1)
print numpy.mean(d2)


d1=numpy.round(d1,1)
d2= numpy.round(d2,1)
d1_m =numpy.round(numpy.mean(d1),1)
d2_m =numpy.round(numpy.mean(d2),1)

print chisquare(d1, numpy.ones(14)*d1_m)
print chisquare(d2, numpy.ones(14)*d2_m)

pylab.plot(d1)
pylab.plot(d2)
pylab.ylim([0,70])
pylab.show()



# v=20
# d=list(numpy.ones(v)*14)+list(numpy.ones(v)*24)
# print chisquare(d, numpy.ones(v*2)*numpy.mean(d))
# v=15
# d=list(numpy.ones(v)*14)+list(numpy.ones(v)*24)
# print chisquare(d, numpy.ones(v*2)*numpy.mean(d))
