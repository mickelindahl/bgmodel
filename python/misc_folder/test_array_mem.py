'''
Created on Oct 1, 2014

@author: mikael
'''

from time import sleep
from scipy import sparse

import numpy
n=10
l=numpy.random.randint(10, size=10)
# sleep(10)
s=sparse.coo_matrix(sparse.coo_matrix(l))
aa=sparse.coo_matrix(0)
print s.todense()
a=sparse.coo_matrix(l)
l=sparse.hstack([0 ,a])
print l

print l.todense()[0]
print list(numpy.squeeze(numpy.asarray((l.todense()))))
print list(numpy.asarray((l.todense())).ravel())

