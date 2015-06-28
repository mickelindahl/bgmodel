'''
Created on Jun 18, 2014

@author: mikael
'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax = Axes3D(fig)

for p in  sorted(dir(ax)):
    print p