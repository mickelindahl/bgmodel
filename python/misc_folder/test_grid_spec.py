'''
Created on Nov 3, 2014

@author: mikael
'''
import matplotlib.gridspec as gridspec
import pylab as plt
from matplotlib.axes import Axes 
from core import my_axes

gs = gridspec.GridSpec(3, 3)

#comma separation does sliceing in a 2d array
# fbefore comma first dimension
# adter second dimension
tmp=gs[0 ,  :]
ax1 = my_axes.convert(plt.subplot(tmp)) #row 0 and all columns
ax1.my_remove_axis(xaxis=False, yaxis=True)
# ax=Axes(plt.figure(), ax1.bbox)
ax2 = plt.subplot(gs[1 , :-1]) #row 1 and 0,1 columns
ax3 = plt.subplot(gs[1: , -1]) #row 1,2 and 2 column
ax4 = plt.subplot(gs[-1 , 0]) #row 2 and col 0
ax5 = plt.subplot(gs[-1 , -2]) #row 2 and col 2

plt.show()