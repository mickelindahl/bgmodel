'''
Created on Sep 18, 2014

@author: mikael
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

size=10.
x=[1,3]
y=[1,3]
stepx=(x[1]-x[0])/size
stepy=(y[1]-y[0])/size
x1,y1=np.meshgrid(np.linspace(x[0], x[1], size+1),
                np.linspace(y[0], y[1], size+1))
x2,y2=np.meshgrid(np.linspace(x[0]+stepx/2, x[1]-stepx/2, size),
                  np.linspace(y[0]+stepy/2, y[1]-stepy/2, size))
data = np.random.random((10,10))
data2 = np.ma.masked_greater(data, 0.5)

x2=np.ma.array(x2, mask=data<0.5)
y2=np.ma.array(y2, mask=data<0.5)

fig, ax = plt.subplots()
im = ax.pcolor(x1, y1, data, cmap=cm.gray, edgecolors='white', linewidths=1)
fig.colorbar(im)
ax.scatter(x2,y2, color='k', edgecolor='w', s=500, marker=r"$Lisa$")
# ax.patch.set_hatch('x')
ax.set_xlim([1,3])
ax.set_ylim([1,3])
plt.show()
