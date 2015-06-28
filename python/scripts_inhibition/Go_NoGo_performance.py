'''
Created on Jul 27, 2014

@author: mikael
'''
import numpy
import core.plot_settings as ps
import pylab
y=numpy.array([[1041, 2422, 2009, 1980],
               [1061, 2387, 1968, 1856 ],
   [1014, 2167, 1960, 1788],   
   ])
x=numpy.array([8,6,4])
fig, axs=ps.get_figure(n_rows=1, n_cols=1, w=500.0, h=500.0, 
                        fontsize=24, linewidth=4)    
ax=axs[0]
for i in range(4):
    ax.plot(x, y[:,i]/float(numpy.max(y)))

ax.my_set_no_ticks(xticks=3) 
ax.set_ylabel('Rel performance')   
ax.set_xlabel('MSN activated (%)')   
pylab.show()