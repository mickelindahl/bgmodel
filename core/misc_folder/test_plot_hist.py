'''
Created on May 8, 2014

@author: lindahlm
'''

import pylab
import numpy
import pprint
import toolbox.plot_settings as pl
pp=pprint.pprint


_, axs=pl.get_figure(n_rows=1, n_cols=1, w=1000.0, h=800.0, fontsize=16) 


ax=axs[0]#pylab.subplot(111)
r=numpy.random.random(100)
ax.hist(r, **{'histtype':'step', 'label':'test 1'})
r=numpy.random.random(100)
ax.hist(r, **{'histtype':'step', 'label':'test 2', 'linestyle':'dashed'})


h, labels=ax.get_legend_handles_labels()

artist=[]
for hh in h:
    color=hh._edgecolor
    linestyle=hh._linestyle
    obj=pylab.Line2D((0,1),(0,0), color=color, linestyle=linestyle)
    artist.append(obj)
  
ax.legend(artist, labels)

pylab.show()