'''
Created on May 13, 2014

@author: mikael
'''
print 'In pylab'
import os
if not os.environ.get('DISPLAY'):
    import matplotlib
    matplotlib.use('Agg')
    
    import matplotlib.pylab as plt
    plt.ioff()

from matplotlib.pylab import *
import matplotlib.pylab
__doc__ = matplotlib.pylab.__doc__

