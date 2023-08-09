"""
Created on May 13, 2014

@author: mikael

put in core_old such that import of pylab always is
done like this.
"""

import os

if not os.environ.get('DISPLAY'):
    import matplotlib

    matplotlib.use('Agg')
    print('Switching to use agg')
    import matplotlib.pylab as plt

    plt.ioff()

from matplotlib.pylab import *

import matplotlib.pylab

__doc__ = matplotlib.pylab.__doc__

print('In pylab')