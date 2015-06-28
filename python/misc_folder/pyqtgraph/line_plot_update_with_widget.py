'''
Created on May 14, 2015

@author: mikael
'''

#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Update a simple plot as rapidly as possible to measure speed.
"""

## Add path to library (just for examples; you do not need this)
# import initExample


from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
from pyqtgraph.ptime import time
from pyqtgraph.dockarea import DockArea, Dock

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

app = QtGui.QApplication([])
win = QtGui.QMainWindow()
area = DockArea()
win.setCentralWidget(area)
win.resize(1000,500)
win.setWindowTitle('pyqtgraph example: dockarea')

d1 = Dock("Dock 1", size=(500,300))
d2 = Dock("Dock 2", size=(500,300))

area.addDock(d1, 'left')      ## place d1 at left edge of dock area (it will fill the whole space since there are no other docks yet)
area.addDock(d2, 'right')   

for e in sorted(dir(pg.PlotWidget().plotItem)):
    print e
    
# p=pg.plot()
# for e in sorted(dir(p)):
#     print e
    
w=pg.PlotWidget()
w.setWindowTitle('pyqtgraph example: PlotSpeedTest')
w.setRange(QtCore.QRectF(0, -20, 5000, 30)) 
w.setLabel('bottom', 'Index', units='B')

curve = w.plot()

# curve= pg.PlotWidget().plotItem

d1.addWidget(w)
d2.addWidget(w)




#curve.setFillBrush((0, 0, 100, 100))
#curve.setFillLevel(0)

#lr = pg.LinearRegionItem([100, 4900])
#p.addItem(lr)

data = np.random.normal(size=(50,5000))
ptr = 0
lastTime = time()
fps = None
def update():
    global curve, data, ptr, p, lastTime, fps
    curve.setData(data[ptr%10])
    ptr += 1
    now = time()
    dt = now - lastTime
    lastTime = now
    if fps is None:
        fps = 1.0/dt
    else:
        s = np.clip(dt*3., 0, 1)
        fps = fps * (1-s) + (1.0/dt) * s
    w.setTitle('%0.2f fps' % fps)
    app.processEvents()  ## force complete redraw for every plot
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)
    
win.show()

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

