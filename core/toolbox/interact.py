'''
Created on May 14, 2015

@author: mikael
'''
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import DockArea, Dock   
from pyqtgraph.parametertree import Parameter, ParameterTree

import numpy
import pyqtgraph as pg

import pprint
pp=pprint.pprint

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

class Base_model(object):
    def __init__(self, **kw):
        self.app=kw.pop('app')
        self.h=kw.pop('h', .1) #integration step size
        self.n_state_variables=kw.pop('n_state_variables', 2)
        self.n_history=kw.pop('n_history', 1000)
        self.kw=kw
        self.params=kw.get('params') #qt paramters, se tree parameter class
        self.update_time=kw.pop('update_time', 0)
        self.widgets=[]
        self.script_name=kw.pop('script_name',__file__.split('/')[-1][0:-3])
        self.start_stim=-10000.0
        
        
        for key, value in kw.items():
            self.__dict__[key] = value
            
        for d in self.params[0]['children']:
            self.__dict__[d['name']] = d['value']

        for d in self.params[1]['children']:
            if 'value' in d.keys():
                self.__dict__[d['name']] = d['value']
        
        self.x=self.n_history
        self.y=numpy.zeros(self.n_state_variables)
        self.y[0]=-70
        self.dy=numpy.zeros(self.n_state_variables)
        
        self.x_history=numpy.arange(0,self.n_history, 1.0)
        self.y_history=numpy.random.rand( self.n_state_variables, self.n_history)
        self.y_history[0,:]=-70   
        
        # Parameters
        self.p = Parameter.create(name='params', type='group', children=self.params)
        self.p.param('Stimulate', 'Start').sigActivated.connect(self.stimulate)
#         self.p=Parameter(**self.params)
        self.p.sigTreeStateChanged.connect(self.change)
        pt = ParameterTree()
        pt.setParameters(self.p, showTop=False)
        pt.setWindowTitle('Parameters')
        
        self.widgets.append(pt)
        
        # Voltage time plot
        w=pg.PlotWidget()
        w.setWindowTitle('Voltage/time')
        w.setRange(QtCore.QRectF(0, -90, 1000, 100)) 
        w.setLabel('bottom', 'Index', units='B')
        w.plotItem.layout.setContentsMargins(20, 20, 20, 20)
        
        ax=w.plotItem.getAxis('left')
        l=[[ (0.0,'0'), (-30, -30), (-60, -60) ], ]
        ax.setTicks(l)
#         ax.setRange(500,1500)
        color=self.kw.get('curve_uv_color',(0,0,0))
        pen=pg.mkPen(color, width=self.kw.get('curve_uv_width',2))
        self.curve_vt = w.plot(pen=pen)
        self.curve_vt.setData(self.x_history, self.y_history[0,:])
        self.widgets.append(w)
    
    
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.update_time)    
#         self.timer.start(1000.)  
    
        self._init_extra_widgets(*kw)        
          
    def _init_extra_widgets(self,*kw):
        pass

    def change(self, param, changes):
        '''
        param - parent parameter
        changes - iterator 
        '''
        print("tree changes:")
        for param, change, data in changes:
            '''
            param  - parameter obj
            change - type of change
            value  - new value
            '''
            setattr(self, param.name(), data)
                  
#             path = p.childPath(param)
#             if path is not None:
#                 childName = '.'.join(path)
#             else:
            childName = param.name()
            print('  parameter: %s'% childName)
            print('  change:    %s'% change)
            print('  data:      %s'% str(data))
            print('  ----------')
        
    def draw(self):
        self.curve_vt.setData(self.x_history, self.y_history[0,:]) 
            
    def f(self, x, y):
        return x*y #dummy function
    
    def params(self):
        raise NotImplementedError
    
    def pre_history(self):
        pass
        
    def post_history(self):
        pass
    
    def rk4(self):
        

        k_1 = self.f(self.x, self.y)
        k_2 = self.f(self.x+0.5*self.h, self.y+0.5*self.h*k_1)
        k_3 = self.f(self.x+0.5*self.h, self.y+0.5*self.h*k_2)
        k_4 = self.f(self.x+self.h, self.y+k_3*self.h)
    
        self.x=self.x+self.h
        self.y = self.y + (1/6.)*(k_1+2*k_2+2*k_3+k_4)*self.h;  #main equation
    
    def save(self):
        d=get_params_as_dic()
        path=self.path+'/'+self.script_name
        if not os.path.isdir(path):
            mkdir
        
        
        
    def stimulate(self):
        self.start_stim=self.x
#         self.I_e+=self.I_stim
    
    def update(self):
        rk4=self.rk4
        rk4()

        if self.x-int(self.x)<self.h:

            self.pre_history()
            self.update_history()
            self.post_history()
            self.draw()

            self.app.processEvents() 
        else:
            self.pre_history()

            self.post_history()
        
    def update_history(self):
        self.x_history[0:-1]=self.x_history[1:]
        self.y_history[:,0:-1]=self.y_history[:,1:]
            
        self.x_history[-1]=self.x
        self.y_history[:,-1]=self.y.transpose()

    
def aeif_params(**kw):
    l=[{'name': 'Model parameters', 
         'type': 'group', 
         'children': 
         [
          {'name': 'a_1', 'type': 'float', 'value': kw['a_1']},
          {'name': 'a_2', 'type': 'float', 'value': kw['a_2']},
          {'name': 'b', 'type': 'float', 'value': kw['b']},
          {'name': 'beta_V_a', 'type': 'float', 'value': kw.get('beta_V_a',0.)},
          {'name': 'beta_E_L', 'type': 'float', 'value': kw.get('beta_E_L',0.)},
          {'name': 'C_m', 'type': 'float', 'value': kw['C_m'], 'siPrefix': True, 'suffix': 'pF'},
          {'name': 'Delta_T', 'type': 'float', 'value': kw['Delta_T'], 'siPrefix': True, 'suffix': 'ms'},
          {'name': 'I_e', 'type': 'float', 'value': kw['I_e'], 'siPrefix': True, 'suffix': 'pA'},
          {'name': 'g_L', 'type': 'float', 'value': kw['g_L'], 'siPrefix': True, 'suffix': 'nS'},
          {'name': 'E_L', 'type': 'float', 'value': kw['E_L'], 'siPrefix': True, 'suffix': 'mV'},
          {'name': 'tata_dop', 'type': 'float', 'value': kw['tata_dop']},
          {'name': 'tau_w', 'type': 'float', 'value': kw['tau_w'], 'siPrefix': True, 'suffix': 'ms'},
          {'name': 'V_a', 'type': 'float', 'value': kw['V_a'], 'siPrefix': True, 'suffix': 'mV'},
          {'name': 'V_peak', 'type': 'float', 'value': kw['V_peak'], 'siPrefix': True, 'suffix': 'mV'},
          {'name': 'V_reset', 'type': 'float', 'value': kw['V_reset'], 'siPrefix': True, 'suffix': 'mV'},
          {'name': 'V_reset_slope1', 'type': 'float', 'value': kw.get('V_reset_slope1',0.0), 'siPrefix': True, 'suffix': 'mV'},      
          {'name': 'V_reset_slope2', 'type': 'float', 'value': kw.get('V_reset_slope2',0.0), 'siPrefix': True, 'suffix': 'mV'},      
          {'name': 'V_reset_max_slope1', 'type': 'float', 'value': kw.get('V_reset_slope1',0.0), 'siPrefix': True, 'suffix': 'mV'},      
          {'name': 'V_reset_max_slope2', 'type': 'float', 'value': kw.get('V_reset_max_slope2',0.0), 'siPrefix': True, 'suffix': 'mV'},      
          {'name': 'V_th', 'type': 'float', 'value': kw['V_th'], 'siPrefix': True, 'suffix': 'mV'},
        ]
        },
        {'name': 'Stimulate', 'type': 'group', 'children': [
        {'name': 'I_stim', 'type': 'float','value': 10.0, 'siPrefix': True, 'suffix': 'pA'},
        {'name': 'stim_time', 'type': 'float','value': 100.0, 'siPrefix': True, 'suffix': 'ms'},
        {'name': 'Start', 'type': 'action'},
    ]},
        {'name': 'Store  state', 'type': 'group', 'children': [
        {'name': 'Path', 'type': 'text', 'value': '/home/mikael/results/interact'},
        {'name': 'Save', 'type': 'action'},
    ]},]
    return l
    
class Aeif(Base_model):
    def _init_extra_widgets(self,*kw):
        
        # Comet for recovery current vs time plot
        w=pg.PlotWidget()
        w.setWindowTitle('Recovery current')
        w.setRange(QtCore.QRectF(0, -200, 1000, 400)) 
        w.setLabel('bottom', 'Index', units='B')
        w.plotItem.layout.setContentsMargins(30, 30, 30, 30)
        
        color=self.kw.get('curve_uv_color',(0,0,0))
        pen=pg.mkPen(color, width=self.kw.get('curve_uv_width',2))
        self.curve_ut = w.plot(pen=pen)
        self.widgets.append(w) 
        
        
        # Comet for voltage recovery current plot
        w=pg.PlotWidget()
        w.setWindowTitle('Recovery current/voltage')
        w.setRange(QtCore.QRectF(-100, -200, 130, 400)) 
        w.setLabel('bottom', 'Index', units='B')
        w.plotItem.layout.setContentsMargins(30, 30, 30, 30)
        self.widgets.append(w) 
                
        color=self.kw.get('curve_uv_color',(0,0,0))
        pen=pg.mkPen(color, width=self.kw.get('curve_uv_width',2) )
        self.curve_uv = w.plot(pen=pen)
        self.curve_uv_nullcline0 = w.plot(pen=pen)
        self.curve_uv_nullcline1 = w.plot(pen=pen)
        
        color=self.kw.get('curve_uv_color',(255,0,0))
        brush=pg.mkBrush(color=color)
        self.curve_uv_comet= w.plot(symbolBrush= brush, symbol='o', symbolSize=15.)
        
        self.draw()
              
    def draw(self):
        self.curve_vt.setData(self.x_history, self.y_history[0,:])
        self.widgets[1].setRange(QtCore.QRectF(self.x-1000, -90, 1000, 100)) 
        self.curve_ut.setData(self.x_history, self.y_history[1,:])
        self.widgets[2].setRange(QtCore.QRectF(self.x-1000,-200, 1000, 400)) 
        self.curve_uv.setData(self.y_history[0,-100:], self.y_history[1,-100:])
        
        v=numpy.linspace(-90,10,100)
        n0,n1=self.get_nullclines(v)
        self.curve_uv_nullcline0.setData(v, n0)
        self.curve_uv_nullcline1.setData(v, n1)
        
        self.curve_uv_comet.setData([self.y_history[0,-1]],[self.y_history[1,-1]])

    def get_nullclines(self, v):
        
        y=numpy.array([v, numpy.zeros(len(v))])
        out=map(self.f,  [None]*y.shape[1], list(y.transpose())) 
        n0,n1= zip(*out)
        n0=numpy.array(n0)*self.C_m
        n1=numpy.array(n1)*self.tau_w
        return n0,n1
        
            
    def f(self, x, y):
        dy0=(-self.g_L*(y[0]-self.E_L)+self.g_L*self.Delta_T
            *numpy.exp((y[0]-self.V_th)/self.Delta_T)-y[1]+self.I_e
            +(self.I_stim if self.start_stim+self.stim_time>self.x else 0.0))/self.C_m
        dy1=((self.a_1 if self.V_a>y[0] else self.a_2)*(y[0]-self.V_a)-y[1])/self.tau_w
        
        return  numpy.array([dy0, dy1])   
    
    def pre_history(self):
        if self.y[0]>=self.V_peak: 
            self.y_history[0,-1]=self.V_peak
            self.y[0]=self.V_peak
            self.y[1]=self.b
        
    def post_history(self):
        if self.y[0]==self.V_peak:
            if self.y[1]<0:
                if self.y[0]>=self.V_peak: 
                    self.y[0]=min(self.V_reset+self.y[1]*self.V_reset_slope1, self.V_reset_max_slope1);
            else:
                self.y[0]=min(self.V_reset+self.y[1]*self.V_reset_slope2, self.V_reset_max_slope2); 
          
class Window_base(object):
    def __init__(self, obj):
        self.obj=obj #model object

    def layout(self):
        raise NotImplementedError
          
class Window_aeif(Window_base):

    def __init__(self,**kw):
        self.app = kw.get('app')
        self.obj = kw.get('obj')
        self.title=kw.get('title')
        self.win = QtGui.QMainWindow()

    def layout(self):
        
        area = DockArea()
        self.win.setCentralWidget(area)
        self.win.resize(1500,1200)
        self.win.setWindowTitle(self.title)
        
        docks=[]
        docks.append(Dock("Parameters", size=(500,1000)))
        docks.append(Dock("Voltage", size=(1000,250)))
        docks.append(Dock("Recovery current", size=(1000,250)))
        docks.append(Dock("Phase plane", size=(1000,500)))
        
        area.addDock(docks[0], 'left')      ## place d1 at left edge of dock area (it will fill the whole space since there are no other docks yet)
        area.addDock(docks[1], 'right') 
        area.addDock(docks[2], 'bottom',docks[1]) 
        area.addDock(docks[3], 'bottom',docks[2])
                 
        for d,w in zip(docks, self.obj.widgets):
            d.addWidget(w)
            
    def show(self):
        self.win.show()
#         import sys
#         if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

def dummy_params():
    d={'name': 'Dummy parameters', 
     'type': 'group', 
     'children': 
     [
      {'name': 'dummy1', 'type': 'float', 'value': 2.0},
      {'name': 'dummy2', 'type': 'float', 'value': 0.0},
    ]
     }
    return d
    
def dummy_aeif():
    d={'AMPA_1_E_rev': 0.0,
     'AMPA_1_Tau_decay': 4.0,
     'C_m': 60.0,
     'Delta_T': 16.2,
     'E_L': -80.2,
     'GABAA_1_E_rev': -84.0,
     'GABAA_1_Tau_decay': 8.0,
     'I_e': 6.0,
     'NMDA_1_E_rev': 0.0,
     'NMDA_1_Sact': 16.0,
     'NMDA_1_Tau_decay': 160.0,
     'NMDA_1_Vact': -20.0,
     'V_a': -70.0,
     'V_peak': 15.0,
     'V_reset': -70.0,
     'V_reset_max_slope1': -60.0,
     'V_reset_max_slope2': -70.0,
     'V_reset_slope1': -10.0,
     'V_reset_slope2': 0.0,
     'V_th': -64.0,
     'a_1': 0.3,
     'a_2': 0.0,
     'b': 0.05,
     'beta_I_AMPA_1': -0.45,
     'beta_I_GABAA_1': -0.24,
     'beta_I_NMDA_1': -0.45,
     'g_L': 10.0,
     'tata_dop': 0.0,
     'tau_w': 333.0,
     'type_id': 'my_aeif_cond_exp'}
    return d
   
class Dummy_model():
    def __init__(self):
        self.widgets=[] 
        
        for i in range(4):
            color=(0,0,0)
            pen=pg.mkPen(color, width=2)
            w = pg.PlotWidget(title="Plot {0}".format(i))
            w.plotItem.layout.setContentsMargins(30, 30, 30, 30)
            w.plot(numpy.random.normal(size=100),  pen=pen)
            self.widgets.append(w)            


import unittest
class TestBaseModel(unittest.TestCase):
    def setUp(self):
        self.app=QtGui.QApplication([])
        self.kw={'app':self.app,
                 'params':dummy_params()}
    
        
    def test_1_init(self):
        bm=Base_model(**self.kw)
        
    def test_2_change(self):
        bm=Base_model(**self.kw)
        p = Parameter(**dummy_params())
        p_children=p.children()
        bm.change(p, [[p_children[0], 'value', 2.0]])
    
    def test_3_rk4(self):
        bm=Base_model(**self.kw)
        x=bm.x
        y=bm.y
        bm.rk4()
        print x,y,bm.x,bm.y

    def test_4_update(self):
        bm=Base_model(**self.kw)
        bm.update()

class TestAeif(unittest.TestCase):
    def setUp(self):
        self.app=QtGui.QApplication([])
        self.kw={'app':self.app,
                 'params':aeif_params(**dummy_aeif())}
        
    def test_1_f(self):
        bm=Aeif(**self.kw)
        out=bm.f(1,numpy.array([-75,0]))
        self.assertListEqual([0.60255085873637382,  0.0046846846846846871], list(out))

    def test_2_update(self):
        bm=Aeif(**self.kw)
        bm.update()   

    def test_3_with_window(self):
        bm=Aeif(**self.kw)

        w=Window_aeif(**{'app':self.app,
                         'obj': bm,
                         'title':'Dummy'})
        w.layout()
#         bm.start()
        w.show()
        
class TestWindow(unittest.TestCase):     
    def setUp(self):
        self.app=QtGui.QApplication([])
        self.obj=Dummy_model()

        
    def test_1_layout_aeif(self):
        w=Window_aeif(**{'app':self.app,
                         'obj': self.obj,
                         'title':'Dummy'})
        w.layout()
#         w.show()
#         

        
if __name__ == '__main__':
    d={
       TestBaseModel:[
#                      'test_1_init',
#                      'test_2_change',
#                         'test_3_rk4',
#                         'test_4_update',
                     
                ],
         TestAeif:[ 
#                    'test_1_f' ,
#                     'test_2_update',
                    'test_3_with_window',
                ],     
        TestWindow:[
#                      'test_1_layout_aeif',
                    ],
  
       }
    test_classes_to_run=d
    suite = unittest.TestSuite()
    for test_class, val in  test_classes_to_run.items():
        for test in val:
            suite.addTest(test_class(test))

    unittest.TextTestRunner(verbosity=2).run(suite)




    