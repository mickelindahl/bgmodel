"""
Created on May 14, 2015

@author: mikael
"""

from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import DockArea, Dock   
from pyqtgraph.parametertree import Parameter, ParameterTree
from scripts_inhibition import base_neuron
from core import data_to_disk
from core import directories as dr
from core.network import default_params

import datetime as dt
import numpy
import os
import pylab
import pyqtgraph as pg

import pprint
pp=pprint.pprint

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

class Base_model(object):
    def __init__(self, **kw):
        self.app=kw.pop('app')
        self.data_label=''
        self.date_time=dt.datetime.now().strftime('%Y_%m_%d-%H_%M')
        self.h=kw.pop('h', .1) #integration step size
        self.n_state_variables=kw.pop('n_state_variables', 2)
        self.n_history=kw.pop('n_history', 2000)
        self.kw=kw
        self.params=kw.get('params') #qt paramters, se tree parameter class
        self.update_time=kw.pop('update_time', 0)
        self.widgets=[]
        self.script_name=kw.pop('script_name',__file__.split('/')[-1][0:-3])
        self.start_stim=-10000.0
        self.scale_plot=10
        
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
        
        self.x_history=numpy.arange(0,self.n_history, self.scale_plot*self.h)
        self.y_history=numpy.random.rand( self.n_state_variables, self.n_history/(self.scale_plot*self.h))
        self.y_history[0,:]=-70   
        
        # Parameters
        self.p = Parameter.create(name='params', type='group', children=self.params)
        self.p.sigTreeStateChanged.connect(self.change)
        pt = ParameterTree()
        pt.setParameters(self.p, showTop=False)
        pt.setWindowTitle('Parameters')
        
        self.widgets.append(pt)
        
        # Voltage time plot
        w=pg.PlotWidget()
        w.setWindowTitle('Voltage/time')
        
        w.setRange(QtCore.QRectF(0, -90, self.n_history, 100)) 
        w.setLabel('bottom', 'Time', units='s')
        w.plotItem.layout.setContentsMargins(20, 20, 20, 20)
        
        ax=w.plotItem.getAxis('left')
        l=[[ (0.0,'0'), (-30, -30), (-60, -60), (-90, -90) ], ]
        ax.setTicks(l)
        
        w.plotItem.getAxis('bottom').setScale(0.001)
        ax.setLabel('Voltage', units='V')
        ax.setScale(0.001)
#         ax.setWidth(w=2)
        
#         ax.setRange(500,1500)
        color=self.kw.get('curve_uv_color',(0,0,0))
        pen=pg.mkPen(color, width=self.kw.get('curve_uv_width',5))
        self.curve_vt = w.plot(pen=pen)
        self.curve_vt.setData(self.x_history, self.y_history[0,:])
        self.widgets.append(w)
    
    
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.update_time)    
#         self.timer.start(1000.)  
    
        self._init_extra(**kw)        
          
    def _init_extra(self,*kw):
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
        pp(self.get_params_as_dic())
        if hasattr(self, 'change_extra'):
            self.change_extra()
            
    def draw(self):
        self.curve_vt.setData(self.x_history, self.y_history[0,:]) 
            
            
    def f(self, x, y):
        return x*y #dummy function
    
    
    def get_params_as_dic(self):

        d={}
        for ch0 in self.p.children():
            d[ch0.name()]={}
            for ch1 in ch0.children():
                if ch1.value() is None:
                    continue
                
                d[ch0.name()].update({ch1.name():ch1.value()})
        
        return d
     
    def params(self):
        raise NotImplementedError
    
    def pre_history(self):
        pass
        
    def post_history(self):
        pass
    
    def rk4(self):
        
        f=self.f
        k_1 = f(self.x, self.y)
        k_2 = f(self.x+0.5*self.h, self.y+0.5*self.h*k_1)
        k_3 = f(self.x+0.5*self.h, self.y+0.5*self.h*k_2)
        k_4 = f(self.x+self.h, self.y+k_3*self.h)
    
        self.x=self.x+self.h
        self.y = self.y + (1/6.)*(k_1+2*k_2+2*k_3+k_4)*self.h;  #main equation
        
        
        if any(numpy.isnan(self.y)):
            print 'hej'
            
#             y=self.y_history[:,-1]
            print self.y
            self.y[0]=self.V_peak
            self.y[1]=self.y_history[1,-1]
    
    def save(self):

        d=self.get_params_as_dic()
        
        path=dr.HOME_DATA+'/'+self.script_name+'/'+self.date_time
        
        if not os.path.isdir(path):
            data_to_disk.mkdir(path)
        
        l=os.listdir(path)
        n=len(l)
        
        data_to_disk.pickle_save(d, path+'/data_'+str(n)+'_'+self.data_label)
        
    def stimulate(self):
        self.start_stim=self.x
#         self.I_e+=self.I_stim
    
    def update(self):
        rk4=self.rk4
        rk4()

#         print self.x/self.h %/self.scale_plot
        if self.x/self.h %self.scale_plot<1:

            self.pre_history()
            self.update_history()
            self.post_history()
            self.draw()

            self.app.processEvents() 
        else:
            self.pre_history()

            self.post_history()
        
    def update_history(self):
#         self.x_history+=1.
        self.y_history[:,0:-1]=self.y_history[:,1:]
            
#         self.x_history[-1]=self.x
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
        {'name': 'stimulate', 'type': 'group', 'children': [
        {'name': 'I_stim', 'type': 'float','value': 10.0, 'siPrefix': True, 'suffix': 'pA'},
        {'name': 'stim_time', 'type': 'float','value': 100.0, 'siPrefix': True, 'suffix': 'ms'},
        {'name': 'start', 'type': 'action'},
    ]},
        {'name': 'store_state', 'type': 'group', 'children': [
        {'name': 'save', 'type': 'action'},
#          {'name': 'data_label', 'type': 'text', 'value': ''},
    ]},]
    return l
    
def neuron_params(**kw):

    
    l=[
       {'name': 'IV, IF, nullcline, etc', 'type': 'group', 'children': [
        {'name': 'run base_neuron', 'type': 'action'},
        {'name': 'run GP-ST network', 'type': 'action'},
        {'name': 'ahp_I_e', 'type': 'float', 'value': kw['ahp_I_e']},
        {'name': 'ahp_curr', 'type': 'group', 'children': [
          {'name': 'ahp_curr_start', 'type': 'float', 'value': kw['ahp_curr_start']},
          {'name': 'ahp_curr_stop', 'type': 'float', 'value': kw['ahp_curr_stop']},
          {'name': 'ahp_curr_step', 'type': 'float', 'value': kw['ahp_curr_step']}]
         },
        {'name': 'I_E', 'type': 'float', 'value': kw['I_E']},

        {'name': 'if_I_vec', 'type': 'group', 'children': [
          {'name': 'if_I_vec_start', 'type': 'float', 'value': kw['if_I_vec_start']},
          {'name': 'if_I_vec_stop', 'type': 'float', 'value': kw['if_I_vec_stop']},
          {'name': 'if_I_vec_step', 'type': 'float', 'value': kw['if_I_vec_step']}]
         },
        {'name': 'irf_curr', 'type': 'group', 'children': [
          {'name': 'irf_curr_0', 'type': 'float', 'value': kw['irf_curr_0']},
          {'name': 'irf_curr_1', 'type': 'float', 'value': kw['irf_curr_1']},
          {'name': 'irf_curr_2', 'type': 'float', 'value': kw['irf_curr_2']}]
         },                                                            
        {'name': 'iv_I_vec', 'type': 'group', 'children': [
          {'name': 'iv_I_vec_start', 'type': 'float', 'value': kw['iv_I_vec_start']},
          {'name': 'iv_I_vec_stop', 'type': 'float', 'value': kw['iv_I_vec_stop']},
          {'name': 'iv_I_vec_step', 'type': 'float', 'value': kw['iv_I_vec_step']}]
         },
        {'name': 'nc_V', 'type': 'group', 'children': [
          {'name': 'nc_V_start', 'type': 'float', 'value': kw['nc_V_start']},
          {'name': 'nc_V_stop', 'type': 'float', 'value': kw['nc_V_stop']},
          {'name': 'nc_V_step', 'type': 'float', 'value': kw['nc_V_step']}]
         },
        {'name': 'rs_curr', 'type': 'group', 'children': [
          {'name': 'rs_curr_0', 'type': 'float', 'value': kw['rs_curr_0']},
          {'name': 'rs_curr_1', 'type': 'float', 'value': kw['rs_curr_1']},
          {'name': 'rs_curr_2', 'type': 'float', 'value': kw['rs_curr_2']},
          {'name': 'rs_curr_3', 'type': 'float', 'value': kw['rs_curr_3']},
          {'name': 'rs_curr_4', 'type': 'float', 'value': kw['rs_curr_4']},
          {'name': 'rs_curr_5', 'type': 'float', 'value': kw['rs_curr_5']}]
         },
        {'name': 'rs_time', 'type': 'group', 'children': [
          {'name': 'rs_time_0', 'type': 'float', 'value': kw['rs_time_0']},
          {'name': 'rs_time_1', 'type': 'float', 'value': kw['rs_time_1']},
          {'name': 'rs_time_2', 'type': 'float', 'value': kw['rs_time_2']},
          {'name': 'rs_time_3', 'type': 'float', 'value': kw['rs_time_3']},
          {'name': 'rs_time_4', 'type': 'float', 'value': kw['rs_time_4']},
          {'name': 'rs_time_5', 'type': 'float', 'value': kw['rs_time_5']}]
         },        
        {'name': 'rs_I_e', 'type': 'float', 'value': kw['rs_I_e']},]
    }]
    return l
    
class Aeif(Base_model):
    def _init_extra(self,**kw):
        
        self.neuron=kw.get('base_neuron')
        self.neuron_params=kw.get('neuron_params')
        for d in self.neuron_params[0]['children']:
            if 'children' in d.keys():
                for dd in d['children']:
                    self.__dict__[dd['name']] = dd['value']
            else:
                pp(d)
                if 'value' in d.keys():
                    self.__dict__[d['name']] = d['value']
        
        
        self.dyn_xticks=[[100*i,i*0.1] for i in range(0,22,4)]
        
        self.script_name=kw.get('script_name', __file__.split('/')[-1][0:-3])
          
        self.p.param('store_state', 'save').sigActivated.connect(self.save)
        self.p.param('stimulate', 'start').sigActivated.connect(self.stimulate)
   
        
        # Comet for recovery current vs time plot
        w=pg.PlotWidget()
        w.plotItem.getAxis('bottom').setScale(0.001)
        w.plotItem.getAxis('left').setLabel('Current', units='A')
        w.plotItem.getAxis('left').setScale(0.000000000001)
        
        w.setWindowTitle('Recovery current')
        w.setRange(QtCore.QRectF(0, -200, self.n_history, 400)) 
        w.setLabel('bottom', 'Time', units='s')
        w.plotItem.layout.setContentsMargins(20, 20, 20, 20)
        
        color=self.kw.get('curve_uv_color',(0,0,0))
        pen=pg.mkPen(color, width=self.kw.get('curve_uv_width',5))
        self.curve_ut = w.plot(pen=pen)
        self.widgets.append(w) 
        
        
        # Comet for voltage recovery current plot
        w=pg.PlotWidget()
        w.plotItem.getAxis('bottom').setScale(0.001)
        w.plotItem.getAxis('left').setLabel('Recovery current u', units='A')
        w.plotItem.getAxis('left').setScale(0.000000000001)
        w.setWindowTitle('Recovery current/voltage')
        w.setRange(QtCore.QRectF(-100, -50, 100, 200)) 
        w.setLabel('bottom', 'Voltage', units='V')
        w.plotItem.layout.setContentsMargins(10, 10, 10, 10)
        self.widgets.append(w) 

        color=self.kw.get('curve_uv_color',(0,0,0))
        pen=pg.mkPen(color, width=self.kw.get('curve_uv_width',5) )
        self.curve_uv = w.plot(pen=pen)
#         x,y=self.get_nullcline_extreme_points()
#         self.curve_uv_extreme_point = w.plot(x,y, pen=pen)
        color=self.kw.get('curve_uv_color',(0,0,0))
        self.curve_uv_nullcline0 = w.plot(pen=pen)
        self.curve_uv_nullcline1 = w.plot(pen=pen)
        
        color=self.kw.get('curve_uv_color',(255,0,0))
        brush=pg.mkBrush(color=color)
        self.curve_uv_comet= w.plot(symbolBrush= brush, symbol='o', symbolSize=15.)
       
          
        # Parameters base_neuron
        self.p2 = Parameter.create(name='params', type='group', children=self.neuron_params)
        self.p2.sigTreeStateChanged.connect(self.change)
        self.p2.param('IV, IF, nullcline, etc', 'run base_neuron').sigActivated.connect(self.run_neuron_thread)
        self.p2.param('IV, IF, nullcline, etc', 'run GP-ST network').sigActivated.connect(self.run_GP_STN_network_thread)
        pt = ParameterTree()
        pt.setParameters(self.p2, showTop=False)
        pt.setWindowTitle('Parameters')
        
        self.widgets.append(pt)

        # Threshold type
        w=pg.PlotWidget()
#         w.plotItem.getAxis('bottom').setScale(0.001)
        w.plotItem.getAxis('left').setLabel('a/g_L')
#         w.plotItem.getAxis('left').setScale(0.000000000001)
        w.setWindowTitle('Recovery current/voltage')
        w.setRange(QtCore.QRectF(0, 0, 4, 1)) 
        w.setLabel('bottom', 'tau_m/tau_w')
        w.plotItem.layout.setContentsMargins(10, 10, 10, 10)
        self.widgets.append(w)         
                
        color=self.kw.get('curve_uv_color',(0,0,0))
        pen=pg.mkPen(color, width=self.kw.get('curve_uv_width',5) )
        c1,c2=self.get_threshold_oscillation_curves()
        x, y=self.get_threshold_oscilltion_point()
        color=self.kw.get('curve_uv_color',(255,0,0))
        brush=pg.mkBrush(color=color)
        self.curve_oscillation1 = w.plot(c1[0],c1[1],pen=pen)
        self.curve_oscillation2 = w.plot(c2[0],c2[1],pen=pen)
        self.curve_oscillation_points1 = w.plot([x[0]], [y[0]], 
                                               symbolBrush= brush, 
                                               symbol='o', 
                                               symbolSize=25.)
        color=self.kw.get('curve_uv_color',(0,0,255))
        brush=pg.mkBrush(color=color)
        self.curve_oscillation_points2 = w.plot([x[1]], [y[1]], 
                                       symbolBrush= brush, 
                                       symbol='o', 
                                       symbolSize=15.)

        self.draw()
    
    def change_extra(self):
        c1,c2=self.get_threshold_oscillation_curves()
        x, y=self.get_threshold_oscilltion_point()
        self.curve_oscillation1.setData(c1[0],c1[1])
        self.curve_oscillation2.setData(c2[0],c2[1])
        self.curve_oscillation_points1.setData([x[0]], [y[0]])    
        self.curve_oscillation_points2.setData([x[1]], [y[1]]) 
        
    def update_dyn_ticks(self):
#         print self.dyn_xticks
        l1=[[l[0]-1,l[1]] if l[0]>0 else 
            [self.dyn_xticks[-1][0]+400.-1 , self.dyn_xticks[-1][1] +0.4]
            for l in self.dyn_xticks ]
        self.dyn_xticks=sorted(l1, key=lambda x: x[0])

    
    def draw(self):
        self.curve_vt.setData(self.x_history, self.y_history[0,:])
        self.update_dyn_ticks()
                
        for i in range(1,3):
            ax=self.widgets[i].plotItem.getAxis('bottom')
            l=[self.dyn_xticks, ]
            ax.setTicks(l)       

        self.curve_ut.setData(self.x_history, self.y_history[1,:])
        self.curve_uv.setData(self.y_history[0,-100:], self.y_history[1,-100:])
        
        v=numpy.linspace(-90,10,100)
        n0,n1=self.get_nullclines(v)
        self.curve_uv_nullcline0.setData(v, n0)
        self.curve_uv_nullcline1.setData(v, n1)
        
        self.curve_uv_comet.setData([self.
                                     y_history[0,-1]],[self.y_history[1,-1]])

#     def get_nullcline_extreme_points(self):
#         kw={'a_1':self.a_1,
#             'a_2':self.a_2,
#             'Delta_T':self.Delta_T,
#             'g_L':self.g_L,
#             'E_L':self.E_L,
#             'V':numpy.linspace(-100,0, 50),
#             'V_a':self.V_a,
#             'V_th':self.V_th}
#         from core import my_population
#         x,y=my_population.get_nullcline_aeif(**kw)
#         return x,y


    def get_nullclines(self, v):
        
        y=numpy.array([v, numpy.zeros(len(v))])
        out=map(self.f,  [None]*y.shape[1], list(y.transpose())) 
        n0,n1= zip(*out)
        n0=numpy.array(n0)*self.C_m
        n1=numpy.array(n1)*self.tau_w
        return n0,n1
        
    def get_threshold_oscilltion_point(self):
        tau_m=self.C_m/self.g_L
        x=tau_m/self.tau_w
        y1=self.a_1/self.g_L
        y2=self.a_2/self.g_L
        return [[x, x], [y1, y2]]
         
    def get_threshold_oscillation_curves(self):
        x=numpy.linspace(0., 8., 300.)
        y0=x
        y1=(1./4.*x)*(1-1./x)**2
        return [[x,y0],[x,y1]]
            
    def f(self, x, y):
#         print type(y[0])
        if any(numpy.isnan(y)):
            print 'hej'
            
            y=self.y_history[:,-1]
            print y
#         print y
#         if y[1] is numpy.NAN:
#             y[1]=10*5
        dy0=(-self.g_L*(y[0]-self.E_L)+self.g_L*self.Delta_T
            *numpy.exp((y[0]-self.V_th)/self.Delta_T)-y[1]+self.I_e
            +(self.I_stim if self.start_stim+self.stim_time>self.x else 0.0))/self.C_m
        dy1=((self.a_1 if self.V_a>y[0] else self.a_2)*(y[0]-self.V_a)-y[1])/self.tau_w
        
        
        return  numpy.array([dy0, dy1])   
    
    def pre_history(self):
        if self.y[0]>=self.V_peak: 
            self.y_history[0,-1]=self.V_peak
            self.y[0]=self.V_peak
            self.y[1]=self.b+self.y[1]
        
    def post_history(self):
        if self.y[0]==self.V_peak:
            if self.y[1]<0:
                if self.y[0]>=self.V_peak: 
                    self.y[0]=min(self.V_reset+self.y[1]*self.V_reset_slope1, self.V_reset_max_slope1);
            else:
                self.y[0]=min(self.V_reset+self.y[1]*self.V_reset_slope2, self.V_reset_max_slope2); 
  
    def run_neuron_thread(self):
        self.run_neuron()
#         from threading import Thread
#         p = Thread(target=self.run_neuron, args=())
#         p.start()
#         p.close()
  
    def run_neuron(self):

        self._run_neuron()
        pylab.show()
  
    def _run_neuron(self):

        script_name=self.script_name
        
        par=default_params.Inhibition()
        
        params=self.get_params_as_dic()
        d=par.dic['nest'][self.neuron]
        for key, val in params['Model parameters'].items():                              
            if key not in d.keys():
                continue
            d[key]=val

           
              
        kw={
            'ahp_curr':tuple(numpy.arange(self.ahp_curr_start,
                                    self.ahp_curr_stop,
                                    self.ahp_curr_step)), 
            'ahp_I_e':.5,

            
            'I_E':0.0, 
            
            'if_I_vec':tuple(numpy.arange(self.if_I_vec_start,
                                    self.if_I_vec_stop,
                                    self.if_I_vec_step)),
            
            
            'irf_curr':(self.irf_curr_0,
                        self.irf_curr_1,
                        self.irf_curr_2),
            

            'iv_I_vec': tuple(numpy.arange(self.iv_I_vec_start,
                                     self.iv_I_vec_stop,
                                     self.iv_I_vec_step)),
            
            'model':'my_aeif_cond_exp',
            
            'nc_V':tuple(numpy.arange(self.nc_V_start,
                                self.nc_V_stop,
                                self.nc_V_step)),
            
            'rs_curr':(self.rs_curr_0,
                       self.rs_curr_1,
                       self.rs_curr_2,
                       self.rs_curr_3,
                       self.rs_curr_4,
                       self.rs_curr_5),
            
            'rs_time':(self.rs_time_0,
                       self.rs_time_1,
                       self.rs_time_2,
                       self.rs_time_3,
                       self.rs_time_4,
                       self.rs_time_5),

            'rs_I_e':self.rs_I_e,
            
            }
#         pp(d)
        kwhash=kw.copy()
        kwhash.update(d)
        s=str(hash(frozenset(kwhash.items())))
        
        kw['if_params']=d
        kw['iv_params']=d
        kw['nc_params']=d  
        kw['rs_params']=d 
        kw['file_name']= dr.HOME_DATA+'/'+script_name+'/'+'run_neuron'+'_'+s
        kw['file_name_figs']=dr.HOME_DATA+'/fig/'+script_name+'/'+'run_neuron'+'_'+s
    
        from_disk=os.path.isdir( kw['file_name'])
#         main(*args, **kwargs)
        base_neuron.main(from_disk=from_disk,
               kw=kw,
               net='Net_0',
               script_name=script_name,
               setup=base_neuron.Setup(50,20) )
      
    def run_GP_STN_network_thread(self):
        self.run_GP_STN_network()
#         from threading import Thread
#         p = Thread(target=self.run_neuron, args=())
#         p.start()
#         p.close()
  
    def run_GP_STN_network(self):

        self._run_GP_STN_network()
        pylab.show(block=True)
            
    def _run_GP_STN_network(self):
    
    
        script_name=self.script_name
        
        par=default_params.Inhibition()
        
        params=self.get_params_as_dic()
        d=par.dic['nest'][self.neuron]
        
        for key, val in params['Model parameters'].items():                              
            if key not in d.keys():
                continue
            d[key]=val
        pp(d)
        
        
        delay=3.
        kw={    
        'gi_amp':1,
        'gi_n':300,
        
        'gi_st_delay':delay,
        'gi_gi_delay':1.,
        
        'local_num_threads':4,
        
        'sim_time':3500.0, 
        'st_gi_delay':delay,
        'st_amp':0,
        'st_n':100,
        }
        
        kwhash=kw.copy()
        kwhash.update(d)
        s=str(hash(frozenset(kwhash.items())))

        

        kw['file_name']= dr.HOME_DATA+'/'+script_name+'/'+'run_GP_STN_network'+'_'+s
        kw['file_name_figs']=dr.HOME_DATA+'/fig/'+script_name+'/'+'run_GP_STN_network'+'_'+s      
        kw['p_st']=d
    
        from_disk=os.path.isdir( kw['file_name'])*2
        
        from scripts_inhibition import GP_STN_oscillations
        GP_STN_oscillations.main(from_disk=from_disk,
                   kw=kw,
                   net='Net_0',
                   script_name=__file__.split('/')[-1][0:-3],
                   setup=GP_STN_oscillations.Setup(50,20) )
        
          
        
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
        self.win.resize(2000,1900)
        self.win.setWindowTitle(self.title)
        
        docks=[]
        docks.append(Dock("Parameters", size=(500,1000)))
        docks.append(Dock("Voltage", size=(1000,250)))
        docks.append(Dock("Recovery current", size=(1000,250)))
        docks.append(Dock("Phase plane", size=(1000,500)))
        docks.append(Dock("Test set", size=(500,1000)))  
        docks.append(Dock("Analysis", size=(500,350)))  
                      
        area.addDock(docks[4], 'left')      ## place d1 at left edge of dock area (it will fill the whole space since there are no other docks yet)
        area.addDock(docks[0], 'right',docks[4]) 
        area.addDock(docks[1], 'right') 
        area.addDock(docks[2], 'bottom',docks[1]) 
        area.addDock(docks[3], 'bottom',docks[2])
        area.addDock(docks[5], 'bottom',docks[0])                 
        for d,w in zip(docks, self.obj.widgets):
            d.addWidget(w)
            
    def show(self):
        self.win.show()
#         import sys
#         if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

def dummy_neuron_params():
    d={
        'ahp_curr_start':0,
        'ahp_curr_stop':350,
        'ahp_curr_step':20,
        
        'ahp_I_e':.5,
        
        'I_E':0.0, 
        
        'if_I_vec_start':-99,
        'if_I_vec_stop':301,
        'if_I_vec_step':10,
        
        'irf_curr_0':0,
        'irf_curr_1':10,
        'irf_curr_2':40,
                
        'iv_I_vec_start':-200, 
        'iv_I_vec_stop':0,
        'iv_I_vec_step':10,
        
        'model':'my_aeif_cond_exp',
        
        'nc_V_start':-80, 
        'nc_V_stop':-30, 
        'nc_V_step':1,
                 
        'rs_curr_0':-70,
        'rs_curr_1':-70,
        'rs_curr_2':-70,
        'rs_curr_3':-40,
        'rs_curr_4':-70,
        'rs_curr_5':-100,
        
        'rs_time_0':300.,
        'rs_time_1':450.,
        'rs_time_2':600.,
        'rs_time_3':300.,
        'rs_time_4':300.,
        'rs_time_5':300.,
                    
        'rs_I_e':.5,
        }
    return d
def dummy_params():
    d=[{'name': 'Par 1', 
     'type': 'group', 
     'children': 
     [
      {'name': 'dummy1', 'type': 'float', 'value': 2.0},
      {'name': 'dummy2', 'type': 'float', 'value': 0.0},
    ]
     },
       {'name': 'Par 2', 
     'type': 'group', 
     'children': 
     [
      {'name': 'dummy3', 'type': 'float', 'value': 2.0},
      {'name': 'dummy4', 'type': 'float', 'value': 0.0},
    ]
     },
       ]
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
        p = Parameter.create(params=dummy_params())
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
        
    def test_5_get_params_as_dic(self):
        bm=Base_model(**self.kw)
        d1=bm.get_params_as_dic()        
        d2={'Par 1': {'dummy1': 2.0, 'dummy2': 0.0},
            'Par 2': {'dummy3': 2.0, 'dummy4': 0.0}}
        self.assertDictEqual(d1, d2)

class TestAeif(unittest.TestCase):
    def setUp(self):
        self.app=QtGui.QApplication([])
        self.kw={'app':self.app,
                 'base_neuron':'ST',
                 'neuron_params':neuron_params(**dummy_neuron_params()),
                 'params':aeif_params(**dummy_aeif()), 
                 'script_name':__file__.split('/')[-1][0:-3]}
        
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
    
    def test_4_run_neuron(self):
        bm=Aeif(**self.kw)
        bm._run_neuron()
        pylab.show()
        
    def test_5_run_GP_STN_network(self):
        bm=Aeif(**self.kw)
        bm.run_GP_STN_network()
        pylab.show()
        
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
#                     'test_5_get_params_as_dic',
                     
                ],
         TestAeif:[ 
#                     'test_1_f' ,
#                     'test_2_update',
                    'test_3_with_window', # must run by it self
#                       'test_4_run_neuron',
#                     'test_5_run_GP_STN_network',
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




    