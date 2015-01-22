# coding:latin
'''
Mikael Lindahl 2010


File:
my_signals

Wrapper for signals module in NeuroTools.  Additional functions can be added 
here. In self.list neurotool object is created through which it can be 
directly accessed 


A matrix of functions to create, manipulate and play with analog signals and
spikes 

Classes
-------
VmList           - AnalogSignalList object used for Vm traces
ConductanceList  - AnalogSignalList object used for conductance traces
CurrentList      - AnalogSignalList object used for current traces
SpikeList        - SpikeList object used for spike trains
'''

from copy import deepcopy
import numpy
import pylab
import scipy.stats
import sys
import unittest
import subprocess

from os.path import expanduser
from toolbox import misc, my_axes
from toolbox import signal_processing as sp
from toolbox.parallelization import map_parallel
# Import StandardPickleFile for saving of spike object
from NeuroTools.io import StandardPickleFile

# import NeuroTools.signals as signals
from NeuroTools import signals
from NeuroTools.signals import ConductanceList
from NeuroTools.signals import CurrentList
from NeuroTools.signals import VmList
from NeuroTools.signals import SpikeList
from NeuroTools.plotting import get_display, set_labels, set_axis_limits

from scipy.sparse import csc_matrix


import pprint
pp=pprint.pprint

class Data_element_base(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            
            if key in ['y_raw']:
                key='_'+key
                if value!=None:
                    value =csc_matrix(value)
            
            self.__dict__[key] = value

    def __repr__(self):
        return self.__class__.__name__
    
    def __str__(self):
        return self.__class__.__name__

    @property
    def y_raw(self):
        v =self._y_raw.todense() #convert from sparse matrix representation
        return numpy.array(v) 
        
class Data_activity_histogram_base(object):
    def plot(self, ax, **k):
           
#         width=(self.x[1]-self.x[0])*0.8        
        if not isinstance(ax,my_axes.MyAxes):
            ax=my_axes.convert(ax)
  
        #h=ax.bar(self.x, self.y, width, yerr=self.y_std, **k)
#         k['histtype']=k.get('histtype','step')
        
        
#         bins=numpy.arange(14)
#         y=reduce(lambda x,y:list(x)+list(y),self.y)
        ax.plot(self.x, self.y, **k)
        
        
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Firing rate (Hz') 
        ax.my_set_no_ticks(xticks=6, yticks=6)
        ax.legend()
         
    def get_activity_histogram_stat(self, **kwargs):
        average=kwargs.get('average',True)
        if not average:
            y=map_parallel(statistical_test, *list([self.y_raw]), **kwargs)
            
        else:
            y=statistical_test(self.y)
        
        return Data_activity_histogram_stat(**{'y':y})
    
def statistical_test(y):
    from scipy.stats import chisquare, binom_test
    y_mean=numpy.mean(y)
    n_obs=len(y)
    
    if y_mean<10:
        l=list(y_mean>y[:n_obs/2])+list(y_mean<y[n_obs/2:])
        success=sum(l)
        failures=n_obs-success
        p=binom_test(numpy.array([success, failures]), 0.5)

    
    else:
        _, p=chisquare(
                        y, 
                        numpy.ones(n_obs)*y_mean,
                        )
    
    return p
    

class Data_activity_histogram(Data_element_base, 
                              Data_activity_histogram_base):
    pass


class Data_activity_histogram_stat_base(object):
    def hist(self, ax,  num=20.0, **k):
        if not isinstance(ax,my_axes.MyAxes):
            ax=my_axes.convert(ax)
            
        k['histtype']=k.get('histtype','step')
        
        
        bins=numpy.linspace(0, 1)
#         y=reduce(lambda x,y:list(x)+list(y),self.y)
        ax.hist(numpy.array(self.y), bins,  **k)
        ax.set_xlim(0, 1)
        ax.set_xlabel('p value') 
        ax.set_ylabel('Count') 
        ax.my_set_no_ticks(xticks=10, yticks=6)
        ax.legend()

class Data_activity_histogram_stat(Data_element_base, 
                                   Data_activity_histogram_stat_base):
    pass

class Data_bar_base(object):
    
    def autolabel(self, ax, rects, top_lable_rotation, top_label_round_off):
        # attach some text labels
        for rect in rects:
            s='{:.'+str(top_label_round_off)+'f}'
            height = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 
                    1.05*height, 
                    s.format(height),
                ha='center', va='bottom', rotation=top_lable_rotation)
    
    def bar(self, ax , **k):
                 
        if not isinstance(ax,my_axes.MyAxes):
            ax=my_axes.convert(ax)
                
            
        N=len(self.y)
        
        ind=numpy.arange(N)
        width=0.8
        colors=k.pop('colors',misc.make_N_colors('jet', N))
        
        if hasattr(self, 'y_std'):
            h=ax.bar(0+ind, self.y, width, yerr=self.y_std, **k )
        else: 
            h=ax.bar(0+ind, self.y, width, **k )
            
        self.autolabel(ax, h)

        for i, b in enumerate(h):
            b.set_color(colors[i])

        ax.set_ylabel('y')
        ax.set_xticks(ind+width/2)

    def bar2(self, ax , **k):
                 
        if not isinstance(ax,my_axes.MyAxes):
            ax=my_axes.convert(ax)
                
        self.y=numpy.array(self.y)
        
        n,m=self.y.shape
        
        ind=numpy.arange(m)
        width=0.8/n

        alphas=numpy.linspace(1,1./n,n)
        colors=k.pop('colors',misc.make_N_colors('jet',m ))
        color_axis=k.pop('color_axis',0)
        alpha=k.pop('alpha',True)
        hatchs=k.pop('hatchs',['']*m)
        top_label=k.pop('top_label',True)
        top_lable_rotation=k.pop('top_lable_rotation',0)
        top_label_round_off=k.pop('top_label_round_off',2)
        H=[]
        for i in range(n):
            if hasattr(self, 'y_std'):
                self.y_std=numpy.array(self.y_std)
#                 h=ax.bar(0+ind, self.y, width, yerr=self.y_std, **k )
                H.append(ax.bar(ind+i*width,self.y[i,:], width, 
                                yerr=self.y_std[i,:], 
                                ecolor='k',
                                 **k ))
            
            else: 

                H.append(ax.bar(ind+i*width,self.y[i,:], width, **k ))
            
        
        for j, h in enumerate(H):
            if top_label:
                self.autolabel(ax, h, top_lable_rotation, top_label_round_off)
            for i, b in enumerate(h):
                if color_axis==0:
                    b.set_color(colors[i])
                if color_axis==1:
                    b.set_color(colors[j])
                b.set_hatch(hatchs[i])
                if alpha:
                    b.set_alpha(alphas[j])
                if k.get('edgecolor', False):
                    b.set_edgecolor(k.get('edgecolor'))

        ax.set_ylabel('y')
        ax.set_xticks(ind+width*n/2)
#         locs, labels = ax.get_xticks()
#         pylab.setp(labels, rotation=rotation)
        
class Data_bar(Data_element_base,Data_bar_base):
    pass

class Data_generic_base(object):
    def plot(self, ax, **k):
                
        if not isinstance(ax,my_axes.MyAxes):
            ax=my_axes.convert(ax)
  
        h=ax.plot(self.x, self.y, **k)
        
        if hasattr(self, 'y_std'):
            color=pylab.getp(h[0], 'color')   
            ax.fill_between(self.x, 
                            self.y-self.y_std,
                            self.y+self.y_std, 
                            facecolor=color, alpha=0.5)  
        
        ax.set_xlabel('x')
        ax.set_ylabel('y') 
        ax.my_set_no_ticks(xticks=6, yticks=6)
        ax.legend()
        


class Data_generic(Data_element_base, Data_generic_base):
    pass  


class Data_phase_diff_base(object):
    
    def hist(self, ax,  num=100.0, **k):
        if not isinstance(ax,my_axes.MyAxes):
            ax=my_axes.convert(ax)
            
        k['histtype']=k.get('histtype','step')
            
        bins=numpy.linspace(-numpy.pi, numpy.pi, num)
        
#         y=reduce(lambda x,y:list(x)+list(y),self.y)

        ax.hist(numpy.array(self.y), bins,  **k)
        ax.set_xlim(-numpy.pi, numpy.pi)
        ax.set_xlabel('Angle (Rad)') 
        ax.set_ylabel('Count') 
        ax.my_set_no_ticks(xticks=10, yticks=6)
        ax.legend()

        
class Data_phase_diff(Data_element_base, Data_phase_diff_base):
    pass


class Data_phases_diff_with_cohere_base(object):
    def hist2(self, ax,  num=300.0, **k):
        if not isinstance(ax,my_axes.MyAxes):
            ax=my_axes.convert(ax)
            
        k['histtype']=k.get('histtype','step')
        k['normed']=1
        k['linestyle']='dashed'
        bins=numpy.linspace(-3*numpy.pi,3* numpy.pi, num)
        y=reduce(lambda x,y:list(x)+list(y),self.y)
        h=ax.hist(numpy.array(self.y.ravel()), bins,  **k)
#         h=ax.bar(self.y_bins[0:-1], self.y, **{'linewith':0})
        color=pylab.getp(h[2][0], 'edgecolor')   
        
        k['linestyle']='solid'
        k['color']=color
        idx=self.idx_sorted[self.coherence[self.idx_sorted]>self.p_conf95]
        ax.hist(numpy.array(self.y[idx,:].ravel()), bins,  **k)
        
#         ax.set_xlim(-numpy.pi, numpy.pi)
        ax.set_xlabel('Angle (Rad)') 
        ax.set_ylabel('Count') 
        ax.my_set_no_ticks(xticks=10, yticks=6)
        ax.legend()
        
    def hist(self, ax,  num=100.0, **k):
        if not isinstance(ax,my_axes.MyAxes):
            ax=my_axes.convert(ax)
        k=deepcopy(k)
#         k['normed']=1
        k['linestyle']=k.pop('linestyle_rest','-')

        rest=k.pop('rest', True)
        p_95=k.pop('p_95', True)
        h=[]
        if rest:
            a=numpy.mean(self.y_bins, axis=0)[0:-1]
            step=numpy.diff(a)[0]
            a=numpy.array([[aa, aa+step] for aa in a]).ravel()
            b=numpy.array([[bb,bb] for bb in numpy.sum(self.y_val, axis=0)],
                          dtype=numpy.float).ravel()
           
            norm=sum(b)*(a[-1]-a[0])/len(a)
            h=ax.plot(a, b/norm, **k)

        
        if p_95:
            k['linestyle']=k.pop('linestyle_p_95','-')
            
            if h:
                color=pylab.getp(h[0], 'color')   
                k['color']=color
            idx=self.idx_sorted[self.coherence[self.idx_sorted]>self.p_conf95]
            y=self.y_val[idx,:]
            y_bins=self.y_bins[idx,:]
            
            a=numpy.mean(y_bins, axis=0)[0:-1]
            step=numpy.diff(a)[0]
            a=numpy.array([[aa, aa+step] for aa in a]).ravel()
            b=numpy.array([[bb,bb] for bb in numpy.sum(y, axis=0)],
                          dtype=numpy.float).ravel()
            
            norm=sum(b)*(a[-1]-a[0])/len(a)
            
            h=ax.plot(a, b/norm, **k)
#         ax.hist(numpy.array(self.y[idx,:].ravel()), bins,  **k)
        
#         ax.set_xlim(-numpy.pi, numpy.pi)
        ax.set_xlabel('Angle (Rad)') 
        ax.set_ylabel('Count') 
        ax.my_set_no_ticks(xticks=10, yticks=6)
        ax.legend()       
class Data_phases_diff_with_cohere(Data_element_base,
                                   Data_phases_diff_with_cohere_base):
    pass

      
class Data_firing_rate_base(object):
    def plot(self, ax, win=100, t_start=0, t_stop=numpy.inf, **k):
                
        if not isinstance(ax,my_axes.MyAxes):
            ax=my_axes.convert(ax)
            
        exp=(self.x>t_start)*(self.x<t_stop)
        x,y=self.x[exp], self.y[exp]
        y=misc.convolve(y, **{'bin_extent':win, 
                              'kernel_type':'triangle',
                              'axis':1})     
        ax.plot(x, y, **k)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Firing rate (spike/s)') 
        ax.my_set_no_ticks(xticks=6, yticks=6)
        ax.legend()

    def get_psd(self, **kwargs):
        y,x=sp.psd(deepcopy(self.y), **kwargs)
        d={'x':x, 'y':y}
        return Data_psd(**d)

    def get_activity_histogram(self, *args, **kwargs):
   
        
        if kwargs.get('average', True):
            y=self.y
            bins=kwargs.get('bins', 14)
            p=kwargs.get('period', 100.0)
            y_mean, y_std, x = get_activity_historgram(y, bins, p)
            d = {'y':y_mean, 'y_std':y_std, 'x':x}
        else:
            
            y=self.y_raw
            bins=[kwargs.get('bins', 14)]*y.shape[0]
            p=[kwargs.get('period', 100.0)]*y.shape[0]
            d=map_parallel(get_activity_historgram,*[y, bins, p])
        
            y_raw=numpy.array(d)[:,0,:]
            y=numpy.mean(y_raw, axis=0)
            y_std=numpy.std(y_raw, axis=0)
            
            x=numpy.array(d)[0,2,:]
            d = {'y':y, 'y_std':y_std, 'y_raw':y_raw,'x':x}
               
        return Data_activity_histogram(**d)




    def get_mean_rate_slices(self, *args, **kwargs):
#          y = self.mean_rate_slices(**kwargs)

        intervals = kwargs.get('intervals', None)
        repetition = kwargs.get('repetition', 1)

#         y = []
#         for start, stop in intervals:
#             kwargs['t_stop'] = stop
#             kwargs['t_start'] = start
#             y_slice=self.y[(self.x>start)*(self.x>stop)]
#             y.append(numpy.mean(y_slice))
            
        l_start, l_stop=zip(*intervals)
        l_x=[self.x]*len(l_start)
        l_y=[self.y]*len(l_start)
        
        args=[l_x, l_y, l_start, l_stop]
        y=map_parallel(wrap_mean_rate_slice, *args )
        y = numpy.array(y)
        
        if not y.shape == (1, ):
            y = numpy.reshape(y, (repetition, len(y) / repetition))
            
        d=get_mean_rate_slices(self.ids, kwargs, y)    

        
        return Data_mean_rate_slices(**d)

def wrap_mean_rate_slice(x, y, start, stop):
    y_slice=y[(x>start)*(x<stop)]
    return numpy.mean(y_slice)
   
class Data_firing_rate(Data_element_base, Data_firing_rate_base):
    pass


class Data_firing_rates_base(object):

    def get_activity_histogram(self, *args, **kwargs):
        y=self.y
        bins=[kwargs.get('bins', 14)]*y.shape[0]
        p=[kwargs.get('period', 100.0)]*y.shape[0]
        
        d=map_parallel(get_activity_historgram,*[y, bins, p])
        
        y_raw=numpy.array(d)[:,0,:]
        y=numpy.mean(y_raw, axis=0)
        y_std=numpy.std(y_raw, axis=0)
        
        x=numpy.array(d)[0,0,:]
        
        d = {'y':y, 'y_std':y_std, 'y_raw':y_raw,'x':x}
        
        return Data_activity_histogram(**d)
        
class Data_firing_rates(Data_element_base, Data_firing_rates_base):
    pass

def get_activity_historgram(y, bins, p):
    n = int(len(y) / p)
    y = y[:n * p]
    y = numpy.reshape(y, [n, p])
    y = numpy.mean(y, axis=0)
    m = int(numpy.ceil(p / bins))
    n_nans = int(m * bins - p)
    j = m - 1
    for _ in range(n_nans):
        y = numpy.insert(y, j, numpy.NaN)
        j += m
    
    assert len(y) == m * bins, 'not equal'
    y = numpy.reshape(y, [bins, m])

    y_mean = scipy.stats.nanmean(y, axis=1)
    y_std = scipy.stats.nanstd(y, axis=1)
    x = numpy.linspace(0, p, bins)

    return y_mean, y_std, x


class Data_fmin_base(object):
    def plot(self, ax, name, **k):
        p,x,y=ax.texts, 0.02,0.9
        if len(p):
            x=p[-1]._x
            y=p[-1]._y-0.1
        s=name+':'+str(self.xopt[-1][0])+' Hz fopt:'+str(self.fopt[-1])[:6]
        p=ax.text( x, y, s, 
                   transform=ax.transAxes, 
            fontsize=pylab.rcParams['font.size']-2, **k)

class Data_fmin(Data_element_base, Data_fmin_base):
    pass

class Data_IF_curve_base(object):
    def plot(self, ax, x=[], part='last', **k):
        if not ax:
            ax=pylab.subplot(111) 
        
        if not x:
            x=numpy.mean(self.curr,axis=1)
       
        if part=='first':isi=self.first
        if part=='mean':isi=self.mean
        if part=='last':isi=self.last
          
        std=numpy.std(isi,axis=1)
        
        m=numpy.mean(isi,axis=1)
        color=pylab.getp(ax.plot(x, m, marker='o', **k)[0], 'color')    
        
        ax.fill_between(x, m-std, m+std, facecolor=color, alpha=0.5)  
        ax.plot(x, isi , **{'color':color})    
        ax.set_xlabel('Current (pA)') 
        ax.set_ylabel('Rate (spike/s)') 
        ax.legend()
    
class Data_IF_curve(Data_element_base, Data_IF_curve_base):
    pass

class Data_isis_base(object):
    def hist(self, ax,**k):
        if not isinstance(ax,my_axes.MyAxes):
            ax=my_axes.convert(ax)
            
        k['histtype']=k.get('histtype','step')
        
        y=reduce(lambda x,y:list(x)+list(y),self.y)
        ax.hist(numpy.array(y), **k)
        ax.set_xlabel('Time (ms)')     
        ax.set_ylabel('Count (#)')
        ax.my_set_no_ticks(xticks=6, yticks=6)
        ax.legend()
       
class Data_isis(Data_element_base, Data_isis_base):
    pass

class Data_IV_curve_base(object):
    def plot(self, ax=None, x=[], **k):
                
        if not isinstance(ax,my_axes.MyAxes):
            ax=my_axes.convert(ax)
        
        if list(x):
            self.x=x 
            
        h=ax.plot(self.x, self.y, **k)
        
        color=pylab.getp(h[0], 'color')   
        ax.fill_between(self.x, 
                        self.y-self.y_std,
                        self.y+self.y_std, 
                        facecolor=color, alpha=0.5)  
        ax.set_xlabel('Current (pA)') 
        ax.set_ylabel('Membrane potential (mV)') 
        ax.legend()
    
class Data_IV_curve(Data_element_base, Data_IV_curve_base):
    pass

class Data_mean_coherence_base(object):
    def plot(self, ax,  x=[], **k):
                
        if not isinstance(ax,my_axes.MyAxes):
            ax=my_axes.convert(ax)
        if list(x):
            self.x=x 
           
        h=ax.plot(self.x, self.y, **k)
        
        if hasattr(self, 'p_conf95'):
            ax.plot(self.x, self.p_conf95, **{'color':'k'}) 
           
        ax.set_xlabel('Frequency (Hz)') 
        ax.set_ylabel('Coherence') 
        ax.set_ylim([0,1])
        ax.my_set_no_ticks(xticks=6, yticks=6)
        ax.legend()
        return h

        
class Data_mean_coherence(Data_element_base, Data_mean_coherence_base):
    pass

class Data_mean_rates_base(object):
    def hist(self, ax, **k):
        if not isinstance(ax,my_axes.MyAxes):
            ax=my_axes.convert(ax)
            
        k['histtype']=k.get('histtype','step')
        
        k['label']=k.get('label','')+' '+str(numpy.mean(self.y))[:4]
        
        ax.hist(numpy.array(self.y), **k)
        ax.set_xlabel('Rate (Hz)')     
        ax.set_ylabel('Count (#)')
        ax.my_set_no_ticks(xticks=6, yticks=6)
        ax.legend()

class Data_mean_rates(Data_element_base, Data_mean_rates_base):
    pass

class Data_mean_rate_parts_base(object):
    def plot(self, ax,  x=[], **k):
                
        if not isinstance(ax,my_axes.MyAxes):
            ax=my_axes.convert(ax)
        if list(x):
            self.x=x 
           
        h=ax.plot(self.x, self.y, **k)
        ax.set_xlabel('Stimuli')
        ax.set_ylabel('Frequency (spike/s)') 
        ax.my_set_no_ticks(xticks=6, yticks=6)
        ax.legend()
        return h
    
    def plot_FF(self, *args, **kwargs):
        ax=args[0]
        h=self.plot(*args, **kwargs)
        color=pylab.getp(h[0], 'color')   
        ax.fill_between(self.x, 
                        self.y-self.y_std,
                        self.y+self.y_std, 
                        facecolor=color, alpha=0.5)
        ax.set_xlabel('Stimuli (spikes/s)')
        
class Data_mean_rate_parts(Data_element_base, Data_mean_rate_parts_base):
    pass

class Data_mean_rate_slices_base(object):
    def plot(self, ax=None, x=[], **k):
                
        if not isinstance(ax,my_axes.MyAxes):
            ax=my_axes.convert(ax)
        
        if list(x):
            self.x=x 
        
        h=ax.plot(self.x.ravel(), self.y.ravel(), **k)

#         color=pylab.getp(h[0], 'color')   
#         ax.fill_between(self.x, 
#                         self.y-self.y_std,
#                         self.y+self.y_std, 
#                         facecolor=color, alpha=0.5)
        
#         if hasattr(self, 'xticklabels'):
#             step=int(len(self.xticklabels)/15+1)
#             ax.set_xticklabels(self.xticklabels[0::step])
#             ax.set_xticks(self.x[0::step])
#             pylab.setp( ax.xaxis.get_majorticklabels(), 
#                         rotation=k.get('rotation',0) )
        
        
#         ax.my_set_no_ticks(xticks=15, yticks=6)
        ax.set_xlabel('Input') 
        ax.set_ylabel('Firing rate (spike/s)') 
#         ax.legend()
    
class Data_mean_rate_slices(Data_element_base, Data_mean_rate_slices_base):
    pass

class Data_psd_base(object):
    def plot(self, ax, **k):
                
        if not isinstance(ax,my_axes.MyAxes):
            ax=my_axes.convert(ax)
  
        ax.plot(self.x[2:], self.y[2:], **k)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Psd') 
        ax.my_set_no_ticks(xticks=6, yticks=6)
        ax.legend()

class Data_psd(Data_element_base, Data_psd_base):
    pass  

class Data_spike_stat_base(object):
    pass

class Data_spike_stat(Data_element_base, Data_spike_stat_base):
    pass

class Data_voltage_traces_base(object):
    def plot(self, ax, id_list=[0], spike_signal=None,  **k):
        
        if not isinstance(ax,my_axes.MyAxes):
            ax=my_axes.convert(ax)
        
        
        for _id in id_list:
            ax.plot(self.x[_id,:], 
                    self.y[_id,:], **k)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Membrane potential (mV)') 
        ax.my_set_no_ticks(xticks=6, yticks=6)
        ax.legend()
    
class Data_voltage_traces(Data_element_base, Data_voltage_traces_base):
    pass

class MyConductanceList(ConductanceList):
    '''
    MyConductanceList(signals, id_list, dt, t_start=0, t_stop=None, dims=None )
    inherit from ConductanceList which has base class AnalogSignalList.  
    
    Arguments:
        signal  - the vector with the data of the AnalogSignal
        dt      - the time step between two data points of the sampled analog signal
        t_start - begining of the signal, in ms.
        t_stop  - end of the SpikeList, in ms. If None, will be inferred from the data
    '''
    
    def __init__(self, signals, id_list, dt, t_start=0, t_stop=None, dims=None ):
        ''' 
        Constructor 
        
        Inherited attributes:
        self.t_start        = float(t_start)
        self.t_stop         = t_stop
        self.dt             = float(dt)
        self.dimensions     = dims
        self.analog_signals = {}
        
        New attributes:
        self.ids = sorted( id_list )     # sorted id list
        
        '''
        
        # About the super() function 
        # (from python programming - Michael Dawson, page 277) 
        # Incorporate the superclass ConductanceList method's functionality. 
        # To add a new attribute ids i need to override the inherited 
        # constructor method from ConductanceList. I also want my new 
        # constructor to create all the attributes from ConductanceList. 
        # This can be done with the function super(). It lets you invoke the 
        # method of a base class(also called a superclass). The first argument 
        # in the function call, 'MyConductanceList', says I want to invoke a 
        # method of the superclass (or base class) of MyConductanceList which 
        # is ConductanceList. The next argument. sef, passes a reference to 
        # the object so that ConductanceList can get to the object and add 
        # its attributes to it. The next part of the statement __init__(
        # signals, id_list, dt, t_start, t_stop, dims) tells python I want to
        # invoke the constructor method of ConductanceList and a want to pass 
        # it the values of signals, id_list, dt, t_start, t_stop and dims.
        
        # Invoke __init__ of base class ConductanceList which has 
        # AnalogSignalList as base class where the method exist.
        super( MyConductanceList, self ).__init__( signals, id_list, dt, 
                                                   t_start, t_stop, dims)
        self._init_extra_attributes()
    
    def _init_extra_attributes(self):
        # Add new attribute
        self.ids = sorted( self.id_list() )     # sorted id list
        
    def my_plot(self,  id_list=None, display=True, kwargs={} ):
        """
        Plot all cells in the AnalogSignalList defined by id_list. Right now
        here to exemplify use of super() on super class method.
        
        Inputs:
            id_list - can be a integer (and then N cells are randomly selected) or a 
                      list of ids. If None, we use all the ids of the SpikeList
            display - axes handle
            kwargs  - dictionary contening extra parameters that will be sent to the plot 
                      function
        
        Examples:
            >> z = subplot(221)
            >> aslist.plot(5, display=z, kwargs={'color':'r'})
        """
        
        # Invoke plot function in ConductanceList
        super(MyConductanceList, self).plot(id_list, None, display, kwargs )
    
    def my_time_slice(self, t_start, t_stop):
        """
        Return a new AnalogSignalList obtained by slicing between t_start and t_stop
        
        Inputs:
            t_start - begining of the new AnalogSignalList, in ms.
            t_stop  - end of the new AnalogSignalList, in ms.
        
        See also
            id_slice
        """
        new_ConductanceList=super( MyConductanceList, self ).time_slice(t_start, t_stop)
        
        new_MyConductanceList=convert_super_to_sub_class(new_ConductanceList, MyConductanceList)
        
        return new_MyConductanceList            
    def my_save(self, userFileName):
        '''
        Save analog list

        Inputs:
            userFileName    - name of file to save
            
        Examples:
            >> userFileName = /home/savename.dat
            >> aslist.save(userFileName)           
        '''
        userFile = StandardPickleFile( userFileName )      # create user file 
       
        # Invoke save function of base class ConductanceList which has 
        # AnalogSignalList as base class where the method exist.
        super(MyConductanceList, self).save( userFile )
                  
class MyCurrentList(CurrentList):
    ''' 
    MyCurrentList(signals, id_list, dt, t_start=0, t_stop=None, dims=None )
    inherit from CurrentList which has base class AnalogSignalList.
    
    Arguments:
        signal  - the vector with the data of the AnalogSignal
        dt      - the time step between two data points of the sampled analog signal
        t_start - begining of the signal, in ms.
        t_stop  - end of the SpikeList, in ms. If None, will be inferred from the data
    '''  
    
    def __init__( self, signals, id_list, dt, t_start=0, t_stop=None, dims=None ):
        ''' 
        Constructor 
        
        Inherited attributes:
        self.t_start        = float(t_start)
        self.t_stop         = t_stop
        self.dt             = float(dt)
        self.dimensions     = dims
        self.analog_signals = {}
        
        New attributes:
        self.ids = sorted( id_list )     # sorted id list
        
        '''
        
        # Invoke __init__ of base class ConductanceList which has 
        # AnalogSignalList as base class where the method exist.
        super( MyCurrentList, self ).__init__( signals, id_list, dt, 
                                                   t_start, t_stop, dims)
        self._init_extra_attributes()
    
    def _init_extra_attributes(self):
        # Add new attribute
        self.ids = sorted( self.id_list() )     # sorted id list
    
    def my_plot(self, id_list = None, display = True, kwargs = {} ):
        """
        Plot all cells in the AnalogSignalList defined by id_list
        
        Inputs:
            id_list - can be a integer (and then N cells are randomly selected) or a 
                      list of ids. If None, we use all the ids of the SpikeList
            display - if True, a new figure is created. Could also be a subplot
            kwargs  - dictionary contening extra parameters that will be sent to the plot 
                      function
        
        Examples:
            >> z = subplot(221)
            >> aslist.plot(5, display=z, kwargs={'color':'r'})
        """
        
        # Invoke plot function in ConductanceList
        super(MyCurrentList, self).plot(id_list, None, display, kwargs )

    def my_time_slice(self, t_start, t_stop):
        """
        Return a new AnalogSignalList obtained by slicing between t_start and t_stop
        
        Inputs:
            t_start - begining of the new AnalogSignalList, in ms.
            t_stop  - end of the new AnalogSignalList, in ms.
        
        See also
            id_slice
        """
        new_CurrentList=super( MyCurrentList, self ).time_slice(t_start, t_stop)
        
        new_MyCurrentList=convert_super_to_sub_class(new_CurrentList, MyCurrentList)
        
        return new_MyCurrentList
        
def my_save(self, userFileName):
        '''
        Save analog list

        Inputs:
            userFileName    - name of file to save
            
        Examples:
            >> userFileName = /home/savename.dat
            >> aslist.save(userFileName)           
        '''
        userFile = StandardPickleFile( userFileName )      # create user file 
       
        # Invoke save function of base class ConductanceList which has 
        # AnalogSignalList as base class where the method exist.
        super(MyCurrentList, self).save( userFile )
         
class MyVmList(VmList):
    ''' 
    MyVmList(signals, id_list, dt, t_start=0, t_stop=None, dims=None )
    inherit from VmList which has base class AnalogSignalList.
    
    Arguments:
        signal  - the vector with the data of the AnalogSignal
        dt      - the time step between two data points of the sampled analog signal
        t_start - begining of the signal, in ms.
        t_stop  - end of the SpikeList, in ms. If None, will be inferred from the data
    '''  
    
    def __init__(self, signals, id_list, dt, t_start=0, t_stop=None, dims=None ):
        ''' 
        Constructor 
        
        Inherited attributes:
        self.t_start        = float(t_start)
        self.t_stop         = t_stop
        self.dt             = float(dt)
        self.dimensions     = dims
        self.analog_signals = {}
        
        New attributes:
        self.ids = sorted( id_list )     # sorted id list
        
        '''
        
        # Invoke __init__ of base class ConductanceList which has 
        # AnalogSignalList as base class where the method exist.
        super( MyVmList, self ).__init__( signals, id_list, dt, 
                                                   t_start, t_stop, dims)
        self._init_extra_attributes()
    
    def _init_extra_attributes(self):
        # Add new attribute
        self.ids = sorted( self.id_list() )     # sorted id list

    def __repr__(self):
        return self.__class__.__name__+':'+str(self.id_list()) 
    

    def Factory_voltage_traces(self, normalized=False, **kwargs):
        t,v=[], []
        if 'spike_signal' in kwargs.keys():
            spike_signal=kwargs['spike_signal']
            self.my_set_spike_peak( 15, spkSignal= spike_signal ) 
            
        for i in self.id_list():
            analog_signal=self.analog_signals[i]
            
 
            time_axis=numpy.linspace(self.t_start,
                                         self.t_stop,
                                         self.signal_length)
            t.append(time_axis)
            v.append(analog_signal.signal)
       
        x= numpy.array(t)
        y=numpy.array(v)
        if y.shape!=x.shape:
            raise
            
        d= {'ids':self.id_list(), 
            'x':numpy.array(t),
            'y':numpy.array(v)}
        
        return Data_voltage_traces(**d)
    
    def id_list(self):
        """ 
        OBS in SpikeList this is a property put not for
        VmList
        
        Return the list of all the cells ids contained in the
        SpikeList object
        
        Examples
            >> spklist.id_list
                [0,1,2,3,....,9999]
        """
        #if not numpy:
        import numpy # Needed when this method is called by __del__,
                     # Some how numpy reference is lost   
        #id_list=numpy.array(self.spiketrains.keys(), int)
        id_list=super( MyVmList, self ).id_list()
        id_list=numpy.sort(id_list)
        return id_list

    def id_slice(self, id_list):
        """ Slice by ids
        """
        new_SpikeList=super( MyVmList, self ).id_slice(id_list)
        
        new_MyVmList=convert_super_to_sub_class(new_SpikeList, MyVmList)
        
        return new_MyVmList

    
    def get_voltage_traces(self, *args, **kwargs):
        return self.Factory_voltage_traces(*args, **kwargs)
        
    def merge(self, analog_signals):
        """
        """

        for _id, analog_signal in analog_signals.analog_signals.items():
            if _id in self.id_list():
                
                s1=self.analog_signals[_id].signal
                s2=analog_signal.signal
                if self.t_start< analog_signal.t_start:
                    s3=numpy.append(s1,s2)
                    t_start=self.t_start
                    t_stop=analog_signal.t_stop
                else:
                    t_start=analog_signal.t_start
                    t_stop=self.t_stop
                    s3=numpy.append(s2,s1)
                    
                self.analog_signals[_id].signal=s3
                self.analog_signals[_id].t_start=t_start
                self.analog_signals[_id].t_stop=t_stop         
#                 print s3.shape
            else:
                self.append(_id, analog_signal)
   
   
        self.t_start     = min(self.t_start, analog_signal.t_start)
        self.t_stop      = max(self.t_stop, analog_signal.t_stop)
        
        if self.t_start!=analog_signal.t_start:
            self.signal_length=(self.signal_length
                                + analog_signals.signal_length)

    def my_plot(self, id_list = None, display = True, kwargs = {} ):
        """
        Plot all cells in the AnalogSignalList defined by id_list
        
        Inputs:
            id_list - can be a integer (and then N cells are randomly selected) or a 
                      list of ids. If None, we use all the ids of the SpikeList
            display - if True, a new figure is created. Could also be a subplot
            kwargs  - dictionary contening extra parameters that will be sent to the plot 
                      function
        
        Examples:
            >> z = subplot(221)
            >> aslist.plot(5, display=z, kwargs={'color':'r'})
        """
        
        # Invoke plot function in ConductanceList
        super(MyVmList, self).plot(id_list, None, display, kwargs )

    def my_time_slice(self, t_start, t_stop):
        """
        Return a new AnalogSignalList obtained by slicing between t_start and t_stop
        
        Inputs:
            t_start - begining of the new AnalogSignalList, in ms.
            t_stop  - end of the new AnalogSignalList, in ms.
        
        See also
            id_slice
        """
        
        #new_VmList = VmList(self.raw_data(), self.id_list(), self.dt, t_start, t_stop, self.dimensions)
        #for id in self.id_list():
        #    new_VmList[id]=self.analog_signals[id].time_slice(t_start, t_stop)
        
        #return new_AnalogSignalList
        
        
        new_VmList=super( MyVmList, self ).time_slice(t_start, t_stop)
        
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        new_MyVmList=convert_super_to_sub_class(new_VmList, MyVmList)
        
        return new_MyVmList
                   
    
    def my_raw_data_id_order(self): 
        '''
        Return data matrix with raw data for each id on row ordered in descending 
        id order. 
        '''    

        values = numpy.concatenate( [self.analog_signals[id].signal for id in self.ids]) 
        #ids    = numpy.concatenate( [id*numpy.ones( len( obj.analog_signals[id].signal ), int ) for id in self.ids] ) 

        data = values.reshape( len( self.ids ),-1 )
        
        return data
    
    def my_save(self, userFileName):
        '''
        Save analog list

        Inputs:
            userFileName    - name of file to save
            
        Examples:
            >> userFileName = /home/savename.dat
            >> aslist.save(userFileName)           
        '''
        userFile = StandardPickleFile( userFileName )      # create user file 
       
        # Invoke save function of base class ConductanceList which has 
        # AnalogSignalList as base class where the method exist.
        super(MyVmList, self).save( userFile )

    def my_set_spike_peak(self, peak, spkSignal, start=0):
        
        for id in self.id_list():
            spikes=numpy.round((spkSignal[id].spike_times-spkSignal.t_start)/self.dt)#-start
            spikes=numpy.array(spikes, int)
            # t_start_voltage=self.analog_signals[id].t_start
            self.time_axis()
            if len(spikes):
                first_spike=spikes[0]
                r=numpy.arange(int(-20/self.dt+first_spike),
                                     int(20/self.dt+first_spike))
                
                r=r[r<len(self.analog_signals[id].signal)]
                
                
                amax=self.analog_signals[id].signal[r].argmax()
                shift=first_spike-r[amax]
                
                signal=self.analog_signals[id].signal
                idx=numpy.array(spikes-shift)
                #idx=spikes
                idx=idx[idx<len(signal)]
                signal[idx] =peak
#                 print 


    def my_image_plot(self, display = None, kwargs = {}):
        
        if not display: ax = pylab.axes()
        else:           ax = display

        
        data_Vm = numpy.array( self.raw_data() )
        data_Vm = numpy.array( zip( *data_Vm ) )
        data_Vm = data_Vm[ 0 ].reshape( len( self.ids ), -1 )  # Transpose
        image = ax.imshow( data_Vm, origin = 'lower', **kwargs)
        ax.set_xlabel('Membrane potential (mV)')
        ax.set_ylabel('Neuron #')  
        ax.set_aspect(aspect='auto')
        return image

    def plot(self, ax=[], id_list=None, v_thresh=None, display=True, **kwargs):
        """
        Plot all cells in the AnalogSignalList defined by id_list
        
        Inputs:
            id_list - can be a integer (and then N cells are randomly selected) or a 
                      list of ids. If None, we use all the ids of the SpikeList
            v_thresh- For graphical purpose, plot a spike when Vm > V_thresh. If None, 
                      just plot the raw Vm
            display - if True, a new figure is created. Could also be a subplot
            kwargs  - dictionary contening extra parameters that will be sent to the plot 
                      function
        
        Examples:
            >> z = subplot(221)
            >> aslist.plot(5, v_thresh = -50, display=z, kwargs={'color':'r'})
        """
        if 'spike_signal' in kwargs.keys():
            spike_signal=kwargs['spike_signal']
            self.my_set_spike_peak( 15, spkSignal= spike_signal )
            del kwargs['spike_signal'] 
        
        if id_list:
            id_list=list(self.id_list()[id_list])
            
        if not ax:
            ax=pylab.subplot(111)
            
        subplot=ax
        id_list   = self._AnalogSignalList__sub_id_list(id_list)
        time_axis = self.time_axis()  
        xlabel = "Time (ms)"
        ylabel = "Membrane Potential (mV)"
        set_labels(subplot, xlabel, ylabel)
        for id in id_list:
            to_be_plot = self.analog_signals[id].signal
            if v_thresh is not None:
                to_be_plot = pylab.where(to_be_plot>=v_thresh-0.02, v_thresh+0.5, to_be_plot)
            if len(time_axis) > len(to_be_plot):
                time_axis = time_axis[:-1]
            if len(to_be_plot) > len(time_axis):
                to_be_plot = to_be_plot[:-1]
                
            if len(time_axis)!=len(to_be_plot):
                time_axis=numpy.linspace(time_axis[0],time_axis[-1], 
                                         len(to_be_plot))
                   
            
            subplot.plot(time_axis, to_be_plot, **kwargs)
            subplot.hold(1)       
#             
#     def time_axis(self, normalized=False):
#         """
#         Return the time axis of the AnalogSignal
#         """
#         return numpy.linspace(self.t_stop,self.t_stop, self.signal_length)
# #         return numpy.arange(self.t_start-norm, self.t_stop-norm, self.dt) 
#          
class MySpikeList(SpikeList):
    """
    MySpikeList(spikes, id_list, t_start=None, t_stop=None, dims=None)
    
    Inherits from SpikeList
    
    Return a SpikeList object which will be a list of SpikeTrain objects.

    Inputs:
        spikes  - a list of (id,time) tuples (id being in id_list)
        id_list - the list of the ids of all recorded cells (needed for silent cells)
        t_start - begining of the SpikeList, in ms. If None, will be infered from the data
        t_stop  - end of the SpikeList, in ms. If None, will be infered from the data
        dims    - dimensions of the recorded population, if not 1D population
    
    t_start and t_stop are shared for all SpikeTrains object within the SpikeList
    
    Examples:
        >> sl = SpikeList([(0, 0.1), (1, 0.1), (0, 0.2)], range(2))
        >> type( sl[0] )
            <type SpikeTrain>
    
    See also
        load_spikelist
    """
    
    def __init__(self, spikes, id_list, t_start=None, t_stop=None, 
                 dims=None):

        # Invoke __init__ of base class ConductanceList which has 
        # AnalogSignalList as base class where the method exist.
        super( MySpikeList, self ).__init__( spikes, id_list, t_start, t_stop, dims)
        self._init_extra_attributes()
        
    def _init_extra_attributes(self):
        # Add new attribute
        
        self.ids = sorted( self.id_list )     # sorted id list
    
    def __getstate__(self):
        #print '__getstate__ executed'
        return self.__dict__
    
    def __setstate__(self, d):
        #print '__setstate__ executed'
        self.__dict__ = d   
    
    def __del__(self):
        for id in self.id_list:
            del self.spiketrains[id]
    
    
    def __repr__(self):
        return (self.__class__.__name__+':'+str(self.id_list)
                +' '+str(self.t_start)+'-'+str(self.t_stop)) 


    def Factory_firing_rate(self,*args, **kwargs):
        
         
        time_bin=kwargs.pop('time_bin', 1 )
         
        t_start=kwargs.get('t_start', None)
        t_stop=kwargs.get('t_stop', None)
        
        
        x=self.time_axis_centerd(time_bin) 
        y=self.firing_rate(time_bin, **kwargs)
        
        if kwargs.get('average', True):
            if t_start: y=y[x>t_start]
            if t_stop: y=y[x<t_stop]        
            y_raw=None    
        else:
            if t_start:y=y[:,x>t_start]
            if t_stop:y=y[:, x<t_stop]
            y_raw=y
            y=numpy.mean(y, axis=0)
        
        if t_start: x=x[x>t_start]
        if t_stop: x=x[x<t_stop]
        
        
        d={'ids':self.id_list, 'x':x,  'y':y, 'y_raw':y_raw}    
        return Data_firing_rate(**d)


    def Factory_phase_diff(self, *args, **kwargs):
        '''
        Returns the phase of the population firing rate filters in the band
        lowcut to highcut. 
        '''
        fs=kwargs.get('fs')
        time_bin=int(1000/fs)
        other=kwargs.get('other', None)
        
        assert other!=None, 'need to provide other spike list'
        
        signal1=self.firing_rate(time_bin, average=True, **kwargs)
        signal2=other.firing_rate(time_bin, average=True, **kwargs)        
        
#         args=[lowcut, highcut, order, fs]
        y=sp.phase_diff(signal1, signal2, kwargs)
 
        d= {'ids1':self.id_list,
            'ids2':other.id_list,
            'x':self.time_axis_centerd(time_bin) , 
            'y':y}

        return Data_phase_diff(**d)


    def Factory_phases_diff_with_cohere(self, *args, **kwargs):
        '''
        Returns the phase of the population firing rate filters in the band
        lowcut to highcut. 
        '''
        fs=kwargs.get('fs')
        low=kwargs.get('lowcut', 0.5)
        high=kwargs.get('highcut', 1.5)
        time_bin=1000./fs
        other=kwargs.get('other', None)
        sample=kwargs.get('sample',10)
        
        assert other!=None, 'need to provide other spike list'
        
        ids1, ids2=shuffle(*[self.id_list, other.id_list],
                           **{'sample':sample})
               
        sl1=self.id_slice(ids1)
        sl2=other.id_slice(ids2)
        
        signals1=sl1.firing_rate(time_bin, average=False, **kwargs)
        signals2=sl2.firing_rate(time_bin, average=False, **kwargs)       
        
#         args=[lowcut, highcut, order, fs]

        y=sp.phases_diff(signals1, signals2, **kwargs)
        vals=[]
        bins=[]
        bins0=numpy.linspace(3*-numpy.pi, 3*numpy.pi, 3*100)
        for yy in y:
            val, bin=numpy.histogram(yy, bins0)
            vals.append(val)
            bins.append(bin)
            
#             hg.append(numpy.histogram(yy, 100))
        
        y=numpy.array(y, dtype=numpy.float16)
#         y=hg
        x2, y2=sp.coherences(signals1, signals2, **kwargs)
        
        idx, v =sp.sort_coherences(x2, y2, low, high)
 
        L=float(len(signals1[0]))/kwargs.get('NFFT')
        p_conf95=numpy.ones(len(x2))*(1-0.05**(1/(L-1)))  
 
        if not kwargs.get('full_data', False):
            y=None
        
        d= {'ids1':self.id_list,
            'ids2':other.id_list,
            'x':self.time_axis_centerd(time_bin) , 
            'y':y,
            'y_val':numpy.array(vals),
            'y_bins':numpy.array(bins),
            'coherence':v,
            'idx_sorted':idx,
            'p_conf95':p_conf95}

        return Data_phases_diff_with_cohere(**d)



    def Factory_isis(self, *args, **kwargs):
        run=kwargs.get('run',1)
        y=numpy.array(self.isi(), dtype=object)
        x=numpy.array([run]*y.shape[0])    
        d={'ids':self.id_list,
           'x':x,
           'y':y}
        return Data_isis(**d)

    def Factory_mean_coherence(self, *args, **kwargs):
        fs=kwargs.get('fs',1000.0)
        kwargs['fs']=fs
        other=kwargs.get('other', None)
        sample=kwargs.get('sample',10)
        
        assert other!=None, 'need to provide other spike list'
        
        time_bin=1000./fs
        
        ids1, ids2=shuffle(*[self.id_list, other.id_list],
                           **{'sample':sample})
               
        sl1=self.id_slice(ids1)
        sl2=other.id_slice(ids2)
        
        signals1=sl1.firing_rate(time_bin, average=False, **kwargs)
        signals2=sl2.firing_rate(time_bin, average=False, **kwargs) 
        
                
        x, y=sp.mean_coherence(signals1, signals2, **kwargs)

        L=float(len(signals1[0])/kwargs.get('NFFT'))
        p_conf95=numpy.ones(len(x))*(1-0.05**(1/(L-1)))  
        
        d={'ids1':ids1,
            'ids2':ids2, 
            'x':x, 
            'y':y,
            'p_conf95':p_conf95} 
        
        
        return Data_mean_coherence(**d)       
    
    
        
    
    def Factory_mean_rates(self, *args, **kwargs):
        run=kwargs.get('run',1)
        y=numpy.array(self.mean_rates(**kwargs)) 
        x=numpy.ones(y.shape)*run
        d= {'ids':self.id_list,  'y':y, 'x':x} 
        return Data_mean_rates(**d)




    def Factory_mean_rate_slices(self, *args, **kwargs):
        y = self.mean_rate_slices(**kwargs)
        
        d = get_mean_rate_slices(self.id_list, kwargs, y)
        
        return Data_mean_rate_slices(**d) 

    def Factory_psd(self, *args, **kwargs):
        
        NFFT=kwargs.get('NFFT', 256)
        fs=kwargs.get('fs',1000)
        noverlap=kwargs.get('noverlap',int(NFFT/2))
        time_bin=int(1000/fs)
        
        signal=self.firing_rate(time_bin, average=True, **kwargs) 
        y,x=sp.psd(signal, **kwargs)
        d= {'ids':self.id_list,
             'x':x , 
             'y':y}
        return Data_psd(**d)
    
        
    def Factory_spike_stat(self, *args, **kwargs):
        d={'rates':{},'isi':{}, 'cv_isi':{}}
        d['rates']['mean']=self.mean_rate(**kwargs)
        d['rates']['std']=self.mean_rate_std(**kwargs)
        d['rates']['SEM']=d['rates']['std']/numpy.sqrt(len(self.id_list))
        d['rates']['CV']=d['rates']['std']/d['rates']['mean']
        
        isi=numpy.concatenate((self.isi()))
        
        d['isi']['raw']=self.isi()
        d['isi']['mean']=numpy.mean(isi,axis=0)
        d['isi']['1000/mean']=1000./numpy.mean(isi,axis=0)
        d['isi']['std']=numpy.std(isi,axis=0)
        d['isi']['CV']=d['isi']['std']/d['isi']['mean']
        
        cv_isi=numpy.array(self.cv_isi())
        d['cv_isi']['raw']=cv_isi
        d['cv_isi']['mean']=scipy.stats.nanmean(cv_isi,axis=0)
        d['cv_isi']['std']=scipy.stats.nanstd(cv_isi,axis=0)
        d['cv_isi']['SEM']=d['cv_isi']['std']/numpy.sqrt(len(self.id_list))
        d['cv_isi']['CV']=d['cv_isi']['std']/d['cv_isi']['mean']
        
        return Data_spike_stat(**d)


    @property
    def id_list(self):
        """ 
        Return the list of all the cells ids contained in the
        SpikeList object
        
        Examples
            >> spklist.id_list
                [0,1,2,3,....,9999]
        """
        #if not numpy:
        import numpy # Needed when this method is called by __del__,
                     # Some how numpy reference is lost   
        id_list=numpy.array(self.spiketrains.keys(), int)
        #id_list=super( MySpikeList, self ).id_list
        id_list=numpy.sort(id_list)
        return id_list
    
            
    def convert2bin(self, id, start, stop, res=1, clip=0):    
        '''
        Convert a spike train from a time stamp representation to a 
        binned spike train
        
        Inputs:
            id       - id to vector with time stamps
            start    - first spike time
            stop     - last spike time
            clip     . clipping if more than one spike falls in abind
            
        Returns:
            st_binned  - vector that contains resulting binned vetor structure
            st_times   - vector of the resulting corresponding time stamps of
                         each bin
        
        remark: Spikes before start and stop are ignored
        '''
        
        st=self.spiketrains[id].spike_times
        
        output=numpy.zeros(numpy.ceil( (stop-start)/res) + 1)
        
        validspikes=st[(st>start)*(st<stop)]
        
        if len(validspikes)!=0:
            if clip:
                for j in validspikes:
                    output[numpy.int_(numpy.round( (j-start)/res) )]=1
            else:
                for j in validspikes:
                    output[numpy.int_(numpy.round( (j-start)/res ))]+=1
                
        return output
    



    def spike_histogram(self, time_bin, normalized=True, binary=False, display=False, **kwargs):
        """
        Generate an array with all the spike_histograms of all the SpikeTrains
        objects within the SpikeList.
          
        Inputs:
            time_bin   - the time bin used to gather the data
            normalized - if True, the histogram are in Hz (spikes/second), otherwise they are
                         in spikes/bin
            display    - if True, a new figure is created. Could also be a subplot. The averaged
                         spike_histogram over the whole population is then plotted
            binary     - if True, a binary matrix of 0/1 is returned
            kwargs     - dictionary contening extra parameters that will be sent to the plot 
                         function
          
        See also
            firing_rate, time_axis
        """
         
        nbins      = self.time_axis(time_bin)
#         N          = len(self)
#         M          = len(nbins)
#         spike_hist=[]
#         if newnum:
#             M -= 1
#         if binary:
#             spike_hist = numpy.zeros((N, M), numpy.int)
#         else:
#             spike_hist = numpy.zeros((N, M), numpy.float32)
#         subplot    = get_display(display)
#  
#         def fun(): 
        t_stop=kwargs.get('t_stop', numpy.Inf)
 
        n=len(self.id_list)
        args=[[self.spiketrains[_id].spike_times[self.spiketrains[_id].spike_times<t_stop]  
               for _id in self.id_list],
              [time_bin]*n, 
              [normalized]*n,
              [binary]*n, 
              [nbins]*n, 
              self.id_list]
        
        spike_hist=map_parallel(spike_histogram_fun, *args, 
                                **{'local_num_threads': kwargs.get('local_num_threads',1)})
        
#         out=numpy.array(spike_hist, dtype=numpy.float32)
#         print 'hej', len(self.ids),kwargs.get('local_num_threads',1), out.shape
#         for idx, id in enumerate(self.id_list):
#  
#             self.fun(time_bin, normalized, binary, nbins, spike_hist, id)
          
        return numpy.array(spike_hist, dtype=numpy.float32)

             
    def firing_rate(self, time_bin, **kwargs):
#         display=kwargs.get('display', False)
        average=kwargs.get('average', True)
#         binary=kwargs.get('binary', False)
        ids=kwargs.get('ids', False)
        proportion_connected=kwargs.get('proportion_connected', False)
#         call=super(MySpikeList, self)
        result = self.spike_histogram(time_bin, **kwargs)
        
        if ids:
            result=result[ids,:]
        if proportion_connected:
            upper = int(proportion_connected*result.shape[0])
            result=result[0:upper,:]

        if average:
            return numpy.mean(result, axis=0)
        else:
            return result  
#         kwargs=kwargs.get('kwargs',{})
#         return call.firing_rate(time_bin, display, average, binary, kwargs)

    def get_mean_coherence(self,*args, **kwargs):
        
#         try:
            return self.Factory_mean_coherence(*args, **kwargs)
#         except Exception as e:
#             s='\nTrying to do Factory_mean_coherence in get_mean_coherence'
#             s+='\nargs: {}'.format(args)
#             s+='\nkwargs: {}'.format(kwargs)
#             
#             raise type(e)(e.message + s), None, sys.exc_info()[2]
 
    def get_firing_rate(self, *args, **kwargs):
        return self.Factory_firing_rate(*args, **kwargs)
    
    def get_isi(self, *args, **kwargs): 
        return self.Factory_isis(args, **kwargs)

    def get_mean_rate(self, run=1,  **kwargs):       
        mr=self.mean_rate(**kwargs)
        x=numpy.ones(mr.shape)*run
        return Data_bar(**{'ids':self.id_list, 
                         'x':x,
                         'y':mr})
   
    def get_mean_rate_slices(self, *args, **kwargs):
        return self.Factory_mean_rate_slices(*args, **kwargs)
  

    
    def get_mean_rates(self, *args, **kwargs):
        return self.Factory_mean_rates(*args, **kwargs)




    def get_psd(self, *args, **kwargs):
        return self.Factory_psd(*args, **kwargs)
    
    def get_phase(self, lowcut, highcut, order,  fs, **kwargs):       
        
        '''
        Returns the phase of the population firing rate filters in the band
        lowcut to highcut. 
        '''
        time_bin=int(1000/fs)
        
        signal=self.firing_rate(time_bin, average=True, **kwargs)        
        y=sp.phase(signal, **kwargs)

        return {'ids':self.id_list,
                'x':self.time_axis_centerd(time_bin) , 
                'y':y}




    def get_phase_diff(self, *args, **kwargs):       
        return self.Factory_phase_diff(*args, **kwargs)

    def get_phases_diff_with_cohere(self, *args, **kwargs):       
        return self.Factory_phases_diff_with_cohere(*args, **kwargs)

    def get_phases(self, lowcut, highcut, order,  fs, **kwargs):       
        
        '''
        Returns the phase of the population firing rate filters in the band
        lowcut to highcut. 
        '''
        time_bin=int(1000/fs)
        
        signals=self.firing_rate(time_bin, average=False, **kwargs)        
        y=sp.phases(signals, **kwargs)

        x=numpy.array(self.time_axis_centerd(time_bin)*len(signals))
        return {'ids':self.id_list,
                'x':x , 
                'y':y}    
    
    def get_raster(self, *args, **kwargs):

        r=self.my_raster()
        return {'ids':r[2],
                'x':r[0],
                'y':r[1],
               }
    
    def get_spike_stats(self, *args, **kwargs):
        return self.Factory_spike_stat(*args, **kwargs)
     

    def mean_rate(self, **kwargs):
        
        return numpy.mean(self.mean_rates(**kwargs))
     
    def mean_rate_std(self, **kwargs):
        return numpy.std(self.mean_rates(**kwargs))    
    
    def mean_rate_slices(self, **kwargs):
        intervals = kwargs.get('intervals', None)
        repetition = kwargs.get('repetition', 1)
        y = []
        for start, stop in intervals:
            kwargs['t_stop'] = stop
            kwargs['t_start'] = start
            y.append(self.mean_rate(**kwargs))
        
        y = numpy.array(y)
        if not y.shape == (1, ):
            y = numpy.reshape(y, (repetition, len(y) / repetition))
        return y   

    
    def mean_rates(self, t_start=None, t_stop=None, **kwargs):
        """ 
        Returns a vector of the size of id_list giving the mean firing rate for each neuron

        Inputs:
            t_start - begining of the selected area to compute std(mean_rate), in ms
            t_stop  - end of the selected area to compute std(mean_rate), in ms
        
        If t_start or t_stop are not defined, those of the SpikeList are used
        
        See also
            mean_rate, mean_rate_std
        """
#         rates = []
        n=len(self.id_list)
        t_starts=[t_start]*n
        t_stops=[t_stop]*n
        l_spiketrains=[self.spiketrains]*n
        
        local_num_threads=kwargs.pop('local_num_threads',1)
        rates=map_parallel(wrap_compute_mean_rate, *[l_spiketrains,
                                                          self.id_list, 
                                                          t_starts, 
                                                          t_stops], 
                                **{'local_num_threads': local_num_threads})
        
        

#         for id in self.id_list:
#             rates.append(self.spiketrains[id].mean_rate(t_start, t_stop))
#         
        return rates
    
#     def mean_rates(self, **kwargs):
#         t_start=kwargs.get('t_start', None)
#         t_stop=kwargs.get('t_stop', None)
#         call=super(MySpikeList, self)
#         return call.mean_rates(t_start, t_stop)

    
    def time_slice(self, t_start, t_stop):
        """ 
        Return a new SpikeTrain obtained by slicing between t_start and t_stop,
        where t_start and t_stop may either be single values or sequences of
        start and stop times.
        
        Inputs:
            t_start - begining of the new SpikeTrain, in ms.
            t_stop  - end of the new SpikeTrain, in ms.

        Examples:
            >> spk = spktrain.time_slice(0,100)
            >> spk.t_start
                0
            >> spk.t_stop
                100
            >>> spk = spktrain.time_slice([20,70], [40,90])
            >>> spk.t_start
                20
            >>> spk.t_stop
                90
            >>> len(spk.time_slice(41, 69))
                0
        """
        new_SpikeList=super( MySpikeList, self ).time_slice(t_start, t_stop)
        
        new_MySpikeList=convert_super_to_sub_class(new_SpikeList, MySpikeList)
        
        return new_MySpikeList


    def id_slice(self, id_list):
        """ Slice by ids
        """
        new_SpikeList=super( MySpikeList, self ).id_slice(id_list)
        
        new_MySpikeList=convert_super_to_sub_class(new_SpikeList, MySpikeList)
        
        return new_MySpikeList
    
    def merge(self, spikelist, relative=False):
        """
        For each cell id in spikelist that matches an id in this SpikeList,
        merge the two SpikeTrains and save the result in this SpikeList.
        Note that SpikeTrains with ids not in this SpikeList are appended to it.
        
        Inputs:
            spikelist - the SpikeList that should be merged to the current one
            relative  - if True, spike times are expressed in a relative
                        time compared to the previsous one
            
        Examples:
            >> spklist.merge(spklist2)
            
        See also:
            concatenate, append, __setitem__
        """

        super( MySpikeList, self ).merge(spikelist, relative)

        self.t_start=numpy.min([v.t_start for v in self.spiketrains.values()])
        self.t_stop=numpy.max([v.t_stop for v in self.spiketrains.values()])
        # Update exrtra property
        self.ids.extend(spikelist.ids)
        self.ids.sort()
    
              
    def my_firing_rate(self, bin=200, display=True, id_list=[], n_rep=1, kwargs={}):          
        '''
        Plot all the instantaneous firing rates along time (in Hz) 
        over all ids in id_list from histogram generated by spike_histogram_n_rept.
        
        Inputs:
            bin        - the time bin used to gather the data
            display    - If True, a new figure is created. Could also be a subplot
            id_list    - list with ids to use for creation of histogram
            n_rep      - Number of experimental repetitions with same stimulation. 
                     E.g. if n_runs=3 then it is assumed that three 
                     experimental runs is recorded and the data will be 
                     sliced up into three time intervals.
            kwargs     - dictionary contening extra parameters that will be sent to the plot 
                         function
        '''
        
        ax = get_display(display)
        
        timeAxis, histograms = self.my_spike_histogram_n_rep( bin, id_list, 
                                                            n_rep, True )
        
        firingRates = numpy.mean( histograms, axis = 0 )
        
        if display:
            ax.plot(timeAxis, firingRates, **kwargs)
            ax.set_xlabel( 'Time (ms)' )
            ax.set_ylabel( 'Frequency (Hz)' )
        else: 
            return timeAxis, firingRates
    
    def my_raster(self, id_list=None, t_start=None, t_stop=None,
                   subsampling=1, **kwargs):
        """
        Generatse a raster plot for the SpikeList in a subwindow of interest,
        defined by id_list, t_start and t_stop. 
        
        Inputs:
            id_list - can be a integer (and then N cells are randomly selected) or a list of ids. If None, 
                      we use all the ids of the SpikeList
            t_start - in ms. If not defined, the one of the SpikeList object is used
            t_stop  - in ms. If not defined, the one of the SpikeList object is used
            kwargs  - dictionary contening extra parameters that will be sent to the plot 
                      function
        
        Examples:
            >> z = subplot(221)
            >> spikelist.raster_plot(display=z, kwargs={'color':'r'})
        
        See also
            SpikeTrain.raster_plot
        """
        
        if id_list == None: 
            id_list = self.id_list
            spk = self
        else:
            spk = self.id_slice(id_list)

        if t_start is None: t_start = spk.t_start
        if t_stop is None:  t_stop  = spk.t_stop
        if t_start != spk.t_start or t_stop != spk.t_stop:
            spk = spk.time_slice(t_start, t_stop)


        ids, spike_times = spk.convert(format="[ids, times]")
        
        new_id_list=numpy.arange(0, len(id_list)+0)
        ids_map = dict(zip(id_list, new_id_list))
        new_ids=[ids_map[id] for id in ids]
        
        return (numpy.array(spike_times[::subsampling]),
                 numpy.array(new_ids[::subsampling]), new_id_list)
    
    def my_raster_plot(self, id_list=None, t_start=None, t_stop=None, display=True, kwargs={}, subsampling=1):       
        subplot = get_display(display)
        
        spike_times, new_ids=self.my_raster(id_list, t_start, t_stop, kwargs, subsampling)
        
        
        if len(spike_times) > 0:
            subplot.plot(spike_times, new_ids, ',', **kwargs)
        xlabel = "Time (ms)"
        ylabel = "Neuron #"
        set_labels(subplot, xlabel, ylabel)
        
        
        new_id_list=numpy.arange(len(id_list))
        min_id = numpy.min(new_id_list)
        max_id = numpy.max(new_id_list)
        length = t_stop - t_start
        set_axis_limits(subplot, t_start-0.05*length, t_stop+0.05*length, min_id-2, max_id+2)
        pylab.draw()
          
    def my_raster_plot_cluster(self, id_list=[], t_start=None, t_stop=None, display=True, clusters = [], kwargs={}):
        """
        (functional?) Generate a raster plot for the SpikeList in a subwindow of interest,
        defined by id_list, t_start and t_stop. 
        
        Inputs:
            id_list - can be a integer (and then N cells are randomly selected) or a list of ids. If None, 
                      we use all the ids of the SpikeList
            t_start - in ms. If not defined, the one of the SpikeList object is used
            t_stop  - in ms. If not defined, the one of the SpikeList object is used
            display - if True, a new figure is created. Could also be a subplot
            clusters - vector containing code for cluster belonging of each spike 
                       train opatain from clustering analysis. Plotted accordingly.
            kwargs  - dictionary contening extra parameters that will be sent to the plot 
                      function
        
        Examples:
            >> z = subplot(221)
            >> spikelist.raster_plot(display=z, kwargs={'color':'r'})
        
        See also
            SpikeTrain.raster_plot
        """
        
        ax = get_display(display)
        
        spk = self.list
        
        if t_start is None: t_start = spk.t_start
        if t_stop is None:  t_stop  = spk.t_stop
        
        ids, spike_times = spk.convert(format="[ids, times]")
        idx = numpy.where((spike_times >= t_start) & (spike_times <= t_stop))[0]
        
        sorted_index = numpy.argsort( clusters )                            # Sort spike trains accoringly to clusters
        for i, id in enumerate(self.ids):
            ids[ids==id] = -self.ids[ sorted_index[ i ] ]
        ids = abs(ids)
                 
        
        if len(spike_times) > 0:
            ax.plot(spike_times, ids, ',', **kwargs)
        xlabel = "Time (ms)"
        ylabel = "Neuron #"
        set_labels(ax, xlabel, ylabel)
        
        min_id = numpy.min(spk.id_list())
        max_id = numpy.max(spk.id_list())
        length = t_stop - t_start
        set_axis_limits(ax, t_start-0.05*length, t_stop+0.05*length, min_id-2, max_id+2)
        pylab.draw()
                   

    def my_save(self, userFileName):
        '''
        Save spike list

        Inputs:
            userFileName    - name of file to save
            
        Examples:
            >> userFileName = /home/savename.dat
            >> aslist.save(userFileName)           
        '''
        userFile = StandardPickleFile( userFileName )      # create user file 
       
        # Invoke save function of base class ConductanceList which has 
        # AnalogSignalList as base class where the method exist.
        super(MySpikeList, self).save( userFile )

    def raw_data_binned(self, t_start, t_stop, res=1, clip=1):
        
        
        data=[]
        for id in self.ids:
            
            output=self.convert2bin(id, t_start, t_stop, res, clip)
            data.append(output)
            
        data=numpy.array(data)
        times=numpy.linspace(t_start+res/2,t_stop-res/2, data.shape[1]) 
        
        
        return times, data    
    
    def spike_count(self,start, stop, res=1, clip=0):
        count=0
        for id in self.ids:
            
            count=count+self.convert2bin(id, start, stop, res=res, clip=clip)
            
        return numpy.array(count,ndmin=2)


    def time_axis_centerd(self, time_bin):
        x=self.time_axis(time_bin) 
        
        return numpy.linspace(x[0]+time_bin/2., x[-1]-time_bin/2., len(x)-1)

def spike_histogram_fun(spike_times, time_bin, normalized, binary, nbins, _id):
    hist, edges = numpy.histogram(spike_times, nbins)
    hist = hist.astype(float)
    if normalized: # what about normalization if time_bin is a sequence?
        hist *= 1000.0 / float(time_bin)
    if binary:
        hist = hist.astype(bool)
    return hist
    
class BaseListMatrix(object):
    ''' 
    Composite class. 
    Made up of a 2d array of signal objects. Take the signals
    obejcts as instance variables (dependancy injection).
    '''
    def __init__(self, o, *args, **kwargs):
        
        self.m = to_numpy_2darray(o)       
        self.attr=None 
        self.allowed=kwargs.get('allowed',[])


    @property
    def shape(self):
        return self.m.shape
    
    def __getitem__(self, key):   
        m=numpy.matrix(self.m)
        m=m.__getitem__(key)
        if type(m)!=numpy.matrix:
            _m=numpy.empty((1,1), dtype=object)
        
            _m[0,0]=m
            m=_m
        m=numpy.array(m)
        return self.__class__(m)       
    
    def __getattr__(self, name):
        if name in self.allowed:
            self.attr=name
            return self._caller
        else:
#             for s in sorted(self.allowed):
#                 print s
            raise AttributeError(name)

    def __getstate__(self):
        #print 'im being pickled'
        return self.__dict__
    def __setstate__(self, d):
        #print 'im being unpickled with these values'
        self.__dict__ = d

    def __iter__(self):
        for i,j,obj in iter2d(self.m):
            yield i,j,obj

    def __repr__(self):
        return self.__class__.__name__+':\n'+str(self.m) 
    
        
    def _caller(self, *args, **kwargs):
        
        other=kwargs.get('other', None)
        if other:    
            o=other.merge(axis=0)
            o=o.merge(axis=1)        
            kwargs['other']=o.m[0,0]
        
        w=self.merge(axis=0)
        w=w.merge(axis=1)    
        
        call=getattr(w.m[0,0], self.attr)
        d=call(*args, **kwargs)
        return d
        

    def concatenate(self, a, *args, **kwargs):
        
        axis=kwargs.get('axis', 0)
        m=transpose_if_axis_1(axis, self.m)
        
        #if type(a)==list:
        #    a=to_numpy_2darray(a)
        
        s='Wrong length. List need to be {} is {}'
        assert a.shape[1]==m.shape[1], s.format(m.shape[1], a.shape[1])
        
#         a = to_numpy_2darray(a)
        m=numpy.concatenate((m,a.m), axis=0)
        
        self.m=transpose_if_axis_1(axis, m)
    def get_m(self, i,j):
        return self.m[i,j]
                

class VmListMatrix(BaseListMatrix):
    
    def __init__(self, matrix, *args, **kwargs):
        super( VmListMatrix, self ).__init__( matrix, *args,
                                                     **kwargs)
        self.allowed=kwargs.get('allowed',['plot', 
                                           'get_voltage_traces',
                                           'get_IV_curve'
                                           ]) 

        
    def Factory_IV_curve(self,*args, **kwargs):
        w=self.merge(axis=1)    
#         w=self
        x=numpy.zeros(w.m.shape)
        y=numpy.zeros(w.m.shape)
        y_std=numpy.zeros(w.m.shape)
        id_list=[]
        for i,j, obj in iter2d(w.m):
            x[i,j]=i
            y[i,j]=numpy.mean(obj.mean())
            y_std[i,j]=numpy.mean(obj.std())
            id_list=set(id_list).union(obj.id_list()) 
            
        d= {'ids':list(id_list),
            'y':y.ravel(),
            'y_std':y_std.ravel(),  
            'x':x.ravel()}
        return Data_IV_curve(**d)  
        
    def get_IV_curve(self, *args, **kwargs):
        return self.Factory_IV_curve(*args, **kwargs)
 
        
    def merge(self, axis=0, *args, **kwargs):

        m=transpose_if_axis_1(axis, self.m)
            
        'merge along rows'
        a=numpy.empty(shape=(1,m.shape[1]), dtype=object)
        a[0,:]=deepcopy(m[0,:])
            
        for i in xrange(1, m.shape[0]):
            for j in xrange(m.shape[1]):
                call=getattr(a[0,j], 'merge')
                call(m[i,j])
        
        a=transpose_if_axis_1(axis, a)
        a=[list(aa) for aa in a]
        
        return VmListMatrix(a)



    
class SpikeListMatrix(BaseListMatrix):
    def __init__(self, matrix, *args, **kwargs):
        
        super( SpikeListMatrix, self ).__init__( matrix, *args, **kwargs)
        
        self.allowed=kwargs.get('allowed', allowed_spike_list_functions())   
    
    
    
    def Factory_IF_curve(self, *args, **kwargs):
#         if self.isi_parts['y']==[]:
#             self.compute_set('isi_parts',*[],**{})
        d=self.get_isi_parts(*args, **kwargs)
        
        isi={'curr':[], 'first':[], 'mean':[], 'last':[]}
#         x, y=self.get('isi_parts', attr_list=['x', 'y'])

        x, y=d['x'], d['y']
        
        y=y.ravel()
        for xx, yy in zip(x,y):
            #if type(yy)!=list:
            #    yy=[yy]
                
            for yyy in yy:
                if not yyy.any():
                    yyy=[1000000.]
                isi['first'].append( yyy[ 0 ] )           # retrieve first isi
                isi['mean'].append( numpy.mean( yyy ) )   # retrieve mean isi
                isi['last'].append( yyy[ -1 ] )           # retrieve last isi
                isi['curr'].append(xx[0])
                
#                 if isi['last'][-1]==0:
#                     print 'je'
            n=len(yy)
           
        for key in isi.keys():
            a=numpy.array(isi[key])
            if a.shape[0]/n>=2:
                a=a.reshape((a.shape[0]/n, n))
            
            
            if key!='curr':
                a=1000./a #Convert to firing rate
            isi[key]=a
#         return isi['curr'], isi['first'], isi['mean'], isi['last']
        return Data_IF_curve(**isi)
    
    def Factory_mean_rate_parts(self, *args, **kwargs):
        #OBS merge over axis 1
        w=self.merge(axis=1)
        x=numpy.zeros(w.m.shape)
        y=numpy.zeros(w.m.shape)
        y_std=numpy.zeros(w.m.shape)
        id_list=[]
        for i,j, obj in iter2d(w.m):
            x[i,j]=i
            y[i,j]=obj.mean_rate(**kwargs)
            y_std[i,j]=obj.mean_rate_std(**kwargs)
            id_list=set(id_list).union(obj.id_list) 
            
        d= {'ids':list(id_list),
            'y_std':y_std.ravel(),
            'y':y.ravel(), 
            'x':x.ravel()}
        return Data_mean_rate_parts(**d)
     
#     def __getslice__(self, key):
#         return SpikeListMatrix(self.m[key])
    def as_spike_list(self):
        return self.merge(axis=0).merge(axis=1).m[0,0]    
      
    
      
    def get_raster(self, *args, **kwargs):
        
        self.attr='get_raster'
        l=self._caller(*args, **kwargs)
        
#         n=0
#         for d in l.ravel():
#             d['ids']+=n
#             d['x']+=n
#             n+=len(d['ids'])
        return l
    
    def get_IF_curve(self, *args, **kwargs):
        return self.Factory_IF_curve(*args, **kwargs) 
    
    def get_isi_parts(self, run=1, **kwargs): 
        w=self.merge(axis=1)
        x=numpy.zeros(w.m.shape)
        y=numpy.empty(w.m.shape, dtype=object)
        id_list=[]
        for i,j, obj in iter2d(w.m):
            x[i,j]=i
            y[i,j]=obj.isi(**kwargs)
            id_list=set(id_list).union(obj.id_list) 
            
               
        return {'ids':id_list,
                'x':x,
                'y':y}

    def get_mean_rate_parts(self, *args, **kwargs):
        return self.Factory_mean_rate_parts(*args, **kwargs)
 

    def merge(self, axis=0, *args, **kwargs):

        m=transpose_if_axis_1(axis, self.m)
            
        'merge along rows'
        a=numpy.empty(shape=(1,m.shape[1]), dtype=object)
        a[0,:]=deepcopy(m[0,:])
            
        for i in xrange(1, m.shape[0]):
            for j in xrange(m.shape[1]):
                call=getattr(a[0,j], 'merge')
                call(m[i,j])
        
        a=transpose_if_axis_1(axis, a)
        a=[list(aa) for aa in a]
        
        return SpikeListMatrix(a)

    def merge_matricies(self, other):
        new=deepcopy(self)

        for i,j, obj1 in iter2d(new.m):
            obj2=deepcopy(other.m[i,j])
            
            obj1.merge(obj2)
            
        return new
        
def allowed_spike_list_functions():
    l=[
       'firing_rate',
       'get_firing_rate',
       'get_IF_curve',
       'get_isi',
       'get_isi_IF',
       'get_mean_coherence',
       'get_mean_rate',
       'get_mean_rate_parts',
       'get_mean_rates', 
       'get_mean_rate_slices', 
       'get_raster',
       'get_phase',
       'get_phase_diff',
       'get_phases',
       'get_phases_diff_with_cohere',
       'get_psd',
       'get_spike_stats',
       'mean_rate',
       'mean_rates', 
       'merge',
       'my_raster',
       ]
    return l

    
def convert_super_to_sub_class(superClass, className):
        ''' Convert a super class object into a sub class object'''
        subClass = superClass
        del superClass
        subClass.__class__ = className
        subClass._init_extra_attributes()
        
        return subClass   


def iter2d(m):
    for i in xrange(m.shape[0]):
        for j in xrange(m.shape[1]):
            yield i, j, m[i,j]

def to_numpy_2darray(m):

    if type(m) is numpy.ndarray:
        if len(m.shape)==1:
            m=list(m)
        else:
            m=[list(mm) for mm in m]

    #if not 2d make
    
    if type(m)!=list:
        m=[m]
    if type(m[0])!=list:
        m=[m]
        
    assert m[0][0]!=list, 'lists to deep'
        
    a=numpy.empty(shape=(len(m),len(m[0])),  dtype=object)
    for i, j, _ in iter2d(a):
        a[i,j]=m[i][j]
    return a    


def dummy_data(flag, **kwargs):

    sw=misc.sample_wr  
    sa=numpy.random.sample
    np_rand=numpy.random.random
    sin=numpy.sin
    
    
    modulated=kwargs.get('modulated', True) 
    n_events=kwargs.get('n_events', 50)    
    n_pop=kwargs.get('n_pop', 10)
    n_sets=kwargs.get('n_sets', 3)
    id_start=kwargs.get('id_start',0)
    reset=kwargs.get('reset',False)
    run=kwargs.get('run',0)
    scale=kwargs.get('scale',2)
    set=kwargs.get('set',0)
    set_slice=misc.my_slice(set, n_pop, n_sets)
    shift=kwargs.get('shift',0.)
    sim_time=kwargs.get('sim_time', 1000.0)
    
    
    if not reset:
        start=int(sim_time*run)
        stop=int(sim_time*(1+run))
    else:
        start=int(0)
        stop=int(sim_time)
        
    V_rest=kwargs.get('V_rest', 60.0)
    ids=range(id_start, n_pop+id_start)[set_slice.get_slice()]
    if flag=='spike':
            
        n_events=(n_events+100*run+50*set)*sim_time/1000.0
        n=numpy.random.randint(int(n_events*0.8), n_events)

        i, t=[],[]#numpy.array(sw(ids,n_events))
        #t=numpy.array(sw(range(start, stop), n))
        for j in xrange(n_pop):
            i.append(n*[j+id_start])
            a=numpy.array(range(start, stop))
            a=a*(1+0.1*numpy.random.random(len(a)))
            a=a[(a>start)*(a<stop)]
            numpy.random.shuffle(a)
            
            t.append(a[0:n])
        i=numpy.concatenate(i)
        t=numpy.concatenate(t)
        ind=numpy.argsort(t)    
        i, t=numpy.take(i, ind), numpy.take(t, ind)
        
        jitter= numpy.random.normal(loc=0,  scale=scale, size=(len(t)))
        if modulated:
            v=numpy.sin(t*2*numpy.pi/100-numpy.pi*shift)
        else:
            v=0
            
        p_events=v+jitter
        i,t=i[p_events>0.6],t[p_events>0.6]
        
        l=MySpikeList( zip(i,t), ids, t_start=start, t_stop=stop)
    
    if flag=='voltage':
        size=len(range(n_pop)[set_slice.get_slice()])
        y=(0.995+0.01*np_rand((size, sim_time)))
        for yy in y:
            yy*=V_rest+run*2+set*2-numpy.random.rand()*1        
        ids_events=numpy.mgrid[0:n_pop, 1+sim_time*run:1+sim_time*(run+1)][0]
        ids_events=numpy.array(ids_events, int)  
        ids_events=numpy.ravel(ids_events[set_slice.get_slice()])
        y=numpy.ravel(y)
        
        
        signals=zip(numpy.ravel(ids_events), numpy.ravel(y))
        
        l=MyVmList(signals, ids, 1, t_start=start, t_stop=stop)
        
    return l

def dummy_data_matrix(flag, **kwargs):
    n_sets=4
    shape=kwargs.get('matrix_shape',(n_sets,4))
    kwargs['n_sets']=n_sets
    l=[]
    for i in range(shape[0]):
        l.append([])
        kwargs['run']=i
        for j in range(shape[1]):
            kwargs['set']=j
            l[-1].append(dummy_data( flag, **kwargs))
    
    if flag=='spike':
        return SpikeListMatrix(l)
    if flag=='voltage':
        return VmListMatrix(l)    


def get_mean_rate_slices(ids,  kwargs, y):
    shape = y.shape
    x = numpy.array(kwargs.get('x', numpy.arange(shape[1])))
    xticklabels = kwargs.get('xticklabels', x)
    y_mean = numpy.mean(y, axis=0)
    y_std = numpy.std(y, axis=0)
    d = {'ids':ids, 'y_raw_data':y, 
        'x':x, 
        'x_set':numpy.array(kwargs.get('x_set', x)), 
        'xticklabels':xticklabels, 
        'y':y_mean, 
        'y_std':y_std}
    
    return d
   
def shuffle(*args, **kwargs):
    out=[]
    sample=kwargs.get('sample',1)
    for a in args:
        a_copy=deepcopy(a)
        numpy.random.shuffle(a_copy)
        out.append(a_copy[0:sample])
    return out 
    
def transpose_if_axis_1(axis, m):
    if axis==0:
        pass
    if axis==1:
        m=m.transpose()
    return m

def wrap_compute_mean_rate(spiketrains, _id, t_start, t_stop):
    return spiketrains[_id].mean_rate(t_start, t_stop)

class TestDataElement(unittest.TestCase):
    def setUp(self):
        self.sim_time=6000.0
        self.sl=dummy_data('spike', **{'run':0, 'set':0, 'n_sets':1,
                                       'sim_time':self.sim_time,
                                       'scale':0.1})
        self.slnm=dummy_data('spike', **{'run':0, 'set':0, 'n_sets':1,
                                       'sim_time':self.sim_time,
                                       'scale':1, 'modulated':False})
        
        self.vl=dummy_data('voltage', **{'run':0, 
                                         'set':0, 
                                         'n_sets':1,
                                         'sim_time':self.sim_time})


    def test_bar(self):
        ax=pylab.subplot(111)
        data=[1.5,1.2] 
        obj=Data_bar(**{'y':data,})
        obj.bar(ax, **{'edgecolor':'k'})
        ax.set_xticklabels(['Label 1', 'Label 2'])
#         pylab.show()
 
 

           
    def test_bar2(self):
        ax=pylab.subplot(111)
        data=[[1, 2, 4],
              [1.5,1.2, 1.3]
              ] 
        obj=Data_bar(**{'y':data,})
        obj.bar2(ax, **{ 'colors':['r', 'b'], 
                        'alpha':False,
                        'color_axis':1, 
                        'hatchs':['-', 'x', 'x'], 
                        'edgecolor':'k'})
        ax.set_xticklabels(['Label 1', 'Label 2'])
#         pylab.show()

    def test_spike_stat(self):    
        obj=self.sl.Factory_spike_stat()
#         pp(obj)
#         obj.plot(pylab.subplot(111), win=20.0)
#         pylab.show()

 
    def test_element(self):
        d={'x':1,'y':2, 'z':3}
        obj=Data_element_base(**d)
        for key in d.keys():
            self.assertTrue(key in dir(obj))
 



    def test_firing_rate_plot(self):    
        obj=self.sl.Factory_firing_rate(**{'average':False})
        ax=pylab.subplot(111)
        obj.plot(ax, win=20.0)
        
        obj=self.sl.Factory_firing_rate(**{'average':False,
                                           'ids':[0,1,8,9]})
        obj.plot(ax, win=20.0)
        
#         pylab.show()
 
    def test_activity_histogram(self):
        obj1=self.sl.Factory_firing_rate()
        obj2=obj1.get_activity_histogram()
        self.assertAlmostEqual(obj2.get_activity_histogram_stat().y, 
                               0.0, delta=0.001) 
          
        obj1=self.slnm.Factory_firing_rate()
        obj2=obj1.get_activity_histogram()
        self.assertAlmostEqual(obj2.get_activity_histogram_stat().y, 
                               1.0, delta=0.05) 
          
        obj1=self.sl.Factory_firing_rate(**{'average':False})
        obj2=obj1.get_activity_histogram(**{'average':False})
          
        self.assertAlmostEqual(numpy.mean(obj2.get_activity_histogram_stat(**{'average':False}).y), 
                               0.0, delta=0.001) 

         
    def test_phases_diff_cohere_plot(self):
           
        other=self.sl
        kwargs={
                'NTFF':256,
                'fs':100.0,
                'NFFT':256,
                'noverlap':int(256/2),
                'other':other,
                'sample':10.,   
                   
                'lowcut':10,
                'highcut':20,
                'order':3,
                'bin_extent':10.,
                'kernel_type':'gaussian',
                'params':{'std_ms':5.,
                          'fs': 100.0},
                 
                'full_data':True,
         
                }
                   
        obj=self.sl.Factory_phases_diff_with_cohere(**kwargs)
        ax=pylab.subplot(111)
        obj.hist(ax)
        obj.hist2(ax)
#         pylab.show()



    def test_IF_curve_plot(self):
           
        self.sl=dummy_data_matrix('spike', **{'run':0, 'set':0, 'n_sets':1,
                                       'sim_time':self.sim_time})
        obj=self.sl.Factory_IF_curve()
        obj.plot(pylab.subplot(111), part='mean')
#         pylab.show()
      

    def test_IV_curve_plot(self):
         
        self.vlm=dummy_data_matrix('voltage', **{'run':0, 'set':0, 'n_sets':1,
                                       'sim_time':self.sim_time})
        pylab.figure()
        obj=self.vlm.Factory_IV_curve()
        obj.plot(pylab.subplot(111))
#         pylab.show()         
          
    def test_isis_hist(self):    
        obj=self.sl.Factory_isis()
        pylab.figure()
        obj.hist(pylab.subplot(111))
#         pylab.show()
  
    def test_mean_rates_hist(self):
        obj=self.sl.Factory_mean_rates()
        pylab.figure()
        obj.hist(pylab.subplot(111))
#         pylab.show(obj)   
  
    def test_mean_rate_parts_plot(self):
           
        #OBS merge over axis 1
        self.sl=dummy_data_matrix('spike', **{'run':0, 'set':0, 'n_sets':1,
                                       'sim_time':self.sim_time})
        obj=self.sl.Factory_mean_rate_parts()
        obj.plot(pylab.subplot(111))
#         pylab.show()
 
    def test_voltage_traces(self):
        pylab.figure()
        obj=self.vl.Factory_voltage_traces()
        obj.plot(pylab.subplot(111))        
#         pylab.show() 

    def test_mean_rate_slices(self):
        obj=self.sl.Factory_mean_rate_slices(**{'intervals':[[0,100], [200, 300], [300, 400],
                                                     [600,700], [800, 900], [900, 1000]],
                                        'repetition':2, 'local_num_threads':1})
        obj.plot(pylab.subplot(111))
#         pylab.show()

    def test_firing_rate_get_mean_rate_slices(self):
        obj_fr=self.sl.Factory_firing_rate(1)
        obj1=obj_fr.get_mean_rate_slices(**{'intervals':[[0,100], [200, 300], [300, 400],
                                                     [600,700], [800, 900], [900, 1000]],
                                        'repetition':2, 
                                        'xticklabels':['i', 'ii', 'iii'],
                                        'local_num_threads':2})
        obj1.plot(pylab.subplot(111))
        obj2=self.sl.Factory_mean_rate_slices(**{'intervals':[[0,100], [200, 300], [300, 400],
                                                     [600,700], [800, 900], [900, 1000]],
                                        'repetition':2, 'local_num_threads':1})
        self.assertListEqual(list(obj1.y), list(obj2.y))
        self.assertListEqual(list(obj1.y_std), list(obj2.y_std))
#         obj.plot(pylab.subplot(111))
#         pylab.show()
         
class TestSpikeList(unittest.TestCase):
    def setUp(self):
    
        self.sim_time=1000.0
        self.sl=dummy_data('spike', **{'run':0, 'set':0, 'n_sets':1,
                                       'sim_time':self.sim_time})
        
        self.home=expanduser("~")
        

    def test_1_get_phase(self):
        kwargs={'bin_extent':100.0,
                'inspect':False,
                'kernel_type':'gaussian',
                'params':{'std_ms':20,
                          'fs': 1000.0}}
        lowcut, highcut, order, fs=10.0, 20.0, 3, 1000.0
        d=self.sl.get_phase(lowcut, highcut, order, fs, **kwargs)
        self.assertEqual(d['x'].shape, d['y'].shape)
         


    def test_2_get_psd(self):
 
        NFFT, Fs=256, 1000.0
        d=self.sl.get_psd( NFFT, Fs)
        self.assertEqual(d.x.shape, d.y.shape)        
         
         


    def test_3_mean_rate(self):
        r=self.sl.mean_rate(t_start=0, t_stop=1000, **{'local_num_threads':1})
        r2=self.sl.mean_rate(t_start=0, t_stop=1000,  **{'local_num_threads':2})
        r3=self.sl.mean_rate(t_start=0, t_stop=1000,  **{'local_num_threads':3})
        self.assertEqual(r, r2)
        self.assertEqual(r, r3)
#         print r, r2, r3
         
         
    def test_4_firing_rate(self):
        fr=self.sl.firing_rate(1, **{'local_num_threads':1,
                                     't_stop':500})
        fr2=self.sl.firing_rate(1, **{'local_num_threads':3,
                                      't_stop':500})
        pylab.plot(fr)
        pylab.plot(fr2)
        pylab.show()
        self.assertListEqual(list(fr), list(fr2))
        
#         print fr
#         print fr2


    def test_41_firing_rate_mpi(self):
        import cPickle as pickle  
        import os
        from toolbox.data_to_disk import mkdir
            
        data_path= self.home+('/results/unittest/my_signals'
                         +'/firing_rate_mpi/')
        script_name=os.getcwd()+('/test_scripts_MPI/'
                                 +'my_signals_firing_rate_mpi.py')
        
        fileName=data_path+'data_in.pkl'
        fileOut=data_path+'data_out.pkl'
        mkdir(data_path)
        
        f=open(fileName, 'w') #open in binary mode 
        pickle.dump(self.sl, f, -1)
        f.close()
        

        
        np=4        
        p=subprocess.Popen(['mpirun', '-np', str(np), 'python', 
                            script_name, fileName, fileOut],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
#                             stderr=subprocess.STDOUT
                            )
        
        out, err = p.communicate()
#         print out
#         print err

        fr=self.sl.firing_rate(1, **{'local_num_threads':1})    
        
        f=open(fileOut, 'rb') #open in binary mode
        fr2=pickle.load(f)
        f.close()
        
        self.assertListEqual(list(fr), list(fr2))
        
    def test_5_mean_rate_slices(self):
        mrs=self.sl.mean_rate_slices(**{'intervals':[[0,100], [300, 400],
                                                     [600,700], [900, 1000]],
                                        'repetition':2, 'local_num_threads':1})
        mrs2=self.sl.mean_rate_slices(**{'intervals':[[0,100], [300, 400],
                                                     [600,700], [900, 1000]],
                                        'repetition':2, 'local_num_threads':3})
 
        self.assertListEqual([list(a) for a in mrs], 
                             [list(a) for a in mrs2])
#         print mrs
#         print mrs2

class TestSpikeListMatrix(unittest.TestCase):
    
        
    def setUp(self):
        
        self.n_runs=4
        self.n_sets=4
        self.n_pop=10
        l,l2=[],[]
        for run in xrange(self.n_runs):
            l.append([])
            l2.append([])
            for set in xrange(self.n_sets):               
                msl=dummy_data('spike', **{'run':run, 
                                           'set':set,
                                           'n_sets':self.n_sets,
                                           'n_pop':self.n_pop})
                l[run].append(msl) 
                
                msl=dummy_data('spike', **{'id_start':self.n_pop,
                                           'run':run, 
                                           'set':set,
                                           'n_sets':self.n_sets,
                                           'n_pop':self.n_pop})
                l2[run].append(msl)  
 
        self.spike_lists=l
        self.spike_lists2=l2


    

    def test_1_create(self):
        slc=SpikeListMatrix(self.spike_lists) 
        #print slc

    def test_10_item(self):
        slc=SpikeListMatrix(self.spike_lists) 
        self.assertEqual(slc[0:2,0:2].shape, (2,2))
        self.assertEqual(slc[0:2,0].shape, (2,1))
        self.assertEqual(slc[0,0:2].shape, (1,2))
        self.assertEqual(slc[0,0].shape, (1,1))

    def test_2_calls_wrapped_class(self):
        other=SpikeListMatrix(self.spike_lists2)
        calls=[
               ['firing_rate', [100], {'average':True,
                                       'local_num_threads':2}],

               ['get_firing_rate', [100],{'average':True}],
               ['get_isi',[],{}],
               ['get_mean_coherence', [],{'fs':256.0,
                                          'NFFT':256,
                                          'noverlap':int(256/2),
                                          'other':other,
                                          'sample':2.,
                                      }],
               ['get_mean_rate', [],{}],
               ['get_mean_rate_parts',[],{}],
               ['get_mean_rates',[],{}],
               ['get_mean_rate_slices',[], {'intervals':[[i*500, i*500+100]
                                                         for i in range(6)],
                                            'repetitions':3}], 
               ['get_psd', [], {'NFFT':256,
                                'fs':1000.0}],
               ['get_phase', [],{'lowcut':10,
                              'highcut':20,
                              'order':3,
                              'fs':1000.0}],
               ['get_phases', [],{'lowcut':10,
                              'highcut':20,
                              'order':3,
                              'fs':1000.0}],
               ['get_phase_diff',[],{'lowcut':10,
                              'highcut':20,
                              'order':3,
                              'fs':1000.0,
                              'bin_extent':10.,
                              'kernel_type':'gaussian',
                              'other':other,
                              'params':{'std_ms':5.,
                                        'fs': 1000.0},
                 }],
               ['get_spike_stats', [],{}],
               ['mean_rate', [], {'t_start':250, 't_stop':4000, 'local_num_threads':2}], 
               ['mean_rates', [], {}], 
               ['merge', [], {}],
               ['my_raster', [], {}],
               ['merge_matricies', [other],{}]
               ]
        
        slc=SpikeListMatrix(self.spike_lists)
        r=[]
        for call, a, k in calls:
            func=getattr(slc, call)
            r.append(func(*a, **k))
            d=r[-1]
            if call in [
                        'get_mean_rate',
                        ]:        
                self.assertEqual(d.x.shape, d.y.shape)
            if call in ['get_firing_rates',
                        'get_mean_rates',
                        'get_mean_coherence',
                        'get_isi']:
                self.assertEqual(d.x.shape, d.y.shape) 
                
    def test_3_class_methods(self):
        calls=[['get_raster', [],{}]]
        slc=SpikeListMatrix(self.spike_lists)
        r=[]
        for call, a, k in calls:
            func=getattr(slc, call)
            r.append(func(*a, **k))
        #print r

    def test_4_merge_spike_matrix(self):
        slc=SpikeListMatrix(self.spike_lists)
        slc1=slc.merge(axis=0)
        slc2=slc.merge(axis=1)
        self.assertEqual((1,self.n_sets), slc1.shape)
        self.assertEqual((self.n_runs,1), slc2.shape)

    def test_5_concatenate(self):
        
        slc1=[]
        slc2=[]
        for set in xrange(self.n_sets):
            slc1.append(dummy_data('spike', **{'run':self.n_runs, 
                                                'set':set, 
                                                'n_sets':self.n_sets}))
        for run in xrange(self.n_runs):
            slc2.append(dummy_data('spike', **{'run':run, 
                                                'set':self.n_sets,
                                                'n_sets':self.n_sets
                                                 }))
            
            
            
        slc01=SpikeListMatrix(self.spike_lists)
        slc02=SpikeListMatrix(self.spike_lists)
        slc01.concatenate(SpikeListMatrix(slc1), axis=0)
        slc02.concatenate(SpikeListMatrix(slc2), axis=1)
        
        self.assertEqual((self.n_runs+1,self.n_sets), slc01.shape)
        self.assertEqual((self.n_runs,self.n_sets+1), slc02.shape)

    
class TestVm_list_matrix(unittest.TestCase):     
    def setUp(self):
        
        self.n_runs=4
        self.n_sets=4
        l=[]=[]
        for run in xrange(self.n_runs):
            l.append([])
            for set in xrange(self.n_sets-1):               
                msl=dummy_data('voltage', **{'run':run, 
                                             'set':set,
                                             'n_sets':self.n_sets})
                l[run].append(msl)  
        self.vm_lists=l

    def test_1_create(self):
        slc=VmListMatrix(self.vm_lists) 
        #print slc

    def test_2_calls_wrapped_class(self):
        calls=[
               ['get_voltage_traces', [], {}],
               ['get_IV_curve',[],{}],
               ]
        
        slc=VmListMatrix(self.vm_lists)
        r=[]
        for call, a, k in calls:
            func=getattr(slc, call)
            r.append(func(*a, **k))
        #print r
        
        
if __name__ == '__main__':
    d={
        TestDataElement:[
#                     'test_bar',
#                     'test_bar2',
#                     'test_spike_stat',
#                     'test_element',
#                     'test_firing_rate_plot',
#                     'test_activity_histogram',
#                     'test_phases_diff_cohere_plot',
#                     'test_IF_curve_plot',  
#                     'test_IV_curve_plot',  
#                     'test_isis_hist',
#                     'test_mean_rates_hist',
#                     'test_mean_rate_parts_plot',
#                     'test_voltage_traces',
#                     'test_mean_rate_slices',
                    ],
          TestSpikeList:[
#                      'test_1_get_phase',
#                      'test_2_get_psd',
#                      'test_3_mean_rate',
#                      'test_4_firing_rate',
#                      'test_41_firing_rate_mpi',
#                      'test_5_mean_rate_slices',
                     ],
        TestSpikeListMatrix:[
#                      'test_1_create',
#                      'test_10_item',
                    'test_2_calls_wrapped_class',
#                      'test_3_class_methods',
#                      'test_4_merge_spike_matrix',
#                      'test_5_concatenate',
                    ],
#         TestVm_list_matrix:[
#                      'test_1_create',
#                      'test_2_calls_wrapped_class',
#                      ]
       }
    test_classes_to_run=d
    suite = unittest.TestSuite()
    for test_class, val in  test_classes_to_run.items():
        for test in val:
            suite.addTest(test_class(test))

    unittest.TextTestRunner(verbosity=2).run(suite)