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
import unittest
from toolbox import misc, my_axes
from toolbox import signal_processing as sp

# Import StandardPickleFile for saving of spike object
from NeuroTools.io import StandardPickleFile


from NeuroTools import signals
from NeuroTools.signals import ConductanceList
from NeuroTools.signals import CurrentList
from NeuroTools.signals import VmList
from NeuroTools.signals import SpikeList
from NeuroTools.plotting import get_display, set_labels, set_axis_limits


class Data_element_base(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self.__dict__[key] = value

    def __repr__(self):
        return self.__class__.__name__
    
    def __str__(self):
        return self.__class__.__name__


class Data_generic_base(object):
    def plot(self, ax, **k):
                
        if not isinstance(ax,my_axes.MyAxes):
            ax=my_axes.convert(ax)
  
        ax.plot(self.x, self.y, **k)
        ax.set_xlabel('x')
        ax.set_ylabel('y') 
        ax.my_set_no_ticks(xticks=6, yticks=6)
        ax.legend()

class Data_generic(Data_element_base, Data_generic_base):
    pass        
class Data_firing_rate_base(object):
    def plot(self, ax, win=100, t_stop=0, t_start=numpy.inf, **k):
                
        if not isinstance(ax,my_axes.MyAxes):
            ax=my_axes.convert(ax)
            
        exp=(self.x<t_start)*(self.x>t_stop)
        x,y=self.x[exp], self.y[exp]
        y=misc.convolve(y, **{'bin_extent':win, 
                              'kernel_type':'triangle',
                              'axis':1})     
        ax.plot(x, y, **k)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (spike/s)') 
        ax.my_set_no_ticks(xticks=6, yticks=6)
        ax.legend()

class Data_firing_rate(Data_element_base, Data_firing_rate_base):
    pass


class Data_fmin_base(object):
    def plot(self, ax, name, **k):
        p,x,y=ax.texts, 0.02,0.9
        if len(p):
            x=p[-1]._x
            y=p[-1]._y-0.1
             
        p=ax.text( x, y, name+':'+str(self.xopt[-1][0])+' Hz', 
                   transform=ax.transAxes, 
            fontsize=pylab.rcParams['font.size']-2)

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

class Data_mean_rates_base(object):
    def hist(self, ax, **k):
        if not isinstance(ax,my_axes.MyAxes):
            ax=my_axes.convert(ax)
            
        k['histtype']=k.get('histtype','step')
            
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
    def hist(self):
        raise NotImplementedError
    
class Data_mean_rate_slices(Data_element_base, Data_mean_rate_slices_base):
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
        # is ConductanceList. The next argument. seöf, passes a reference to 
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
                print 


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
        time_bin=kwargs.get('time_bin', 1 )
        t_start=kwargs.get('t_start', None)
        t_stop=kwargs.get('t_stop', None)
        x=self.time_axis_centerd(time_bin) 
        y=self.firing_rate(time_bin, **kwargs)
        if t_start:
            y=y[x>t_start]
            x=x[x>t_start]
        if t_stop:
            y=y[x<t_stop]
            x=x[x<t_stop]
        d={'ids':self.id_list, 'x':x,  'y':y,}    
        return Data_firing_rate(**d)

    def Factory_isis(self, *args, **kwargs):
        run=kwargs.get('run',1)
        y=numpy.array(self.isi(), dtype=object)
        x=numpy.array([run]*y.shape[0])    
        d={'ids':self.id_list,
           'x':x,
           'y':y}
        return Data_isis(**d)
        
    
    def Factory_mean_rates(self, *args, **kwargs):
        run=kwargs.get('run',1)
        y=numpy.array(self.mean_rates(**kwargs)) 
        x=numpy.ones(y.shape)*run
        d= {'ids':self.id_list,  'y':y, 'x':x} 
        return Data_mean_rates(**d)

    def Factory_mean_rate_slices(self, *args, **kwargs):
        intervals=kwargs.get('intervals',None)
        repetitions=kwargs.get('repetitions',1)
        y=[]
        for start, stop in intervals:
            kwargs['t_stop']=stop
            kwargs['t_start']=start
            y.append(self.mean_rate(**kwargs)) 
            fr=self.firing_rate(1)
        y=numpy.array(y)
        y=numpy.reshape(y,(repetitions,len(y)/repetitions))
        y_std=numpy.std(y, axis=0)
        y_mean=numpy.mean(y, axis=0)
        x=numpy.ones(len(y))
        d={'ids':self.id_list,
           'y':y,
           'y_mean':y_mean, 
           'y_std':y_std,
           'x':x}
        return Data_mean_rate_slices(**d) 

    def Factory_spike_stat(self, *args, **kwargs):
        d={'rates':{},'isi':{}}
        d['rates']['mean']=self.mean_rate(**kwargs)
        d['rates']['std']=self.mean_rate_std(**kwargs)
        d['rates']['CV']=d['rates']['std']/d['rates']['mean']
        
        isi=numpy.concatenate((self.isi()))
        d['isi']['mean']=numpy.mean(isi,axis=0)
        d['isi']['std']=numpy.std(isi,axis=0)
        d['isi']['CV']=d['isi']['std']/d['isi']['mean']
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
 
    def firing_rate(self, time_bin, **kwargs):
        display=kwargs.get('display', False)
        average=kwargs.get('average', True)
        binary=kwargs.get('binary', False)
        kwargs=kwargs.get('kwargs',{})
        call=super(MySpikeList, self)
        
        return call.firing_rate(time_bin, display, average, binary, kwargs)

    def get_mean_coherence(self,**kwargs):
        fs=kwargs.get('fs',1000.0)
        kwargs['fs']=fs
        other=kwargs.get('other', None)
        sample=kwargs.get('sample',10)
        
        assert other!=None, 'need to provide other spike list'
        
        time_bin=int(1000/fs)
        
        ids1, ids2=shuffle(*[self.id_list, other.id_list],
                           **{'sample':sample})
               
        sl1=self.id_slice(ids1)
        sl2=other.id_slice(ids2)
        
        signals1=sl1.firing_rate(time_bin, average=False, **kwargs)
        signals2=sl2.firing_rate(time_bin, average=False, **kwargs) 
        
        x, y=sp.mean_coherence(signals1, signals2, **kwargs)
        return {'ids1':ids1,
                'ids2':ids2, 
                'x':x, 
                'y':y,}
 
    def get_firing_rate(self, *args, **kwargs):
        return self.Factory_firing_rate(*args, **kwargs)
    
    def get_isi(self, *args, **kwargs): 
        return self.Factory_isis(args, **kwargs)

    def get_mean_rate(self, run=1,  **kwargs):       
        mr=self.mean_rate(**kwargs)
        x=numpy.ones(mr.shape)*run
        return {'ids':self.id_list, 
                'x':x,
                'y':mr, }
   
    def get_mean_rate_slices(self, *args, **kwargs):
        return self.Factory_mean_rate_slices(*args, **kwargs)
  

    
    def get_mean_rates(self, *args, **kwargs):
        return self.Factory_mean_rates(*args, **kwargs)

    def get_psd(self, NFFT, fs, **kwargs):
        
        noverlap=kwargs.get('noverlap',int(NFFT/2))
        time_bin=time_bin=int(1000/fs)
        
        signal=self.firing_rate(time_bin, average=True, **kwargs) 
        y,x=sp.psd(signal, NFFT, fs, noverlap, **kwargs)
        return {'ids':self.id_list,
                 'x':x , 
                 'y':y}

    def get_phase(self, lowcut, highcut, order,  fs, **kwargs):       
        
        '''
        Returns the phase of the population firing rate filters in the band
        lowcut to highcut. 
        '''
        time_bin=int(1000/fs)
        
        signal=self.firing_rate(time_bin, average=True, **kwargs)        
        y=sp.phase(signal, lowcut, highcut, order, fs, **kwargs)

        return {'ids':self.id_list,
                'x':self.time_axis_centerd(time_bin) , 
                'y':y}


    def get_phase_diff(self, lowcut, highcut, order,  fs, 
                             **kwargs):       
        
        '''
        Returns the phase of the population firing rate filters in the band
        lowcut to highcut. 
        '''
        time_bin=int(1000/fs)
        other=kwargs.get('other', None)
        
        assert other!=None, 'need to provide other spike list'
        
        signal1=self.firing_rate(time_bin, average=True, **kwargs)
        signal2=other.firing_rate(time_bin, average=True, **kwargs)        
        
        args=[lowcut, highcut, order, fs]
        y=sp.phase_diff(signal1, signal2, *args, **kwargs)
 
        return {'ids1':self.id_list,
                'ids2':other.id_list,
                'x':self.time_axis_centerd(time_bin) , 
                'y':y}


    def get_phases(self, lowcut, highcut, order,  fs, **kwargs):       
        
        '''
        Returns the phase of the population firing rate filters in the band
        lowcut to highcut. 
        '''
        time_bin=int(1000/fs)
        
        signals=self.firing_rate(time_bin, average=False, **kwargs)        
        y=sp.phases(signals, lowcut, highcut, order, fs, **kwargs)

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

    def mean_rates(self, **kwargs):
        t_start=kwargs.get('t_start', None)
        t_stop=kwargs.get('t_stop', None)
        call=super(MySpikeList, self)
        return call.mean_rates(t_start, t_stop)

    
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
    

    
    def my_firing_rate_sliding_window(self, bin=100, display=True, id_list=[], step=1, stop=None, kwargs={}):
        ''' 
        Calculate spike rates at ``sample_step`` using s sliding rectangular 
        window.
        
        Arguments:
        bin           Bin size of sliding window
        display       If True, a new figure is created. Could also be a subplot
        id_list       List of ids to calculate firing rate for
        step          Step size for moving sliding window (ms)
        stop          End of spike train
        kwargs        Additional plot arguments
          
        Here the window is centered over over each time point at sampe_step with 
        window size equalling bin_size. Takes spike times, number of 
        neurons no_neurons, bin_size and sample_step as input argument.
        '''  
        
        ax = get_display(display)
        
        if stop is None: stop = self.t_stop
        if not any(id_list): id_list = self.ids    
        
        spikes = {}     # dictionary with spike times   
        for id in id_list: 
            spikes[ id ] = self.spiketrains[ id ].spike_times.copy()                   
        
        n  = int( (stop - bin )/step )          # number of windows
        f = bin/2                               # first window at bin/2
        l = step*n + bin/2                      # last window at bin*n - bin/2 
 
        # Time axis for sliding windows, n_win + 1 due to end points 
        # [ 0, 1, 2] -> two windows and tre timings
        timeAxis = numpy.linspace(f, l, n + 1)           
        
        
        firingRates = []                          # sliding time window data  
        dataSpk   = []
        
        #! Calculate number of spikes in ms bins and then sliding window rates 
        for id in id_list:
                    
            spk = spikes[id] 
            dataSpk.append(spk)

            i = 0
            j = bin/2
            j_max  = stop
            rates = []
                 
            #! For each ms i in 0 to stop at step intervals 
            for tPoint in timeAxis:                                    
                
                sum = numpy.sum( ( tPoint - bin/2 <= spk ) * ( spk < tPoint + bin/2 ) )
                rates.append( 1000.0 * sum/float( bin ) )
                  

            firingRates.append(rates)
        
        firingRates = numpy.array(firingRates)        # Convert to np array
   
        meanFiringRates = numpy.mean( firingRates, axis = 0 )
       
        ax.plot(timeAxis, meanFiringRates,**kwargs)
        ax.set_xlabel( 'Time (ms)' )
        ax.set_ylabel( 'Frequency (spike/s)' )
        
        return timeAxis, firingRates, dataSpk   
        
 
    def my_image_firing_rate_slide_window(self, bin=100, id_list=[], step=1, stop=None, display=True, kwargs={}):
        '''
        Function that create an image with bin size sliding window firing rate as 
        calculated at step intervals. kwargs - dictionary contening extra 
        parameters that will be sent to the plot function    
        
        Arguments:
        bin           Bin size of sliding window
        display       If True, a new figure is created. Could also be a subplot
        id_list       List of ids to calculate firing rate for
        step          Step size for moving sliding window (ms)
        stop          End of spike train
        kwargs        Additional plot arguments
        
        '''
        ax = get_display(display)
        
        if not any(id_list): id_list = self.ids
        
        t, r, spk = self.my_firing_rate_sliding_window(bin, display, id_list, 
                                                    step, stop, kwargs)
               
        kwargs.update( { 'origin' : 'lower', } )
        image = ax.imshow(r, extent=[t[0],t[-1],self.ids[0],self.ids[-1]], 
                          **kwargs)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Neuron #')
        ax.set_aspect(aspect='auto')

        return image

    def my_spike_histogram_n_rep(self, bin=10, id_list=[], n_rep=1, normalized=True):
        '''
        Generate an array with all the spike_histograms of all the SpikeTrains. 
        Each time series should be n_rep repeated simulations. Histogram is then
        generated for one simulation period repeated n_rep times. It is thus
        possible to supply several similar simulations and get the 
        mean firing statistics for all of them. 
        
        Inputs:
            bin        - the time bin used to gather the data
            id_list    - list with ids to use for creation of histogram
            n_rep      - number of experimental repetitions with same 
                         stimulation. E.g. if n_runs=3
            normalized - if True, the histogram are in Hz (spikes/second), 
                         otherwise they are in spikes/bin

        
        In neurophysiology, this is similar to a peristimulus time histogram 
        and poststimulus time histogram, both abbreviated PSTH or PST histogram, 
        are histograms of the times at which neurons fire. These histograms are 
        used to visualize the rate and timing of neuronal spike discharges in 
        relation to an external stimulus or event. The peristimulus time 
        histogram is sometimes called perievent time histogram, and 
        post-stimulus and peri-stimulus are often hyphenated.
        
        To make a PSTH, a spike train recorded from a single neuron is aligned 
        with the onset, or a fixed phase point, of an identical stimulus 
        repeatedly presented to an animal. The aligned sequences are 
        superimposed in time, and then used to construct a histogram.[2] 
        
        Construction procedure
        For each spike train i
        
        1. Align spike sequences with the onset of stimuli that repeated n 
           times. For periodic stimuli, wrap the response sequence back to 
           time zero after each time period T, and count n (n_rep) as the 
           total number of periods of data.
        2. Divide the stimulus period or observation period T into N bins 
           of size Δ seconds ( bin ).
        3. Count the number of spikes k_i_j from all n sequences that 
           fall in the bin j.
        4. Calculate  i_j given by k_i_j/( n*Δ ) in units of estimated 
           spikes per second at time j * Δ.

        The optimal bin size Δ is a minimizer of the formula, (2k-v)/Δ2, where
        k and v are mean and variance of k_i. [3]  
                                                                       
        '''
        
        if not any(id_list): id_list = self.ids
        
        spkList = self.id_slice( id_list )      # Retrieve spike trains 
        
        # Short cuts
        start    = spkList.t_start 
        period   = spkList.t_stop/float(n_rep) 
        
        spike_train_hist = {}       # Create data table for histogram
           
        # Add spikes to data
        for id, spk in spkList.spiketrains.iteritems():
            
            
            # First take modulus t_period of spike times ending up with a vector
            # with spike times between 0 and period    
            spikes_mod_period = spk.spike_times % period                                        
            
            # Create sequence of bins edges, histogram between 0 and period
            bin_sequence = numpy.arange( start, period + bin, bin )    
            
            # Create histogram over bin sequence and convert to spikes per bin 
            hist_n_rep, edges = numpy.histogram( spikes_mod_period, bin_sequence)
            hist_n_rep        = hist_n_rep/float( n_rep )     
        
            spike_train_hist[ id ] = hist_n_rep     # Add to dictionary                                              
 
        # Create np array with all spike trains histograms   
        histograms = numpy.array( spike_train_hist.values() )                                      
        
        # Convert data to spike rates
        if normalized: histograms *= 1000.0/bin                                            

        # Center time axis over bin mid points 
        timeAxis = numpy.linspace(bin_sequence[0]+bin/2., 
                                  bin_sequence[-1]-bin/2., 
                                  len(bin_sequence)-1)
        return timeAxis, histograms
                                           
    def my_image_spike_histogram(self, bin=10, display=True, id_list=[], n_rep=1, normalized=True, kwargs = {} ):
        '''
        Plot an image of all the spike_histograms generated by 
        spike_histogram_n_rep
    
        
        Arguments:
        bin        - the time bin used to gather the data
        display    - If True, a new figure is created. Could also be a subplot
        id_list    - list with ids to use for creation of histogram
        n_rep      - Number of experimental repetitions with same stimulation. 
                     E.g. if n_runs=3 then it is assumed that three 
                     experimental runs is recorded and the data will be 
                     sliced up into three time intervals.
        normalized - if True, the histogram are in Hz (spikes/second), 
                     otherwise they are in spikes/bin
        kwargs     . Additional plot arguments
        '''        
        
        ax = get_display(display)
        
        timeAxis, histograms = self.spike_histogram_n_rep( bin, id_list, 
                                                            n_rep, normalized )
             
        kwargs.update( { 'origin' : 'lower', } )
        image = ax.imshow( histograms, **kwargs )
        ax.set_xlabel( 'Time (ms)'     )
        ax.set_ylabel( 'Neuron #'      )
        ax.set_aspect( aspect = 'auto' )
        
        
        n_points=len(timeAxis)
        xticks = numpy.arange( 0, n_points, n_points*0.2)
        xticklabels = numpy.arange( 0, timeAxis[-1], timeAxis[-1]*0.2)
        ax.set_xticks( xticks )
        ax.set_xticklabels( [ str( l ) for l in xticklabels ] )
        
        n_ids=len(id_list)
        yticks =  numpy.arange( 0, n_ids, n_ids*0.2)
        ax.set_yticks( yticks)
        ax.set_yticklabels( id_list[ yticks ] )
    
        return image
           
    def my_plot_spike_histogram(self, bin=10, display=True, id_list=[], n_rep=1, normalized=True, kwargs = {} ):    
        '''
        Plot an mean spike histogram for for all ids generated by 
        spike_histogram_n_rep
        
        Arguments:
        bin        - the time bin used to gather the data
        display    - If True, a new figure is created. Could also be a subplot
        id_list    - list with ids to use for creation of histogram
        n_rep      - Number of experimental repetitions with same stimulation. 
                     E.g. if n_runs=3 then it is assumed that three 
                     experimental runs is recorded and the data will be 
                     sliced up into three time intervals.
        normalized - if True, the histogram are in Hz (spikes/second), 
                     otherwise they are in spikes/bin
    
        '''        
        ax = get_display(display)
        
        timeAxis, histograms = self.spike_histogram_n_rep( bin, id_list, 
                                                            n_rep, normalized )
              
        ax.bar(left=timeAxis, height=histograms, width=0.8, 
               bottom=None, hold=None, **kwargs)
              
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
        
        s='Wrong length. List need to be {}'
        assert a.shape[1]==m.shape[1], s.format(m.shape[1])
        
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
    
    n_events=kwargs.get('n_events', 50)    
    n_pop=kwargs.get('n_pop', 10)
    n_sets=kwargs.get('n_sets', 3)
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
    ids=range(n_pop)[set_slice.get_slice()]
    if flag=='spike':
            
        n_events=(n_events+100*run+50*set)*sim_time/1000.0
        n=numpy.random.randint(int(n_events*0.8), n_events)

        i, t=[],[]#numpy.array(sw(ids,n_events))
        #t=numpy.array(sw(range(start, stop), n))
        for j in xrange(n_pop):
            i.append(n*[j])
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
        p_events=numpy.sin(t*2*numpy.pi/50-numpy.pi*shift)+jitter
        i,t=i[p_events>0.3],t[p_events>0.3]
        
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


class TestDataElement(unittest.TestCase):
    def setUp(self):
        self.sim_time=1000.0
        self.sl=dummy_data('spike', **{'run':0, 'set':0, 'n_sets':1,
                                       'sim_time':self.sim_time})
        self.vl=dummy_data('voltage', **{'run':0, 'set':0, 'n_sets':1,
                                       'sim_time':self.sim_time})
    
    
    def test_element(self):
        d={'x':1,'y':2, 'z':3}
        obj=Data_element_base(**d)
        for key in d.keys():
            self.assertTrue(key in dir(obj))

    def test_firing_rate_plot(self):    
        obj=self.sl.Factory_firing_rate()
        obj.plot(pylab.subplot(111), win=100.0)
#       pylab.show()
    
    
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

class TestSpikeList(unittest.TestCase):
    def setUp(self):
    
        self.sim_time=1000.0
        self.sl=dummy_data('spike', **{'run':0, 'set':0, 'n_sets':1,
                                       'sim_time':self.sim_time})
        
        
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
        self.assertEqual(d['x'].shape, d['y'].shape)        
        
class TestSpikeListMatrix(unittest.TestCase):
    
        
    def setUp(self):
        
        self.n_runs=4
        self.n_sets=4
        l=[]=[]
        for run in xrange(self.n_runs):
            l.append([])
            for set in xrange(self.n_sets-1):               
                msl=dummy_data('spike', **{'run':run, 
                                            'set':set,
                                            'n_sets':self.n_sets})
                l[run].append(msl)  
        self.spike_lists=l
    
        
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
        other=SpikeListMatrix(self.spike_lists)
        calls=[
               ['firing_rate', [100], {'average':True}],

               ['get_firing_rate', [100],{'average':True}],
               ['get_isi',[],{}],
               ['get_mean_coherence', [],{'fs':1000.0,
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
               ['get_psd', [256,1000.0],{}],
               ['get_phase', [10,20,3,1000.0],{}],
               ['get_phases', [10,20,3,1000.0],{}],
               ['get_phase_diff', 
                [10,20,3,1000.0],
                {'bin_extent':10.,
                 'kernel_type':'gaussian',
                 'other':other,
                 'params':{'std_ms':5.,
                           'fs': 1000.0},
                 }],
               ['get_spike_stats', [],{}],
               ['mean_rate', [], {'t_start':250, 't_stop':4000}], 
               ['mean_rates', [], {}], 
               ['merge', [], {}],
               ['my_raster', [], {}],
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
                self.assertEqual(d['x'].shape, d['y'].shape)
            if call in ['get_firing_rates',
                        'get_mean_rates',
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
        self.assertEqual((1,self.n_sets-1), slc1.shape)
        self.assertEqual((self.n_runs,1), slc2.shape)

    def test_5_concatenate(self):
        
        slc1=[]
        slc2=[]
        for set in xrange(self.n_sets-1):
            slc1.append(dummy_data('spike', **{'run':self.n_runs, 
                                                'set':set, 
                                                'n_sets':self.n_sets}))
        for run in xrange(self.n_runs):
            slc2.append(dummy_data('spike', **{'run':run, 
                                                'set':self.n_sets-1,
                                                'n_sets':self.n_sets
                                                 }))
            
            
            
        slc01=SpikeListMatrix(self.spike_lists)
        slc02=SpikeListMatrix(self.spike_lists)
        slc01.concatenate(SpikeListMatrix(slc1), axis=0)
        slc02.concatenate(SpikeListMatrix(slc2), axis=1)
        
        self.assertEqual((self.n_runs+1,self.n_sets-1), slc01.shape)
        self.assertEqual((self.n_runs,self.n_sets), slc02.shape)

    
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
    test_classes_to_run=[
                        TestDataElement,
                        TestSpikeList,
                        TestSpikeListMatrix,
                        TestVm_list_matrix
                         ]
    suites_list = []
    for test_class in test_classes_to_run:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)
    
    #suite =unittest.TestLoader().loadTestsFromTestCase(TestNode)
    #suite =unittest.TestLoader().loadTestsFromTestCase(TestNode_dic)
    #suite =unittest.TestLoader().loadTestsFromTestCase(TestStructure)
    unittest.TextTestRunner(verbosity=2).run(big_suite)
    
    #unittest.main()    

    
       