from NeuroTools import check_dependency

HAVE_INTERVAL = check_dependency('interval')

if HAVE_INTERVAL:
    from interval import *

import numpy

class Interval(object):
    """
    Interval(start_times, end_times).

    Inputs:
        start_times - A list of the start times for all the sub intervals considered, in ms
        stop_times  - A list of the stop times for all the sub intervals considered, in ms
    
    Examples:
        >> itv = Interval([0,100,200,300],[50,150,250,350])
        >> itv.time_parameters()
            0, 350
    """
    
    def __init__(self, start_times, end_times) :
        """
        Constructor of the Interval object.

        """
        if HAVE_INTERVAL:
            self.start_times = start_times
            self.end_times   = end_times
            # write the intervals to an interval object (pyinterval)
            scalar_types = (int, float, numpy.float, numpy.float32, numpy.float64, numpy.int, numpy.int8,
                            numpy.int16, numpy.int32, numpy.int64)
            test = isinstance(start_times, scalar_types)
            if test:
                self.start_times = [self.start_times]
            test = isinstance(end_times, scalar_types)
            if test:
                self.end_times = [self.end_times]
            if len(self.start_times) != len(self.end_times) :
                raise Exception("There sould be an equal number of starts and stops")
            self.interval_data = interval(*numpy.transpose(numpy.array([start_times,end_times])))
        else:
            test = isinstance(start_times, int) or isinstance(start_times, float)
            assert test, "Interval package not present, start_times should be a number !"
            test = isinstance(end_times, int) or isinstance(end_times, float)
            assert test, "Interval package not present, end_times should be a number !"
            self.start_times = [start_times]
            self.end_times   = [end_times]

    def intersect(self, itv) :
        self.interval_data = self.interval_data & itv.interval_data

    def union(self, itv) :
        self.interval_data = self.interval_data | itv.interval_data

    def __str__(self):
        return str(self.interval_data)

    def __len__(self):
        return shape(self.interval_data)[0]
    
    def __getslice__(self, i, j):
        """
        Return a sublist of the spike_times vector of the SpikeTrain
        """
        return self.interval_data & interval([i,j])

    def time_parameters(self):
        """
        Return the time parameters of the SpikeTrain (t_start, t_stop)
        """
        bounds = self.interval_data.extrema
        return (bounds[0][0],bounds[-1][0])
    
    def t_start(self):
        if HAVE_INTERVAL:
            return self.interval_data.extrema[0][0]
        else:
            return self.start_times[0]
    
    def t_stop(self):
        if HAVE_INTERVAL:
            return self.interval_data.extrema[-1][0]
        else:
            return self.end_times[0]
    
    def copy(self):
        """
        Return a copy of the SpikeTrain object
        """
        return Interval(self.start_times, self.end_times, self.t_start, self.t_stop)

    def offset(self, start=None, end=None) :
        """
        Modifies globally the intervals by offsetting the start and end of the stimulation. 
        
        The start and/or stop arguments should be tuples (limit, offset) where limit
        defines the reference limit (0 : start, 1: end) from where the new value is
        defined by adding the offset value to the current start/end.          
        """
        n_intervals = len(list(self.interval_data))
        new_iv = zeros((n_intervals,2))
        if start is None :
            for i in range(n_intervals) :
                new_iv[i,0] = self.interval_data[i][0]
        else :
            if start[0] == 0 :
                for i in range(n_intervals) :
                    new_iv[i,0] = self.interval_data[i][0] + start[1]
            if start[0] == 1 :
                for i in range(n_intervals) :
                    new_iv[i,0] = self.interval_data[i][1] + start[1]
        
        if end is None :
            for i in range(n_intervals) :
                new_iv[i,1] = self.interval_data[i][1]
        else :
            if end[0] == 0 :
                for i in range(n_intervals) :
                    new_iv[i,1] = self.interval_data[i][0] + end[1]
            if end[0] == 1 :
                for i in range(n_intervals) :
                    new_iv[i,1] = self.interval_data[i][1] + end[1]

        self.interval_data = interval(*list(new_iv))

    def total_duration(self) :
        """
        Return the total duration of the interval
        """  
        tot_duration = 0
        for i in self.interval_data :
            tot_duration += i[1] - i[0]
        return tot_duration


    def slice_times(self, times):
        spikes_selector = numpy.zeros(len(times), dtype=numpy.bool)
        if HAVE_INTERVAL:
            for itv in self.interval_data :
                spikes_selector = spikes_selector + (times > itv[0])*(times <= itv[1])
        else:
            spikes_selector = (times >= self.t_start()) & (times <= self.t_stop())
        return numpy.extract(spikes_selector, times)



#def build_psth(spiketrain, eventtrain, before_Dt, after_Dt, intervals=None):
    #"""         
    #build a psth of the spikes around the events 
    
    #If intervals != None, the eventtrain is restricted to the intervals provided.
    #"""
    ## tested : generates a correct PSTH when loaded with following data :
    ##spikes = [0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,100.,101.,102.,103.,104.,105.,106.,200.,201.,202.,203.,204.]
    ## 
    #if intervals != None:
        ## keep only the events that are included in IntervalTrain object
        #eventtrain = intervals.of_spikes(eventtrain)

    #nRepeats = len(eventtrain.spike_times)
    ## accumuate spikes around the events. 
    #spikes_around_event = []
    #for event_time in eventtrain.spike_times :
        #spikes_around_event.extend(np.extract((spiketrain.spike_times > event_time - before_Dt)*(spiketrain.spike_times <= event_time + after_Dt), spiketrain.spike_times) - event_time)

    #return np.sort(spikes_around_event, kind="quicksort"), nRepeats
