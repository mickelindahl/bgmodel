'''
Created on Apr 4, 2014

@author: lindahlm
'''

import numpy


#n_ = n_ * std::exp(-resolution / cp.tau_n_) + n_add_ / cp.tau_n_;

def dopa_trace(ts, n, dt, tau):
    n=1/tau
    trace=[]
    for n_add in ts:
        n=n*numpy.exp(-dt/tau)+n_add/tau
        trace.append(n)
    return trace

def poisson_spikes(stop_time, rate):
    import random
    ts=numpy.zeros(stop_time)
    t=0
    while 1:
        t+=random.expovariate(rate)*1000.0
        if t>=(stop_time-1):
            break
        ts[int(t)]+=1
    return ts

def estimate(dt, rate, tau):
    return 1./(1.-numpy.exp(-dt/tau)/tau)*rate/1000.0

if __name__ == '__main__':
    
    rate=1000
    n=10000*10
    dt=1.
    tau=100.
    v=estimate(dt, rate, tau)
    v2=rate/1000.0
    ts=poisson_spikes(n, rate)
    trace=dopa_trace(ts, n, dt, tau)
    
    import pylab
    pylab.plot(trace)
    pylab.title('Mean:{} S1:{} S2:{}'.format(numpy.mean(trace),v,v2))
    pylab.show()
    
    pass