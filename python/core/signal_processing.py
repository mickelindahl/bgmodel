'''
Created on Jul 11, 2013

@author: lindahlm
'''
from __future__ import division
import numpy
import sys
from numpy import mean, hanning, cov
from numpy import zeros, ones, diagonal, transpose, matrix
from numpy import resize, sqrt, divide, array
from numpy import dot, conjugate, absolute, arange #matrixmultiply, \
Float=numpy.float64 
Complex=numpy.complex 
    

from numpy.fft import fft
from core.parallelization import map_parallel
from scipy.signal import butter, lfilter, hilbert, filtfilt
from os.path import expanduser

from core import misc
import pylab

def butter_bandpass(lowcut, highcut, fs, order=5):
    
    # Nyquist frequency is pi radians / sample.
    nyq = 0.5 * fs
    # Low and high are the normalized frequencies
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_lfilter(data, lowcut, highcut, fs, order=5):
    '''
    fs         - sampling frequency
    lowcut     - low cut frequency
    highcut    - high cut frequency
    '''
    
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    
    #y = filtfilt(b, a, data)
    return y

def butter_bandpass_filtfilt(data, lowcut, highcut, fs, order=5):
    '''
    fs         - sampling frequency
    lowcut     - low cut frequency
    highcut    - high cut frequency
    '''
    
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    y = lfilter(b, a, y[::-1])
    return y[::-1]

    #y = filtfilt(b, a, data)
    #return y
    

def csd(x, y, NFFT=256, fs=2, noverlap=0, **kwargs):
    """
    The cross spectral density Pxy by Welches average periodogram
    method. The vectors x and y are divided into NFFT length
    segments. Each segment is detrended by function detrend and
    windowed by function window.  noverlap gives the length of the
    overlap between segments.  The product of the direct FFTs of x and
    y are averaged over each segment to compute Pxy, with a scaling to
    correct for power loss due to windowing.  fs is the sampling
    frequency.

    (x-x_mean) for x in window 0,256

    Detrend (reomve) mean from traces avoids having high value at zero frequency.
    For two poisson process the the zero frequency of the spectrum will
    be the mean of the poisson process and the area under the spectrum
    equals the variance. (learned from Arvind)

    NFFT must be a power of 2

    Refs:
      Bendat & Piersol -- Random Data: Analysis and Measurement
        Procedures, John Wiley & Sons (1986)

    """
#     from scipy.signal import welch
    
    window=kwargs.get('window',window_hanning )
    detrend=kwargs.get('detrend',detrend_mean )
        
    if NFFT % 2:
        raise ValueError, 'NFFT must be a power of 2'

    # zero pad x and y up to NFFT if they are shorter than NFFT
    if len(x)<NFFT:
        n = len(x)
        x = resize(x, (NFFT,))
        x[n:] = 0
    if len(y)<NFFT:
        n = len(y)
        y = resize(y, (NFFT,))
        y[n:] = 0

    # for real x, ignore the negative frequencies
    if x.dtype==Complex: numFreqs = NFFT
    else: numFreqs = NFFT//2+1
    freqs = fs/NFFT*arange(0,numFreqs)    
    
    if not numpy.any(x-y):
        Pxy=numpy.zeros(freqs.shape)
    
    else:
        windowVals = window(ones((NFFT,),x.dtype))
        step = NFFT-noverlap
        ind = range(0,len(x)-NFFT+1,step)
        n = len(ind)
        Pxy = zeros((numFreqs,n), Complex)
    
  
        # do the ffts of the slices
        for i in range(n):
            thisX = x[ind[i]:ind[i]+NFFT]
            thisX = windowVals*detrend(thisX)
            thisY = y[ind[i]:ind[i]+NFFT]
            try:
                thisY = windowVals*detrend(thisY)
            except:
                print 'l'
            fx = fft(thisX)
            fy = fft(thisY)
            Pxy[:,i] = fy[:numFreqs]*conjugate(fx[:numFreqs])
    
        # Scale the spectrum by the norm of the window to compensate for
        # windowing loss; see Bendat & Piersol Sec 11.5.2
        if n>1: Pxy = mean(Pxy,1)
        Pxy = divide(Pxy, norm(windowVals)**2)
#         print Pxy.shape, x.shape
    
#     assert Pxy.shape==freqs.shape,'Shape differs'+str(Pxy.shape)+'_'+str(freqs.shape)
    
    return Pxy.ravel(), freqs

def cohere_wrap_parallel(args):
    return cohere(*args)

def cohere(x, y, kwargs):
    
    """
    cohere the coherence between x and y.  Coherence is the normalized
    cross spectral density

    Cxy = |Pxy|^2/(Pxx*Pyy)

    The return value is (Cxy, f), where f are the frequencies of the
    coherence vector.  See the docs for psd and csd for information
    about the function arguments NFFT, detrend, windowm noverlap, as
    well as the methods used to compute Pxy, Pxx and Pyy.

    Removing mean from traces avoids having high value at zero frequency.
    For two poisson process the the zero frequency of the spectrum will
    be the mean of the poisson process and the area under the spectrum
    equals the variance. (learned from Arvind)


    """

    fs=kwargs.get('fs', 2)
    NFFT=kwargs.get('NFFT', 256)
    noverlap=kwargs.get('noverlap', 0)
    
    window=kwargs.get('window',window_hanning )
    detrend=kwargs.get('detrend',detrend_mean )
    
    Pxx, f = psd(x, NFFT=NFFT, fs=fs, detrend=detrend,
              window=window, noverlap=noverlap)

    #print 'Pxx'
    Pyy, f = psd(y, NFFT=NFFT, fs=fs, detrend=detrend,
              window=window, noverlap=noverlap)
#     print Pyy[-1]
    Pxy, f = csd(x, y, NFFT=NFFT, fs=fs, detrend=detrend,
              window=window, noverlap=noverlap)
    
    #print 'Pxy'
#     print len(Pxy),  len(Pxx), len(Pyy),len(x), len(y)
    Cxy = divide(absolute(Pxy)**2, Pxx*Pyy)
    
    #Null nans
    Cxy[numpy.isnan(Cxy)]=0
    return Cxy, f

def corrcoef(*args):
    """
    
    corrcoef(X) where X is a matrix returns a matrix of correlation
    coefficients for each row of X.
    
    corrcoef(x,y) where x and y are vectors returns the matrix or
    correlation coefficients for x and y.

    Numeric arrays can be real or complex

    The correlation matrix is defined from the covariance matrix C as

    r(i,j) = C[i,j] / (C[i,i]*C[j,j])
    """

    if len(args)==2:
        X = transpose(array([args[0]]+[args[1]]))
    elif len(args==1):
        X = args[0]
    else:
        raise RuntimeError, 'Only expecting 1 or 2 arguments'

    
    C = cov(X)
#    d = resize(diagonal(C), (2,1))
    d = matrix(resize(diagonal(C), (2,1)))
    r = divide(C,sqrt(d*transpose(d)))[0,1]
    try: return r.real
    except AttributeError: return r

def coherences(signals1, signals2, **kwargs):


#     args=[[s1,s2, kwargs] 
#           for s1, s2 in iter_double(signals1, signals2) 
# #           if numpy.any(s1-s2)
#           ] 
    args=[[s1,s2, kwargs] 
          for s1, s2 in zip(signals1, signals2) 
#           if numpy.any(s1-s2)
          ]     
    

    args=zip(*args)
    try:
        l=map_parallel(cohere, *args, **{'local_num_threads': kwargs.get('local_num_threads',1)})
# 
    except Exception as e:
        s='\nTrying to do map_parallel in coherence'
        s+='\nSignal1: {}'.format(signals1)
        s+='\nSignal2: {}'.format(signals2)
        s+='\nkwargs: {}'.format(kwargs)
        raise type(e)(e.message + s), None, sys.exc_info()[2]
        
#     l=map(cohere, *args)
    try:
        l=numpy.array(l,dtype=float)
    except Exception as e:
        s='\nTrying to do map_parallel in coherence'
        s+='\nl: {}'.format(l)
#         s+='\nSignal2: {}'.format(signals2)
#         s+='\nkwargs: {}'.format(kwargs)
        raise type(e)(e.message + s), None, sys.exc_info()[2]
        
    Cxy, f=l[:,0,:], l[:,1,:]


    if kwargs.get('inspect', False):
        
        L=float(len(signals1[0])/kwargs.get('NFFT'))
        p_conf95=numpy.ones(len(f[0, 2:]))*(1-0.05**(1/(L-1)))  
        
        pylab.subplot(111).plot(numpy.transpose(f[0:3, 2:]), 
                                numpy.transpose(Cxy[0:3, 2:]))    
        pylab.subplot(111).plot(f[0, 2:], 
                                numpy.mean(Cxy[:, 2:], axis=0))
        pylab.subplot(111).plot(f[0, 2:], p_conf95, **{'color':'k'})
        pylab.xlabel('freqs')
        pylab.ylabel('Cxy')
        pylab.show() 

    
    return f, Cxy 

def detrend_mean(x):
    return x - mean(x)

def detrend_none(x):
    return x

def detrend_linear(x):
    """Remove the best fit line from x"""
    # I'm going to regress x on xx=range(len(x)) and return
    # x - (b*xx+a)
    xx = arange(len(x), typecode=x.dtype)
    X = transpose(array([xx]+[x]))
    C = cov(X)
    b = C[0,1]/C[0,0]
    a = mean(x) - b*mean(xx)
    return x-(b*xx+a)


def get_cross_spectral_density(data1, data2, sampling_freq=1000, 
                               NFFT=2048, diagonal=False, noverlap=0):
    results_Pxy=[]
    f=[]
    #samplingfreq=1000.0
    
    #NFFT determines how many frequencies coherence can be calculated for
    for i, d1 in enumerate(data1):
        for j, d2 in enumerate(data2):
            if not i==j or diagonal:
                Pxy, f=csd(d1,d2, NFFT=NFFT, fs=sampling_freq, noverlap=noverlap)
                results_Pxy.append(Pxy)
            
    mean_Pxy=numpy.mean(numpy.array(results_Pxy), axis=0) 
    return f, mean_Pxy 

def iter_double(l1,l2):
    for i in l1:
        for j in l2:
            yield i,j
            

def mean_coherence(signals1, signals2, **kwargs): 
    
      
    f, Cxy=coherences(signals1, signals2, **kwargs) 
    
    f, Cxy=numpy.mean(f, axis=0), numpy.mean(Cxy, axis=0)
     
    
    if kwargs.get('inspect', False):
        pylab.subplot(111).plot(f[2:], Cxy[2:])    
        pylab.xlabel('freqs')
        pylab.ylabel('Cxy')
        pylab.show() 
        
    return f, Cxy

def my_hilbert(x, N=None, axis=-1):
    return hilbert(x,N,axis)

def norm(x):
    return sqrt(dot(x,x))


def _phase(x, **kwargs):

    lowcut=kwargs.get('lowcut', 10)
    highcut=kwargs.get('highcut', 20)
    order=kwargs.get('order',3)
    fs=kwargs.get('fs', 1000.0) 
       
    d={}
    d['raw']=x
    
    # Don not need to convolve
    d['con']=x#misc.convolve(d['raw'], no_mean=True, **kwargs)
    
    call=butter_bandpass_lfilter
    d['bdp']=call(d['con'], lowcut, highcut, fs, order=order)
    
    call=butter_bandpass_filtfilt
    d['bdp_ff']=call(d['con'], lowcut, highcut, fs, order=order)
    
    d['phase']=numpy.angle(my_hilbert(d['bdp_ff']))
    d['phase_con']=numpy.angle(my_hilbert(d['con']))
    
    if kwargs.get('inspect', False):
        for i, key in enumerate(d.keys()):
            x=numpy.linspace(0, 1000.0*len(d[key])/fs, len(d[key]))
            pylab.subplot(6,1,i).plot(x, d[key])    
            pylab.ylabel(key)
        pylab.show() 

    return d

def phase(x, **kwargs):
    d=_phase(x, **kwargs)
    return d['phase']

def phases(signals, **kwargs):

    a=[phase(s, **kwargs) for s in signals]
    return numpy.array(a)

def phase_diff(signal1, signal2, kwargs):
    
    inspect=kwargs.get('inspect_phase_diff', False)
#     kwargs['inspect']=False

    rand=numpy.random.random()
    a=phase(signal1,  **kwargs)
    b=phase(signal2,  **kwargs)
    
    
    if rand>0.5:
        x=a-b
    else:
        x=b-a
    x[x>numpy.pi]=x[x>numpy.pi]-2*numpy.pi
    x[x<-numpy.pi]=x[x<-numpy.pi]+2*numpy.pi

#     inspect=True 
    if inspect:
        
        fig=pylab.figure(figsize=(30.,10.))
        fig.add_subplot(211)
        fig.add_subplot(212)
        fig.get_axes()[0].plot(a)
        fig.get_axes()[0].plot(b)
        
        bins=numpy.linspace(-numpy.pi,numpy.pi, kwargs.get('num',100.))
        fig.get_axes()[1].hist(x, bins)  
        pylab.xlim((-numpy.pi, numpy.pi))  
        pylab.xlabel('Angle')
        pylab.ylabel('Occurance') 
        pylab.show() 
        
    return x,a,b

    


def phases_diff(signals1, signals2, **kwargs):

    idx_filter=kwargs.get('idx_filter', None)
    inspect=kwargs.get('inspect_phases_diff', False)
    kwargs['inspect']=False
#     l=[phase_diff(s1, s2, *args, **kwargs) 
#        for s1,s2 in iter_double(signals1, signals2)]
#     
#     args0=[[s1,s2, kwargs] 
#           for s1, s2 in iter_double(signals1, signals2)
# #           if numpy.any(s1-s2)
#           ] 
    args0=[[s1,s2, kwargs] 
          for s1, s2 in zip(signals1, signals2)
#           if numpy.any(s1-s2)
          ]     
    if numpy.any(idx_filter):
        args0=[args0[idx] for idx in idx_filter]
        
    args=zip(*args0)
    
#     args=[[s1,s2, kwargs] for s1, s2 in iter_double(signals1, signals2)] 
#     args=zip(*args)
#     l=map_parallel(cohere, *args, **{'local_num_threads': kwargs.get('local_num_threads',1)})

    
    l=map_parallel(phase_diff, *args, **{'local_num_threads': kwargs.get('local_num_threads',1)})
    x=numpy.array(zip(*l)[0])
    a,b=numpy.array(zip(*l)[1:])
    #x=numpy.array(l).ravel()
#     inspect=True
    if inspect:
        
        bins=numpy.linspace(2.1*-numpy.pi, 2.1*numpy.pi, kwargs.get('num',100.))
        fig=pylab.figure(figsize=(30.,20.))
        k=1
        for i in range(5):
            for j in range(5):
                fig.add_subplot(5,5,k)
                k+=1
        i=0
        
        for s1,s2,e in zip(signals1, signals2, x):
            
            
           
#             print len(fig.get_axes())
            fig.get_axes()[i].hist(e.ravel(), bins)
            pylab.xlim((4.1*-numpy.pi, 4.1*numpy.pi))
            m1=numpy.mean(args0[i][0])
            m2=numpy.mean(args0[i][1])
            fig.get_axes()[i].set_title(str(m1)[0:3]+','+str(m2)[0:3]+',' +str(len(e)))

            pylab.title(str(len(e)))
            pylab.xlabel('Angle')
            pylab.ylabel('Occurance') 
            
            i+=1
            if i==23:
                break

        
        fig.get_axes()[i].hist(x.ravel(), bins, color='k')
        pylab.xlim((4.1*-numpy.pi, 4.1*numpy.pi)) 
        pylab.xlabel('Angle')
        pylab.ylabel('Occurance') 
        fig.get_axes()[i].set_title(str(len(x))+','+str(len(x.ravel())))
        
                
#         fig=pylab.figure(figsize=(30.,20.))
#         k=1
#         for i in range(5):
#             for j in range(5):
#                 fig.add_subplot(5,5,k)
#                 k+=1
#         i=0
# #         for s1,s2 in zip(a, b):
#         
#         bins=numpy.linspace(0,10000, 2000)
#         for s1,s2 in zip(signals1, signals2):
#             
#             fig.get_axes()[i].hist(s1, bins, histtype='step')
#             fig.get_axes()[i].hist(s2, bins, histtype='step')
# 
#             i+=1
#             if i==23:
#                 break

        fig=pylab.figure(figsize=(30.,20.))
        k=1
        for i in range(5):
            for j in range(5):
                fig.add_subplot(5,5,k)
                k+=1
        i=0
        for s1,s2, _ in args0:
#         for s1,s2 in zip(signals1, signals2):
             
            fig.get_axes()[i].plot(e)
#             fig.get_axes()[i].plot(b)
 
            i+=1
            if i==23:
                break
        pylab.show()
        

    

        
    return x


def psd(x, NFFT=256, fs=2,  noverlap=0,  **kwargs):
    """
    The power spectral density by Welches average periodogram method.
    The vector x is divided into NFFT length segments.  Each segment
    is detrended by function detrend and windowed by function window.
    noperlap gives the length of the overlap between segments.  The
    absolute(fft(segment))**2 of each segment are averaged to compute Pxx,
    with a scaling to correct for power loss due to windowing.  fs is
    the sampling frequency.

    -- NFFT must be a power of 2
    -- detrend and window are functions, unlike in matlab where they are
       vectors.
    -- if length x < NFFT, it will be zero padded to NFFT


    Detrend (removing) mean from traces avoids having high value at zero frequency.
    For two poisson process the the zero frequency of the spectrum will
    be the mean of the poisson process and the area under the spectrum
    equals the variance. (learned from Arvind)
    

    Refs:
      Bendat & Piersol -- Random Data: Analysis and Measurement
        Procedures, John Wiley & Sons (1986)

    """

    window=kwargs.get('window',window_hanning )
    detrend=kwargs.get('detrend',detrend_mean )
              
        
    if NFFT % 2:
        raise ValueError, 'NFFT must be a power of 2'

    # zero pad x up to NFFT if it is shorter than NFFT
    if len(x)<NFFT:
        n = len(x)
        x = resize(x, (NFFT,))
        x[n:] = 0
    

    # for real x, ignore the negative frequencies
    if x.dtype==Complex: numFreqs = NFFT
    else: numFreqs = NFFT//2+1
        
#     freqs = fs/NFFT*arange(0,numFreqs)

    #No signal
#     if not numpy.any(x):
#         Pxx=zeros(freqs.shape) 
#     else:       
    windowVals = window(ones((NFFT,),x.dtype))
    step = NFFT-noverlap
    ind = range(0,len(x)-NFFT+1,step)
    n = len(ind)
    Pxx = zeros((numFreqs,n), Float)
 
    # do the ffts of the slices
    for i in range(n):
#         pylab.subplot(511).plot(windowVals)
#         pylab.subplot(512).plot(x[ind[i]:ind[i]+NFFT])
#         pylab.subplot(513).plot(detrend(x[ind[i]:ind[i]+NFFT]))

        
        thisX = x[ind[i]:ind[i]+NFFT]
        thisX = windowVals*detrend(thisX)
        fx = absolute(fft(thisX))**2
        
#         pylab.subplot(514).plot(thisX)
#         pylab.subplot(515).plot( fs/NFFT*arange(0,len(thisX)), fx)
#         pylab.show()
        
        Pxx[:,i] = fx[:numFreqs]
 
    # Scale the spectrum by the norm of the window to compensate for
    # windowing loss; see Bendat & Piersol Sec 11.5.2
     
    if n>1: Pxx = mean(Pxx,1)
#     print len(Pxx)
    Pxx = divide(Pxx, norm(windowVals)**2)
    freqs = fs/NFFT*arange(0,numFreqs)
#     print len(Pxx)
 
 
    if kwargs.get('inspect', False):
        pylab.subplot(111).plot(freqs[2:], Pxx[2:])    
        pylab.xlabel('freqs')
        pylab.ylabel('Pxx')
        #pylab.ylim((0,numpy.mean(Pxx)))
        pylab.show() 
#         if Pxx.shape!=freqs.shape:
#             pylab.plot(x)
#             pylab.show()
#         assert Pxx.shape==freqs.shape,('Shape differs'
#                                        +str(Pxx.shape)+'_'
#                                        +str(freqs.shape)
#                                        +str(x)
#                                        +str(sum(x)))

    return Pxx.ravel(), freqs


def window_hanning(x):
    return hanning(len(x))*x

import unittest


def dummy_data(**kwargs):
    
    n_events_per_sec=kwargs.get('n_events_per_sec', 30.0)
    
    fs=kwargs.get('fs', 1000.0)
    scale=kwargs.get('scale',0.1)
    shift=kwargs.get('shift',0)
    sim_time=kwargs.get('sim_time', 2000.0)
    start=0
    stop=int(sim_time)
    n_events=int(n_events_per_sec*sim_time/1000.0)
    n=numpy.random.randint(int(n_events*0.2), n_events)

    a=range(start, stop)
    numpy.random.shuffle(a)
    a=a[0:n]
    a=numpy.take(a, numpy.argsort(a))
    
    jitter=numpy.random.normal(loc=0, scale=scale, size=(len(a)))
    p_events=(numpy.sin((a)*2*numpy.pi/50-numpy.pi*shift) + jitter
              +numpy.sin((a)*2*numpy.pi/43-numpy.pi*shift) )
    
    a=a[p_events>0.3]

    hist, _=numpy.histogram(a,numpy.linspace(start, stop, 
                                             (stop-start+1)*fs/1000.0 ))

    return hist*fs
        
def dummy_data_pop(n_pop, **kwargs):
    a=[]
    for _ in xrange(n_pop):
        a.append(dummy_data(**kwargs))
    return numpy.array(a)


def get_kwargs_cohere(m=1, fs=1000.0, n2=128):
    
    kwargs = {'fs':fs, 'inspect':False, 
        'NFFT':n2 * m, 
        'noverlap':int(n2 / 2 * m), 
        'local_num_threads':8}
    return kwargs

def get_kwargs_phase_diff(fs=1000.0):
    kwargs = {'lowcut':15, 'highcut':25, 
        'order':3, 
        'fs':fs, 
        'bin_extent':10., 
        'kernel_type':'gaussian', 
        'params':{'std_ms':5., 
            'fs':fs}}
    return kwargs

class Test_signal_processing(unittest.TestCase):
    
        
    def setUp(self):
        self.sim_time=10000.0
        self.n_pop=30
        
        self.home=expanduser("~")
        
    def test_1_phase(self):
        fs=1000.0
        lowcut, highcut, order,  fs,=10,30,3, fs
        kwargs={'lowcut':lowcut,
                'highcut':highcut,
                'order':order,
                'fs':fs,
                'bin_extent':10.,
                'inspect':True,
                'kernel_type':'gaussian',
                'params':{'std_ms':5.,
                          'fs': 1000.0}}
        a=dummy_data_pop(self.n_pop, **{'fs':fs,
                                        'sim_time':self.sim_time})
        mean_a=numpy.mean(a, axis=0)
        p1=phase(mean_a,  **kwargs)
        p2=phases(a,  **kwargs)  
 
        self.assertEqual(mean_a.shape, p1.shape)
        self.assertEqual(a.shape, p2.shape)      
          
    def test_2_pds(self):
        m=4
        kwargs={'fs':1000.,
                'numpyinspect':False,
                'NFFT':256*m,
                'noverlap':int(256/2*m),
                'normalize':True,
                'inspect':True}
        a=dummy_data_pop(100, **{'scale':1,'sim_time':40000.0})
  
        mean_a=numpy.mean(a, axis=0)
        p,f=psd(mean_a, **kwargs)
        self.assertEqual(p.shape, f.shape)
        
 
 
 
    def test_3_coherences(self):
        m=2
        fs=256
        #N=128 #for 1 Hz sampling
        N=128*8 #for 0.125 Hz sampling
        kwargs = get_kwargs_cohere(m, fs, N)
        n_pop, sim_time=25, 20000.0 
        x=dummy_data_pop(n_pop, **{'scale':0.01,'sim_time':sim_time,
                                   'local_num_threads':4, 'fs':fs})
        y=dummy_data_pop(n_pop, **{'scale':0.01,'sim_time':sim_time,
                                   'shift':0,'local_num_threads':4, 'fs':fs})
   
#         f, c1=coherences(x,y,**kwargs)
        
        kwargs['inspect']=True
           
        c2=mean_coherence(x,y,**kwargs)
               
  
  
 
    def test_4_phase_diff(self):
 
        fs=1000.0
        kwargs = get_kwargs_phase_diff(fs)
         
        n_pop, sim_time=50, 5000.0 
        x=dummy_data_pop(n_pop, **{'fs':fs,
                                   'scale':0.5,'sim_time':sim_time})
        y=dummy_data_pop(n_pop, **{'fs':fs,
                                   'scale':0.5,'sim_time':sim_time,
                                   'shift':0.})
 
        kwargs['inspect']=False
        p0=phase_diff(x[0,:], y[0,:],  kwargs)
        n_pop, sim_time=10, 500.0 
         
        mean_x=numpy.mean(x, axis=0)
         
        mean_y=numpy.mean(y, axis=0)
        kwargs['inspect']=False
        p1=phase_diff(mean_x, mean_y,  kwargs)
        n_pop, sim_time=10, 500.0 
         
         
        x=dummy_data_pop(n_pop, **{'fs':fs,
                                   'scale':0.5,'sim_time':sim_time})
        y=dummy_data_pop(n_pop, **{'fs':fs,
                                   'scale':0.5,'sim_time':sim_time,
                                   'shift':0.})
        kwargs['inspect']=False
        kwargs['local_num_threads']=8
        p2=phases_diff(x, y,  **kwargs)    


    def test_41_phase_diff_mpi(self):
        from core.data_to_disk import mkdir
        
        import pickle
        import os
        import subprocess
        fs=1000.0
        kwargs = get_kwargs_phase_diff(fs)
         
        n_pop, sim_time=10, 500.0 
        x=dummy_data_pop(n_pop, **{'fs':fs,
                                   'scale':0.5,'sim_time':sim_time})
        y=dummy_data_pop(n_pop, **{'fs':fs,
                                   'scale':0.5,'sim_time':sim_time,
                                   'shift':0.})
        kwargs['inspect']=False
        kwargs['local_num_threads']=2

        
        data_path= self.home+('/results/unittest/signal_processing'
                         +'/signal_processing_phase_diff_mpi/')
        script_name=os.getcwd()+('/test_scripts_MPI/'
                                 +'signal_processing_phase_diff_mpi.py')
        
        fileName=data_path+'data_in.pkl'
        fileOut=data_path+'data_out.pkl'
        mkdir(data_path)
        
        f=open(fileName, 'w') #open in binary mode 
        pickle.dump([x,y, kwargs], f, -1)
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
        kwargs['local_num_threads']=2*4
        p0=phases_diff(x, y,  **kwargs)   
        
        f=open(fileOut, 'rb') #open in binary mode
        p1=pickle.load(f)
        f.close()
        l0=numpy.round(p0.ravel(),2)
        l1=numpy.round(p1.ravel(),2)
        
        # abs since order of signal comparisons are random
        # se phase_diff
        l0,l1=list(numpy.abs(l0)), list(numpy.abs(l1))
        self.assertListEqual(l0,l1 )
        

    def test_4_coherence_and_phase_diff(self):
        fs=1000.0
        kwargs_phase_diff = get_kwargs_phase_diff(fs)
         
        m=2
        kwargs_cohere = get_kwargs_cohere(m)
         
        n_pop, sim_time=10, 10000.0 
        x=dummy_data_pop(n_pop, **{'fs':fs,
                                   'scale':0.5,
                                   'sim_time':sim_time})        
        y=dummy_data_pop(n_pop, **{'fs':fs,
                                   'scale':0.5,
                                   'sim_time':sim_time})     
         
   
         
        f, c1=coherences(x, y,**kwargs_cohere)
        idx, v =sort_coherences(f, c1, 19, 21)
         
        L=float(len(x[0])/kwargs_cohere.get('NFFT'))
        p_conf95=numpy.ones(len(f[0, 2:]))*(1-0.05**(1/(L-1)))
        p=[]
        p+=[phases_diff(x, y, idx_filter=idx[0:10], **kwargs_phase_diff).ravel()]
        p+=[phases_diff(x, y, idx_filter=idx[0:50], **kwargs_phase_diff).ravel()]
        p+=[phases_diff(x, y, idx_filter=idx[90:], **kwargs_phase_diff).ravel()]
        p+=[phases_diff(x, y, idx_filter=idx[v[idx]>p_conf95[0]], **kwargs_phase_diff).ravel()]
        p+=[phases_diff(x, y, **kwargs_phase_diff).ravel()]
        
        x*=0.
        y*=0.
#         p+=[phases_diff(x, y, idx_filter=idx[0:10], **kwargs_phase_diff).ravel()]
        if 1:
            for x in p:
                bins=numpy.linspace(-numpy.pi,numpy.pi, 100.0)
                pylab.hist(x, bins,**{'histtype':'step', 'normed':True} )
            pylab.xlim((-numpy.pi, 2*numpy.pi))  
            pylab.xlabel('Angle')
            pylab.ylabel('Occurance') 
            pylab.show() 
            pylab.plot()
#         
        
        
                      
def sort_coherences(f, c, low, high):
    v=(f>low)*(f<high)
    y=numpy.mean(c[:,v[0]],axis=1)
    idx=numpy.argsort(y)[::-1]
    return idx, y

    
    
               
if __name__ == '__main__':
    d={
        Test_signal_processing:[
#                                 'test_1_phase', 
#                                 'test_2_pds', 
#                                 'test_3_coherences', 
#                                 'test_4_phase_diff',
#                                 'test_41_phase_diff_mpi',
                                'test_4_coherence_and_phase_diff',
                                 ],
       }
    test_classes_to_run=d
    suite = unittest.TestSuite()
    for test_class, val in  test_classes_to_run.items():
        for test in val:
            suite.addTest(test_class(test))

    unittest.TextTestRunner(verbosity=2).run(suite)  
    