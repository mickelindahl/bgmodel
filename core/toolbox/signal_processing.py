'''
Created on Jul 11, 2013

@author: lindahlm
'''
from __future__ import division
import numpy
from numpy import mean, hanning, cov
from numpy import zeros, ones, diagonal, transpose, matrix
from numpy import resize, sqrt, divide, array, concatenate
from numpy import convolve, dot, conjugate, absolute, arange, reshape #matrixmultiply, \
Float=numpy.float64 
Complex=numpy.complex 
     
from numpy.fft import fft

from scipy.signal import butter, lfilter, hilbert, filtfilt

from toolbox import misc
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
    method.  The vectors x and y are divided into NFFT length
    segments.  Each segment is detrended by function detrend and
    windowed by function window.  noverlap gives the length of the
    overlap between segments.  The product of the direct FFTs of x and
    y are averaged over each segment to compute Pxy, with a scaling to
    correct for power loss due to windowing.  fs is the sampling
    frequency.

    NFFT must be a power of 2

    Refs:
      Bendat & Piersol -- Random Data: Analysis and Measurement
        Procedures, John Wiley & Sons (1986)

    """
    window=kwargs.get('window',window_hanning )
    detrend=kwargs.get('detrend',detrend_none )
    
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
    freqs = fs/NFFT*arange(0,numFreqs)
    return Pxy, freqs

def cohere(x, y, NFFT=256, fs=2, noverlap=0, **kwargs):
    """
    cohere the coherence between x and y.  Coherence is the normalized
    cross spectral density

    Cxy = |Pxy|^2/(Pxx*Pyy)

    The return value is (Cxy, f), where f are the frequencies of the
    coherence vector.  See the docs for psd and csd for information
    about the function arguments NFFT, detrend, windowm noverlap, as
    well as the methods used to compute Pxy, Pxx and Pyy.

    """
    window=kwargs.get('window',window_hanning )
    detrend=kwargs.get('detrend',detrend_none )
    
    Pxx,f = psd(x, NFFT=NFFT, fs=fs, detrend=detrend,
              window=window, noverlap=noverlap)
    #print 'Pxx'
    Pyy,f = psd(y, NFFT=NFFT, fs=fs, detrend=detrend,
              window=window, noverlap=noverlap)
    #print 'Pyy'
    Pxy,f = csd(x, y, NFFT=NFFT, fs=fs, detrend=detrend,
              window=window, noverlap=noverlap)
    
    #print 'Pxy'
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


    l=[cohere(s1,s2, **kwargs) for s1, s2 in iter_double(signals1, signals2)]   
    l=numpy.array(l,dtype=float)
    Cxy, f=l[:,0,:], l[:,1,:]
    
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


def phase(x, lowcut, highcut, order,  fs, **kwargs):

    d={}
    d['raw']=x
    d['con']= misc.convolve(d['raw'], no_mean=True, **kwargs)
    
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

    return d['phase']

def phases(signals, *args, **kwargs):

    a=[phase(s,*args, **kwargs) for s in signals]
    return numpy.array(a)

def phase_diff(signal1, signal2, *args, **kwargs):
    
    inspect=kwargs.get('inspect', False)
    kwargs['inspect']=False
    a=phase(signal1,  *args, **kwargs)
    b=phase(signal2,  *args, **kwargs)

    x=a-b

    x[x>numpy.pi]=x[x>numpy.pi]-2*numpy.pi
    x[x<-numpy.pi]=x[x<-numpy.pi]+2*numpy.pi

     
    if inspect:
        bins=numpy.linspace(-numpy.pi,numpy.pi, kwargs.get('num',100.))
        pylab.hist(x, bins)  
        pylab.xlim((-numpy.pi, numpy.pi))  
        pylab.xlabel('Angle')
        pylab.ylabel('Occurance') 
        pylab.show() 
        
    return x

    
    

def phases_diff(signal1s, signal2s, *args, **kwargs):

    inspect=kwargs.get('inspect', False)
    kwargs['inspect']=False
    l=[phase_diff(s1, s2, *args, **kwargs) 
       for s1,s2 in iter_double(signal1s, signal2s)]
    
    x=numpy.array(l).ravel()
    
    if inspect:
        bins=numpy.linspace(-numpy.pi,numpy.pi, kwargs.get('num',100.))
        pylab.hist(x, bins)
        pylab.xlim((-numpy.pi, 2*numpy.pi))  
        pylab.xlabel('Angle')
        pylab.ylabel('Occurance') 
        pylab.show() 
    return x


def psd(x, NFFT=256, fs=2,  noverlap=0, normalize=True, **kwargs):
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
    -- normilize if true x/=numpy.mean(x)
    

    Refs:
      Bendat & Piersol -- Random Data: Analysis and Measurement
        Procedures, John Wiley & Sons (1986)

    """
    window=kwargs.get('window',window_hanning )
    detrend=kwargs.get('detrend',detrend_none )
    
    #No signal
    if not numpy.mean(x):
        return [[0],[0]]
            
    if normalize:
        x/=numpy.mean(x)
        
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
        
    windowVals = window(ones((NFFT,),x.dtype))
    step = NFFT-noverlap
    ind = range(0,len(x)-NFFT+1,step)
    n = len(ind)
    Pxx = zeros((numFreqs,n), Float)

    # do the ffts of the slices
    for i in range(n):
        thisX = x[ind[i]:ind[i]+NFFT]
        thisX = windowVals*detrend(thisX)
        fx = absolute(fft(thisX))**2
        Pxx[:,i] = fx[:numFreqs]

    # Scale the spectrum by the norm of the window to compensate for
    # windowing loss; see Bendat & Piersol Sec 11.5.2
    if n>1: Pxx = mean(Pxx,1)
    Pxx = divide(Pxx, norm(windowVals)**2)
    freqs = fs/NFFT*arange(0,numFreqs)
    
    if kwargs.get('inspect', False):
        pylab.subplot(111).plot(freqs[2:], Pxx[2:])    
        pylab.xlabel('freqs')
        pylab.ylabel('Pxx')
        #pylab.ylim((0,numpy.mean(Pxx)))
        pylab.show() 
    
    return Pxx, freqs


def window_hanning(x):
    return hanning(len(x))*x

import unittest


def dummy_data(**kwargs):
    n_events=100.0
    
    fs=kwargs.get('fs', 1000.0)
    scale=kwargs.get('scale',1)
    shift=kwargs.get('shift',0)
    sim_time=kwargs.get('sim_time', 2000.0)
    start=0
    stop=int(sim_time)
    
    n=numpy.random.randint(int(n_events*0.8), n_events)

    a=range(start, stop)
    numpy.random.shuffle(a)
    a=a[0:n]
    a=numpy.take(a, numpy.argsort(a))
    
    jitter=numpy.random.normal(loc=0, scale=scale, size=(len(a)))
    p_events=numpy.sin((a)*2*numpy.pi/50-numpy.pi*shift) + jitter
    
    a=a[p_events>0.3]

    hist, _=numpy.histogram(a,numpy.linspace(start, stop, 
                                             (stop-start+1)*fs/1000.0 ))

    return hist*fs
        
def dummy_data_pop(n_pop, **kwargs):
    a=[]
    for _ in xrange(n_pop):
        a.append(dummy_data(**kwargs))
    return numpy.array(a)

class Test_signal_processing(unittest.TestCase):
    
        
    def setUp(self):
        self.sim_time=1000.0
        self.n_pop=30
        

    def test_1_phase(self):
        fs=250.0
        lowcut, highcut, order,  fs,=10,20,3,fs
        kwargs={'bin_extent':10.,
                'inspect':False,
                'kernel_type':'gaussian',
                'params':{'std_ms':5.,
                          'fs': 1000.0}}
        a=dummy_data_pop(self.n_pop, **{'fs':fs,
                                        'sim_time':self.sim_time})
        mean_a=numpy.mean(a, axis=0)
        p1=phase(mean_a, lowcut, highcut, order,  fs, **kwargs)
        p2=phases(a, lowcut, highcut, order,  fs, **kwargs)  
        self.assertEqual(mean_a.shape, p1.shape)
        self.assertEqual(a.shape, p2.shape)      
        
    def test_2_pds(self):
        m=4
        kwargs={'fs':1000.,
                'numpyinspect':False,
                'NFFT':256*m,
                'noverlap':int(256/2*m),
                'normalize':True}
        a=dummy_data_pop(100, **{'scale':1,'sim_time':40000.0})

        mean_a=numpy.mean(a, axis=0)
        p,f=psd(mean_a, **kwargs)
        self.assertEqual(p.shape, f.shape)
      
    def test_3_coherences(self):
        m=2
        kwargs={'fs':1000,
                'inspect':False,
                'NFFT':256*m,
                'noverlap':int(256/2*m),}
        n_pop, sim_time=10, 5000.0 
        x=dummy_data_pop(n_pop, **{'scale':1,'sim_time':sim_time})
        y=dummy_data_pop(n_pop, **{'scale':1,'sim_time':sim_time,
                                   'shift':25})
        c1=coherences(x,y,**kwargs)
        #kwargs['inspect']=True
        c2=mean_coherence(x,y,**kwargs)
        
    def test_4_phase_shift(self):
        fs=1000.0
        lowcut, highcut, order,  fs,=15,25,3,fs
        kwargs={'bin_extent':10.,
                'kernel_type':'gaussian',
                'params':{'std_ms':5.,
                          'fs': fs}}
        
        n_pop, sim_time=50, 5000.0 
        x=dummy_data_pop(n_pop, **{'fs':fs,
                                   'scale':0.5,'sim_time':sim_time})
        y=dummy_data_pop(n_pop, **{'fs':fs,
                                   'scale':0.5,'sim_time':sim_time,
                                   'shift':0.})
        mean_x=numpy.mean(x, axis=0)
        
        mean_y=numpy.mean(y, axis=0)
        args= lowcut, highcut, order, fs
        kwargs['inspect']=False
        p1=phase_diff(mean_x, mean_y,  *args, **kwargs)
        n_pop, sim_time=5, 500.0 
        
        
        x=dummy_data_pop(n_pop, **{'fs':fs,
                                   'scale':0.5,'sim_time':sim_time})
        y=dummy_data_pop(n_pop, **{'fs':fs,
                                   'scale':0.5,'sim_time':sim_time,
                                   'shift':0.})
        kwargs['inspect']=False
        p2=phases_diff(x,y, *args, **kwargs)  

        
                     
if __name__ == '__main__':
    test_classes_to_run=[Test_signal_processing,
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
    